from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import numpy as np
import os, sys, time
import json

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.logging import setup_logging
import detectron.datasets.udb_datasets as udb_datasets
import detectron.utils.c2 as c2_utils

from detectron.modeling import model_builder
import detectron.utils.net as net_utils
from detectron.core.test_od_v2 import im_detect_all, video_detect_all

from detectron.app.common import print_log, divide_frames
import argparse

if sys.version_info[0] < 3:
  from Queue import Queue as queue
else:
  from queue import Queue as queue
  from builtins import str

if cv2.__version__.startswith('2.'):
  CV_LOAD_IMAGE_COLOR = cv2.CV_LOAD_IMAGE_COLOR
else:
  CV_LOAD_IMAGE_COLOR = cv2.IMREAD_COLOR


def parse_args():
  parser = argparse.ArgumentParser(description='Auto-labeling tool')
  parser.add_argument(
    '--yaml',
    dest='yaml',
    help=u'path to model\'s yaml file',
    default=None,
    type=str
  )
  parser.add_argument(
    '--key',
    dest='key',
    help=u'decryption key (default: None)',
    default=None,
    type=str
  )
  parser.add_argument(
    '--images_dir',
    dest='images_dir',
    help=u'directory where images are located',
    default=None,
    type=str
  )
  parser.add_argument(
    '--out_images_dir',
    dest='out_images_dir',
    help=u'directory where output images are located',
    default=None,
    type=str
  )
  parser.add_argument(
    '--out_xmls_dir',
    dest='out_xmls_dir',
    help=u'directory where output xmls are located',
    default=None,
    type=str
  )
  parser.add_argument(
    '--threads',
    dest='threads',
    help=u'number of threads',
    default=None,
    type=int
  )
  parser.add_argument(
    '--thread_id',
    dest='thread_id',
    help=u'thread id',
    default=None,
    type=int
  )
  parser.add_argument(
    '--video',
    dest='video',
    help='if images are consecutive frames of a video',
    action='store_true'
  )
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()


def initialize_model_from_cfg(weights_file, gpu_id=0, key=None):
    """Initialize a model from the global cfg. Loads test-time weights and
    creates the networks in the Caffe2 workspace.
    """
    model = model_builder.create(cfg.MODEL.TYPE, train=False, gpu_id=gpu_id)
    net_utils.initialize_gpu_from_weights_file(
        model, weights_file, gpu_id=gpu_id, key=key
    )
    model_builder.add_inference_inputs(model)
    workspace.CreateNet(model.net)
    workspace.CreateNet(model.conv_body_net)
    if cfg.MODEL.MASK_ON:
        workspace.CreateNet(model.mask_net)
    if cfg.MODEL.KEYPOINTS_ON:
        workspace.CreateNet(model.keypoint_net)
    workspace.CreateNet(model.rcnn_net)
    return model


def initialize(yaml_path, model_key):
  c2_utils.import_detectron_ops()
  cv2.ocl.setUseOpenCL(False)

  #workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
  #setup_logging(__name__)
  workspace.GlobalInit(['caffe2'])

  merge_cfg_from_file(yaml_path)
  cfg.NUM_GPUS = 1
  assert_and_infer_cfg(cache_urls=False)

  assert not cfg.MODEL.RPN_ONLY, \
    'RPN models are not supported'
  assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
    'Models that require precomputed proposals are not supported'

  model = initialize_model_from_cfg(cfg.TEST.WEIGHTS, key=model_key)
  dataset = udb_datasets.get_dataset(cfg.MODEL.CLASSSET, udb_path=None)

  return model, dataset


def process_od(model, args, dataset):
  im_names = [f for f in os.listdir(args.images_dir) if len(f) > 4 and f[-4:] in ['.jpg', '.png']]
  im_names.sort()
  if args.thread_id is not None:
    start_ids, end_ids = divide_frames(len(im_names), chunks=args.threads)
    im_names = im_names[start_ids[args.thread_id]:end_ids[args.thread_id]+1]
  total_count = len(im_names)

  if args.out_images_dir is not None:
    os.system('mkdir -p {}'.format(args.out_images_dir))
  if args.out_xmls_dir is not None:
    os.system('mkdir -p {}'.format(args.out_xmls_dir))

  if args.video:
    prev_anns = None
    for count, im_name in enumerate(im_names):
      if (count + 1) % 100 == 0:
        if args.thread_id is not None:
          print('[{}][{}/{}] {}'.format(args.thread_id, count + 1, total_count, im_name))
        else:
          print('[{}/{}] {}'.format(count + 1, total_count, im_name))

      file_path = os.path.join(args.images_dir, im_name)
      out_anns = compute_video_od(model, file_path, prev_anns)
      prev_anns = out_anns

      if args.out_images_dir is not None:
        write_img(args.images_dir, args.out_images_dir, im_name, out_anns)
      if args.out_xmls_dir is not None:
        write_xml(dataset, args.images_dir, args.out_xmls_dir, im_name, out_anns)

  else:
    for count, im_name in enumerate(im_names):
      if (count + 1) % 100 == 0:
        if args.thread_id is not None:
          print('[{}][{}/{}] {}'.format(args.thread_id, count + 1, total_count, im_name))
        else:
          print('[{}/{}] {}'.format(count + 1, total_count, im_name))

      file_path = os.path.join(args.images_dir, im_name)
      out_anns = compute_image_od(model, file_path)

      if args.out_images_dir is not None:
        write_img(args.images_dir, args.out_images_dir, im_name, out_anns)
      if args.out_xmls_dir is not None:
        write_xml(dataset, args.images_dir, args.out_xmls_dir, im_name, out_anns)

  if args.thread_id is not None:
    print('[{}] Finished'.format(args.thread_id))
  else:
    print('Finished')


def write_img(images_dir, output_dir, im_name, anns):
    file_path = os.path.join(images_dir, im_name)
    img = cv2.imread(file_path, CV_LOAD_IMAGE_COLOR)
    for ann in anns:
      x1, y1, x2, y2, score, obj_score, pred_label = ann
      x1, y1, x2, y2 = [int(round(val)) for val in [x1, y1, x2, y2]]
      cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
      cv2.putText(img, str(pred_label), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    output_path = os.path.join(output_dir, im_name)
    cv2.imwrite(output_path, img)


def get_filename_wo_ext(path):
  name_wo_path = path.strip().split('/')[-1]
  name_wo_ext = '.'.join(name_wo_path.split('.')[:-1])
  return name_wo_ext


def write_xml(dataset, images_dir, output_dir, im_name, anns):
  file_path = os.path.join(images_dir, im_name)
  img = cv2.imread(file_path, CV_LOAD_IMAGE_COLOR)
  height, width = img.shape[:2]

  output_path = os.path.join(output_dir, get_filename_wo_ext(im_name) + '.xml')
  xml_strs = ['<annotation>\n<size><width>{}</width>\n<height>{}</height></size>'.format(width, height)]
  for ann in anns:
    x1, y1, x2, y2, score, obj_score, pred_label = ann
    name = dataset.classes[pred_label] if dataset is not None else str(pred_label)
    str_obj = '<object><name>{}</name><bndbox><xmin>{:.2f}</xmin><ymin>{:.2f}</ymin><xmax>{:.2f}</xmax><ymax>{:.2f}</ymax></bndbox></object>'.format(name, x1, y1, x2, y2)
    xml_strs.append(str_obj)
  xml_strs.append('</annotation>\n')
  with open(output_path, 'w') as f:
    f.write('\n'.join(xml_strs))


def compute_image_od(model, file_path):
  img = cv2.imread(file_path, CV_LOAD_IMAGE_COLOR)
  with c2_utils.NamedCudaScope(0):
    cls_boxes = im_detect_all(model, img)
  anns = pack_bbox(cls_boxes)
  return anns


def compute_video_od(model, file_path, prev_anns):
  if prev_anns is not None:
    prev_anns = np.array(prev_anns, dtype=np.float32)[:, :4] if len(prev_anns) > 0 else None

  img = cv2.imread(file_path, CV_LOAD_IMAGE_COLOR)
  with c2_utils.NamedCudaScope(0):
    cls_boxes, cls_tracks = video_detect_all(model, img, prev_anns)
  anns = pack_bbox(cls_boxes, cls_tracks=cls_tracks)
  return anns


def pack_bbox(cls_boxes, th_w=5, th_h=5, cls_tracks=None):
  anns = []
  for label, bboxes in enumerate(cls_boxes):
    if label == 0:
      continue

    if cls_tracks is not None:
      for bbox, track in zip(bboxes, cls_tracks[label]):
        x1, y1, x2, y2, score, uncertainty, bg_score = bbox
        obj_score = 1.0 - bg_score
        accepted = (obj_score >= cfg.TEST.FINAL_SCORE_THRESH) \
            or (track > 0 and obj_score >= cfg.TEST.TRACK_FINAL_SCORE_THRESH)
        if accepted and x2 - x1 + 1 >= th_w and y2 - y1 + 1 >= th_h:
          anns.append([x1, y1, x2, y2, score, obj_score, label])

    else:
      for bbox in bboxes:
        x1, y1, x2, y2, score, uncertainty, bg_score = bbox
        obj_score = 1.0 - bg_score
        accepted = (obj_score >= cfg.TEST.FINAL_SCORE_THRESH)
        if accepted and x2 - x1 + 1 >= th_w and y2 - y1 + 1 >= th_h:
          anns.append([x1, y1, x2, y2, score, obj_score, label])

  return anns


def main():
  #try:
    args = parse_args()
    model, dataset = initialize(args.yaml, args.key)
    process_od(model, args, dataset)

  #except KeyboardInterrupt as e:
  #  pass
  #except Exception as e:
  #  pass
