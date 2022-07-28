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
import detectron.utils.c2 as c2_utils

from detectron.modeling import model_builder
import detectron.utils.net as net_utils
from detectron.core.test_od_v2 import im_detect_all, compute_iou
from detectron.core.test_od import box_results_with_argmax

from detectron.app.common import print_log
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
  parser = argparse.ArgumentParser(description='Auto-labeling false positive recognition tool')
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
    '--data_root',
    dest='data_root',
    help='root directory of the dataset',
    default=None,
    type=str
  )
  parser.add_argument(
    '--gt_txt',
    dest='gt_txt',
    help=u'filename of txt-formatted ground-truth',
    default=None,
    type=str
  )
  parser.add_argument(
    '--images_dir',
    dest='images_dir',
    help=u'subfolder where images are located (default: images)',
    default=u'images',
    type=str
  )
  parser.add_argument(
    '--out_txt',
    dest='out_txt',
    help=u'filename of txt-formatted FPR output',
    default=None,
    type=str
  )
  parser.add_argument(
    '--out_images_dir',
    dest='out_images_dir',
    help=u'subfolder where output images are located',
    default=None,
    type=str
  )
  parser.add_argument(
    '--fp_label',
    dest='fp_label',
    help=u'label index of false positive class (default: 21)',
    default=21,
    type=int
  )
  parser.add_argument(
    '--fp_iou_thres',
    dest='fp_iou_thres',
    help=u'IoU threshold between ground-truth boxes and auto-labeled boxes to determine false positives (default: 0.5)',
    default=0.5,
    type=float
  )
  parser.add_argument(
    '--threads',
    dest='threads',
    help=u'number of threads (default: 5)',
    default=8,
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
    '--windows_newline',
    dest='windows_newline',
    help='append carriage return to newline',
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

  return model


def process_fpr(model, args):
  gt_txt_path = os.path.join(args.data_root, args.gt_txt)
  images_dir = os.path.join(args.data_root, args.images_dir)
  gt_anns_all = load_gt(gt_txt_path)
  gt_anns_keys = gt_anns_all.keys()
  gt_anns_keys.sort()
  out_anns_all = {}
  total_count = len(gt_anns_all)
  for count, im_name in enumerate(gt_anns_keys):
    if (count + 1) % 100 == 0:
      if args.thread_id is not None:
        print('[{}][{}/{}] {}'.format(args.thread_id, count + 1, total_count, im_name))
      else:
        print('[{}/{}] {}'.format(count + 1, total_count, im_name))

    if args.thread_id is not None and count % args.threads != args.thread_id:
      continue
    file_path = os.path.join(images_dir, im_name)
    out_anns = compute_image_fpr(model, file_path, gt_anns_all[im_name], args.fp_label, args.fp_iou_thres)
    out_anns_all[im_name] = out_anns

  if args.out_txt is not None:
    newline = '\r\n' if args.windows_newline else '\n'
    out_txt_path = os.path.join(args.data_root, args.out_txt)
    if args.thread_id is not None:
      out_txt_path += '.{}'.format(args.thread_id)
    write_anns(out_txt_path, out_anns_all, newline)

  if args.out_images_dir is not None:
    write_img(images_dir, args.out_images_dir, out_anns_all, args.fp_label)

  if args.thread_id is not None:
    print('[{}] Finished'.format(args.thread_id))
  else:
    print('Finished')


def load_gt(gt_txt_path):
  lines = [line.strip() for line in open(gt_txt_path, 'r')]
  num_lines = len(lines)
  gt_anns_all = {}
  count = 0
  while count < num_lines:
    im_name = lines[count]
    num_gts = int(lines[count + 1])
    gt_anns = []
    for i in range(num_gts):
      tokens = lines[count + 2 + i].split(' ')
      gt_ann = [float(val) for val in tokens[1:5]] + [int(tokens[0])]
      gt_anns.append(gt_ann)
    gt_anns_all[im_name] = gt_anns
    count += (num_gts + 2)

    if count % 10000 == 0:
      print('reading {}/{} lines... {}'.format(count, num_lines, im_name))

  return gt_anns_all


def write_img(images_dir, output_dir, anns_all, fp_label):
  os.system('mkdir -p {}'.format(output_dir))
  for im_name, anns in anns_all.items():
    file_path = os.path.join(images_dir, im_name)
    img = cv2.imread(file_path, CV_LOAD_IMAGE_COLOR)
    for ann in anns:
      x1, y1, x2, y2, label, score, obj_score, pred_label = ann
      x1, y1, x2, y2 = [int(round(val)) for val in [x1, y1, x2, y2]]
      if score < 0.7 and label != fp_label:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
      elif label == fp_label:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    output_path = os.path.join(output_dir, im_name)
    cv2.imwrite(output_path, img)


def write_anns(txt_path, anns_all, newline):
  out_strs = []
  for im_name, anns in anns_all.items():
    out_strs.append(im_name)
    out_strs.append('{:d}'.format(len(anns)))
    for ann in anns:
      x1, y1, x2, y2, label, score, obj_score, pred_label = ann
      out_strs.append('{:d} {:.2f} {:.2f} {:.2f} {:.2f} {:d} \"{:.2f}\"'.format(label, x1, y1, x2, y2, pred_label, score))

  with open(txt_path, 'w') as f:
    f.write(newline.join(out_strs) + newline)


def compute_image_fpr(model, file_path, gt_anns, fp_label, fp_iou_thres):
  img = cv2.imread(file_path, CV_LOAD_IMAGE_COLOR)

  if gt_anns is not None and len(gt_anns) > 0:
    gt_boxes = np.array(gt_anns, dtype=np.float32)[:, :4]
    with c2_utils.NamedCudaScope(0):
      cls_boxes, gt_pred_scores, gt_pred_boxes = im_detect_all(model, img, ex_boxes=gt_boxes)
      scores, _, labels = box_results_with_argmax(gt_pred_scores, gt_pred_boxes)
      obj_scores = 1.0 - gt_pred_scores[:, 0]
      gt_anns = [[x1, y1, x2, y2, gt_label, score, obj_score, label] for (x1, y1, x2, y2, gt_label), score, obj_score, label in zip(gt_anns, scores, obj_scores, labels)]
      anns = pack_bbox(cls_boxes, fp_label)
      anns = merge_bbox(anns, gt_anns, fp_iou_thres) if len(anns) > 0 else gt_anns

  else:
    with c2_utils.NamedCudaScope(0):
      cls_boxes = im_detect_all(model, img)
      anns = pack_bbox(cls_boxes, fp_label)

  return anns


def pack_bbox(cls_boxes, fp_label, th_w=5, th_h=5):
  anns = []
  for label, bboxes in enumerate(cls_boxes):
    if label == 0:
      continue

    for bbox in bboxes:
      x1, y1, x2, y2, score, uncertainty, bg_score = bbox
      obj_score = 1.0 - bg_score
      accepted = (obj_score >= cfg.TEST.FINAL_SCORE_THRESH)
      if accepted and x2 - x1 + 1 >= th_w and y2 - y1 + 1 >= th_h:
        anns.append([x1, y1, x2, y2, fp_label, score, obj_score, label])

  return anns


def merge_bbox(anns, gt_anns, fp_iou_thres):
  boxes = np.array(anns, dtype=np.float32)[:, :4]
  gt_boxes = np.array(gt_anns, dtype=np.float32)[:, :4]
  ious = compute_iou(gt_boxes, boxes)
  fp_anns = [ann for ann, cond in zip(anns, ious.max(axis=0) < fp_iou_thres) if cond]
  return gt_anns + fp_anns


def main():
  #try:
    args = parse_args()
    model = initialize(args.yaml, args.key)
    process_fpr(model, args)

  #except KeyboardInterrupt as e:
  #  pass
  #except Exception as e:
  #  pass
