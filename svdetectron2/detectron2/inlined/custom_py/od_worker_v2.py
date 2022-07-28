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

from detectron.app.common import print_log, connect_to_server, recv_large, send_large, prepare_file, post_anns, report_progress, report_error, report_finish, get_video_frames_opencv

if sys.version_info[0] < 3:
  from Queue import Queue as queue
else:
  from queue import Queue as queue
  from builtins import str

if cv2.__version__.startswith('2.'):
  CV_LOAD_IMAGE_COLOR = cv2.CV_LOAD_IMAGE_COLOR
else:
  CV_LOAD_IMAGE_COLOR = cv2.IMREAD_COLOR


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


def initialize(model_path):
  c2_utils.import_detectron_ops()
  cv2.ocl.setUseOpenCL(False)

  #workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
  #setup_logging(__name__)
  workspace.GlobalInit(['caffe2'])

  cfg.MODEL.TYPE = b'generalized_rcnn'
  cfg.MODEL.CONV_BODY = b'FPN.add_fpn_ResNet152_conv5_body'
  cfg.MODEL.NUM_CLASSES = 9
  cfg.MODEL.FASTER_RCNN = True
  cfg.MODEL.MASK_ON = False
  cfg.MODEL.IGNORED_CLASS_INDEX = 8
  cfg.FPN.FPN_ON = True
  cfg.FPN.MULTILEVEL_ROIS = True
  cfg.FPN.MULTILEVEL_RPN = True
  cfg.RESNETS.STRIDE_1X1 = False
  cfg.RESNETS.TRANS_FUNC = b'bottleneck_transformation'
  cfg.RESNETS.NUM_GROUPS = 32
  cfg.RESNETS.WIDTH_PER_GROUP = 8
  cfg.FAST_RCNN.ROI_BOX_HEAD = b'fast_rcnn_heads.add_roi_2mlp_head'
  cfg.FAST_RCNN.ROI_XFORM_METHOD = b'RoIAlign'
  cfg.FAST_RCNN.ROI_XFORM_RESOLUTION = 7
  cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO = 2
  cfg.TEST.SCALE = 900
  cfg.TEST.MAX_SIZE = 1700
  cfg.TEST.NMS = 0.5
  cfg.TEST.CLASS_AGNOSTIC_NMS = 0.75
  cfg.TEST.TRACK_RESIZE_RATIO = 1.1
  cfg.TEST.TRACK_IOU_THRESH = 0.5
  cfg.TEST.TRACK_FINAL_SCORE_THRESH = 0.5
  cfg.TEST.BBOX_VOTE.ENABLED = True
  cfg.TEST.BBOX_VOTE.VOTE_TH = 0.7
  cfg.TEST.BBOX_VOTE.SCORING_METHOD = b'ID'
  cfg.TEST.RPN_PRE_NMS_TOP_N = 1000
  cfg.TEST.RPN_POST_NMS_TOP_N = 1000
  cfg.TEST.DETECTIONS_PER_IM = 1000
  cfg.TEST.SCORE_THRESH = 0.1
  cfg.TEST.FINAL_SCORE_THRESH = 0.85
  cfg.TEST.BBOX_AUG.ENABLED = True
  cfg.TEST.BBOX_AUG.SCORE_HEUR = b'UNION'
  cfg.TEST.BBOX_AUG.COORD_HEUR = b'UNION'
  cfg.TEST.BBOX_AUG.H_FLIP = False
  cfg.TEST.BBOX_AUG.SCALES = (700, 800,)
  cfg.TEST.BBOX_AUG.MAX_SIZE = 1900
  cfg.TEST.BBOX_AUG.SCALE_H_FLIP = False
  cfg.TEST.BBOX_AUG.SCALE_SIZE_DEP = False
  cfg.TEST.BBOX_AUG.AREA_TH_LO = 225
  cfg.TEST.BBOX_AUG.AREA_TH_HI = 640000
  cfg.TEST.BBOX_AUG.ASPECT_RATIOS = ()
  cfg.TEST.BBOX_AUG.ASPECT_RATIO_H_FLIP = False

  cfg.NUM_GPUS = 1
  assert_and_infer_cfg(cache_urls=False)

  assert not cfg.MODEL.RPN_ONLY, \
    'RPN models are not supported'
  assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
    'Models that require precomputed proposals are not supported'

  model_key = 'a8b7c6d5e4f3g2h1'
  model = initialize_model_from_cfg(model_path, key=model_key)
  dataset = udb_datasets.get_dataset('god_2018_8cls', udb_path=None)

  return model, dataset


def process_task_video_od(model, dataset, conn, task_data, file_path):
  start_frame_id = task_data['start_frame_id']
  end_frame_id = task_data['end_frame_id']
  label_interval = task_data['label_interval']

  all_anns = {}
  prev_anns = None
  for (frame_id, img) in get_video_frames_opencv(file_path, start_frame_id, end_frame_id, 1):
    anns = compute_video_bbox(model, dataset, img, prev_anns)
    prev_anns = anns
    #if frame_id % label_interval != 0:
    #  continue
    all_anns[frame_id] = anns
    report_progress(conn, task_data, frame_id + 1)
  post_anns(conn, task_data, all_anns)


def process_task_image_od(model, dataset, conn, task_data, file_path):
  img = cv2.imread(file_path, CV_LOAD_IMAGE_COLOR)
  anns = compute_image_bbox(model, dataset, img)
  post_anns(conn, task_data, anns)


def process_od(model, dataset, conn, media_root):
  req_data = { 'req_type': 'get_task', 'task_type': 'od' }
  recv_data = {}

  while 'terminate' not in recv_data:
    try:
      send_large(conn, req_data)
      recv_data = recv_large(conn)

      if recv_data is None:
        break

      if 'task' not in recv_data:
        time.sleep(1)
        continue

      task_data = recv_data['task']
      file_path = prepare_file(conn, media_root, task_data)
      if file_path is None:
        report_error(conn, task_data, 'file_not_found')
        continue

      if task_data['file_type'] == 'video':
        process_task_video_od(model, dataset, conn, task_data, file_path)
      else:
        process_task_image_od(model, dataset, conn, task_data, file_path)

      report_finish(conn, task_data)

    except Exception as e:
      print_log('process',
          'Error occurred from recv_data [{}]: {}'.format(recv_data, repr(e)))
      time.sleep(5)
      continue


def compute_image_bbox(model, dataset, img):
  if img is None:
    return {}

  with c2_utils.NamedCudaScope(0):
    cls_boxes = im_detect_all(model, img)
  anns = pack_bbox(dataset, cls_boxes)
  return anns


def compute_video_bbox(model, dataset, img, prev_anns):
  if img is None:
    return {}

  if prev_anns is None:
    prev_boxes = None
  else:
    prev_boxes = [np.array(ann) for ann in prev_anns.values() if len(ann) > 0]
    prev_boxes = np.vstack(prev_boxes).astype(np.float32)[:, :4] if len(prev_boxes) > 0 else None  # safe

  with c2_utils.NamedCudaScope(0):
    cls_boxes, cls_tracks = video_detect_all(model, img, prev_boxes)
  anns = pack_bbox(dataset, cls_boxes, cls_tracks=cls_tracks)
  return anns


def pack_bbox(dataset, cls_boxes, th_w=5, th_h=5, cls_tracks=None):
  anns = {}
  for class_idx, bboxes in enumerate(cls_boxes):
    if class_idx == 0:
      continue

    class_ann = []
    if cls_tracks is not None:
      for bbox, track in zip(bboxes, cls_tracks[class_idx]):
        x1, y1, x2, y2 = [int(round(v)) for v in bbox[:4]]
        score, uncertainty, bg_score = [round(val * 1000) / 1000 for val in bbox[4:]]
        obj_score = 1.0 - bg_score
        accepted = (obj_score >= cfg.TEST.FINAL_SCORE_THRESH) \
            or (track > 0 and obj_score >= cfg.TEST.TRACK_FINAL_SCORE_THRESH)
        if accepted and x2 - x1 + 1 >= th_w and y2 - y1 + 1 >= th_h:
          class_ann.append([x1, y1, x2, y2, obj_score])
    else:
      for bbox in bboxes:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox[:4]]
        score, uncertainty, bg_score = [round(val * 1000) / 1000 for val in bbox[4:]]
        obj_score = 1.0 - bg_score
        accepted = (obj_score >= cfg.TEST.FINAL_SCORE_THRESH)
        if accepted and x2 - x1 + 1 >= th_w and y2 - y1 + 1 >= th_h:
          class_ann.append([x1, y1, x2, y2, obj_score])

    if len(class_ann) > 0:
      class_name = dataset.classes[class_idx]
      if class_name not in anns:
        anns[class_name] = class_ann
      else:
        anns[class_name].extend(class_ann)

  return anns


def main(cfg_path):
  #try:
    cfg = json.loads(open(cfg_path, 'r').read())
    conn = connect_to_server(cfg['server']['host'], cfg['server']['port'])
    model, dataset = initialize(cfg['od']['path'])
    process_od(model, dataset, conn, cfg['media_root'])
    conn.close()

  #except KeyboardInterrupt as e:
  #  pass
  #except Exception as e:
  #  pass
