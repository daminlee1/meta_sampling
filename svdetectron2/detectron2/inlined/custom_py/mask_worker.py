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
from detectron.core.test_mask import im_detect_mask_all

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
  cfg.MODEL.NUM_CLASSES = 81
  cfg.MODEL.FASTER_RCNN = True
  cfg.MODEL.MASK_ON = True
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
  cfg.MRCNN.ROI_MASK_HEAD = b'mask_rcnn_heads.mask_rcnn_fcn_head_v1up4convs'
  cfg.MRCNN.RESOLUTION = 28
  cfg.MRCNN.ROI_XFORM_METHOD = b'RoIAlign'
  cfg.MRCNN.ROI_XFORM_RESOLUTION = 14
  cfg.MRCNN.ROI_XFORM_SAMPLING_RATIO = 2
  cfg.MRCNN.DILATION = 1
  cfg.MRCNN.CONV_INIT = b'MSRAFill'
  cfg.TEST.SCALE = 800
  cfg.TEST.MAX_SIZE = 1333
  cfg.TEST.NMS = 0.5
  cfg.TEST.BBOX_VOTE.ENABLED = True
  cfg.TEST.BBOX_VOTE.VOTE_TH = 0.9
  cfg.TEST.RPN_PRE_NMS_TOP_N = 1000
  cfg.TEST.RPN_POST_NMS_TOP_N = 1000
  cfg.TEST.BBOX_AUG.ENABLED = False
  cfg.TEST.BBOX_AUG.SCORE_HEUR = b'UNION'
  cfg.TEST.BBOX_AUG.COORD_HEUR = b'UNION'
  cfg.TEST.BBOX_AUG.H_FLIP = True
  cfg.TEST.BBOX_AUG.SCALES = (400, 500, 600, 700, 900, 1000, 1100, 1200)
  cfg.TEST.BBOX_AUG.MAX_SIZE = 2000
  cfg.TEST.BBOX_AUG.SCALE_H_FLIP = True
  cfg.TEST.BBOX_AUG.SCALE_SIZE_DEP = False
  cfg.TEST.BBOX_AUG.ASPECT_RATIOS = ()
  cfg.TEST.BBOX_AUG.ASPECT_RATIO_H_FLIP = False
  cfg.TEST.MASK_AUG.ENABLED = False
  cfg.TEST.MASK_AUG.HEUR = b'SOFT_AVG'
  cfg.TEST.MASK_AUG.H_FLIP = True
  cfg.TEST.MASK_AUG.SCALES = (400, 500, 600, 700, 900, 1000, 1100, 1200)
  cfg.TEST.MASK_AUG.MAX_SIZE = 2000
  cfg.TEST.MASK_AUG.SCALE_H_FLIP = True
  cfg.TEST.MASK_AUG.SCALE_SIZE_DEP = False
  cfg.TEST.MASK_AUG.ASPECT_RATIOS = ()
  cfg.TEST.MASK_AUG.ASPECT_RATIO_H_FLIP = False

  cfg.NUM_GPUS = 1
  assert_and_infer_cfg(cache_urls=False)

  assert not cfg.MODEL.RPN_ONLY, \
    'RPN models are not supported'
  assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
    'Models that require precomputed proposals are not supported'

  model_key = 'a8b7c6d5e4f3g2h1'
  model = initialize_model_from_cfg(model_path, key=model_key)
  dataset = udb_datasets.get_dataset('god_2018_coco', udb_path=None)

  return model, dataset


def process_task_video_mask(model, dataset, conn, task_data, file_path):
  start_frame_id = task_data['start_frame_id']
  end_frame_id = task_data['end_frame_id']
  label_interval = task_data['label_interval']
  bboxes = task_data['bboxes']
  bboxes = { int(frame_id): box_list for frame_id, box_list in bboxes.items() }

  all_anns = {}
  for (frame_id, img) in get_video_frames_opencv(file_path, start_frame_id, end_frame_id, label_interval):
    anns = compute_mask(model, dataset, img, bboxes[frame_id])
    all_anns.update(anns)
    report_progress(conn, task_data, frame_id + 1)
  post_anns(conn, task_data, all_anns)


def process_task_image_mask(model, dataset, conn, task_data, file_path):
  bboxes = task_data['bboxes']

  img = cv2.imread(file_path, CV_LOAD_IMAGE_COLOR)
  anns = compute_mask(model, dataset, img, bboxes)
  post_anns(conn, task_data, anns)


def process_mask(model, dataset, conn, media_root):
  req_data = { 'req_type': 'get_task', 'task_type': 'mask' }
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
        process_task_video_mask(model, dataset, conn, task_data, file_path)
      else:
        process_task_image_mask(model, dataset, conn, task_data, file_path)

      report_finish(conn, task_data)

    except Exception as e:
      print_log('process',
          'Error occurred from recv_data [{}]: {}'.format(recv_data, repr(e)))
      time.sleep(5)
      continue


def compute_mask(model, dataset, img, bboxes):
  if len(bboxes) == 0:
    return {}

  cls_boxes = [[] for _ in dataset.classes.values()]
  cls_box_ids = [[] for _ in dataset.classes.values()]
  for (box_id, cls_id, x1, y1, x2, y2) in bboxes:
    cls_boxes[dataset.cls_map_8cls[cls_id]].append([x1, y1, x2, y2, 1.0])
    cls_box_ids[dataset.cls_map_8cls[cls_id]].append(box_id)
  cls_boxes = [np.array(box_list, dtype=np.float32).reshape((-1, 5)) for box_list in cls_boxes]

  with c2_utils.NamedCudaScope(0):
    cls_masks = im_detect_mask_all(model, img, cls_boxes)

  if cls_masks is None:
    return {}

  cls_masks_ = [[] for _ in range(9)]
  cls_boxes_ = [[] for _ in range(9)]
  cls_box_ids_ = [[] for _ in range(9)]
  for cls_id, masks in enumerate(cls_masks):
    if cls_id in dataset.cls_map_8cls_inv:
      inv_id = dataset.cls_map_8cls_inv[cls_id]
      cls_masks_[inv_id].extend(masks)
      cls_boxes_[inv_id].extend(cls_boxes[cls_id])
      cls_box_ids_[inv_id].extend(cls_box_ids[cls_id])

  anns = pack_mask(dataset, cls_masks_, cls_boxes_, cls_box_ids_)
  return anns


def pack_mask(dataset, cls_masks, cls_boxes, cls_box_ids):
  anns = {}
  for (masks, bboxes, box_ids) in zip(cls_masks, cls_boxes, cls_box_ids):
    for (mask, bbox, box_id) in zip(masks, bboxes, box_ids):
      json_str = json.dumps({ 'bbox': bbox[:4].astype(np.int).tolist(), 'mask': mask })
      anns[box_id] = json_str
  return anns


def main(cfg_path):
  try:
    cfg = json.loads(open(cfg_path, 'r').read())
    conn = connect_to_server(cfg['server']['host'], cfg['server']['port'])
    model, dataset = initialize(cfg['mask']['path'])
    process_mask(model, dataset, conn, cfg['media_root'])
    conn.close()

  except KeyboardInterrupt as e:
    pass
  except Exception as e:
    pass
