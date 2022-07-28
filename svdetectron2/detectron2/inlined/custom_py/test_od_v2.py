from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2
import numpy as np
import scipy.optimize

from caffe2.python import core
from caffe2.python import workspace

from detectron.core.config import cfg
import detectron.utils.boxes as box_utils
from detectron.core.test_od import cls_boxes_to_boxes, box_results_with_argmax, resize_boxes, compute_score_bbox, class_agnostic_nms, box_results_with_nms_and_limit, _get_blobs, _get_rois_blob, _add_multilevel_rois_for_test


def compute_iou(boxes1, boxes2):
  areas1 = (boxes1[:, 2] - boxes1[:, 0] + 1.0) * (boxes1[:, 3] - boxes1[:, 1] + 1.0)
  areas2 = (boxes2[:, 2] - boxes2[:, 0] + 1.0) * (boxes2[:, 3] - boxes2[:, 1] + 1.0)
  ious = np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
  for i, (x1, y1, x2, y2) in enumerate(boxes1[:, :4]):
    iw = np.maximum(0.0, np.minimum(x2, boxes2[:, 2]) - np.maximum(x1, boxes2[:, 0]) + 1.0)
    ih = np.maximum(0.0, np.minimum(y2, boxes2[:, 3]) - np.maximum(y1, boxes2[:, 1]) + 1.0)
    ious[i] = (iw * ih) / (areas1[i] + areas2 - iw * ih)
  return ious


def video_detect_all(model, im, prev_boxes, refine=True):
  if cfg.TEST.BBOX_AUG.ENABLED:
    pred_scores, pred_boxes, im_scale = im_detect_bbox_aug(model, im, refine=refine)
  else:
    pred_scores, pred_boxes, im_scale = im_detect_bbox(model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)

  if prev_boxes is not None and len(prev_boxes) > 0:
    resized_prev_boxes = resize_boxes(prev_boxes, cfg.TEST.TRACK_RESIZE_RATIO, im)
    prev_pred_scores, prev_pred_boxes = im_refine_bbox(model, im, im_scale, resized_prev_boxes)
    scores1, prev_boxes, _ = box_results_with_argmax(prev_pred_scores, prev_pred_boxes)
    prev_pred_scores, prev_pred_boxes = im_refine_bbox(model, im, im_scale, prev_boxes)
    scores2, prev_boxes, _ = box_results_with_argmax(prev_pred_scores, prev_pred_boxes)
    pred_scores = np.vstack([pred_scores, prev_pred_scores])  # safe
    pred_boxes = np.vstack([pred_boxes, prev_pred_boxes])  # safe

  if len(pred_boxes) > 0:
    _, _, cls_boxes = box_results_with_nms_and_limit(pred_scores, pred_boxes)
    cls_boxes = class_agnostic_nms(cls_boxes)
  else:
    cls_boxes = [[] for _ in range(pred_scores.shape[1])]

  boxes, labels = cls_boxes_to_boxes(cls_boxes)
  if len(boxes) == 0:
    cls_tracks = [np.zeros((len(cls_boxes[c]),), dtype=np.int) for c in range(len(cls_boxes))]
    return cls_boxes, cls_tracks

  tracks = np.zeros((len(boxes),), dtype=np.int)
  if prev_boxes is not None and len(prev_boxes) > 0:
    ious = compute_iou(prev_boxes, boxes)
    prev_indices, indices = scipy.optimize.linear_sum_assignment(1.0 - ious)
    for (prev_idx, idx) in zip(prev_indices, indices):
      if ious[prev_idx, idx] >= cfg.TEST.TRACK_IOU_THRESH:
        tracks[idx] = prev_idx + 1
    #indices = np.argsort(-boxes[:, 4])
    #results = np.hstack([tracks.reshape((-1, 1)), labels.reshape((-1, 1)), np.round(boxes[:, 4] * 1000).reshape((-1, 1))]).astype(np.int)[indices]
    #print(results)

  cls_boxes = [boxes[labels == c] for c in range(len(cls_boxes))]
  cls_tracks = [tracks[labels == c] for c in range(len(cls_boxes))]
  return cls_boxes, cls_tracks


def im_detect_all(model, im, refine=True, ex_boxes=None):
  if cfg.TEST.BBOX_AUG.ENABLED:
    pred_scores, pred_boxes, im_scale = im_detect_bbox_aug(model, im, refine=refine)
  else:
    pred_scores, pred_boxes, im_scale = im_detect_bbox(model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)

  _, _, cls_boxes = box_results_with_nms_and_limit(pred_scores, pred_boxes)
  cls_boxes = class_agnostic_nms(cls_boxes)

  if ex_boxes is not None and len(ex_boxes) > 0:
    ex_pred_scores, ex_pred_boxes = im_refine_bbox(model, im, im_scale, ex_boxes)
    return cls_boxes, ex_pred_scores, ex_pred_boxes

  return cls_boxes


def im_detect_bbox(model, im, target_scale, target_max_size):
  inputs, im_scale = _get_blobs(im, None, target_scale, target_max_size)
  for k, v in inputs.items():
    workspace.FeedBlob(core.ScopedName(k), v)
  workspace.RunNet(model.net.Proto().name)
  pred_scores, pred_boxes = compute_score_bbox(im, im_scale)
  return pred_scores, pred_boxes, im_scale


def im_refine_bbox(model, im, im_scale, boxes):
  # skips running conv_body_net and rpn
  # must be run right after running at least conv_body_net part
  inputs = { 'rois': _get_rois_blob(boxes, im_scale) }
  if cfg.FPN.MULTILEVEL_ROIS:
    _add_multilevel_rois_for_test(inputs, 'rois')
  for k, v in inputs.items():
    if k.startswith('rois'):
      workspace.FeedBlob(core.ScopedName(k), v)
  workspace.RunNet(model.rcnn_net.Proto().name)
  pred_scores, pred_boxes = compute_score_bbox(im, im_scale)
  return pred_scores, pred_boxes


def im_refine_pred_boxes(model, im, im_scale, pred_scores, pred_boxes):
  # skips running conv_body_net and rpn
  # must be run right after running at least conv_body_net part
  num_rois, num_classes = pred_scores.shape
  pred_boxes = pred_boxes.reshape((num_rois, num_classes, 4))
  scores = pred_scores[:, 1:].reshape((-1,))
  boxes = pred_boxes[:, 1:].reshape((-1, 4))
  boxes = boxes[scores >= cfg.TEST.SCORE_THRESH]
  if len(boxes) == 0:
    pred_scores = np.zeros((0, num_classes), dtype=np.float32)
    pred_boxes = np.zeros((0, num_classes * 4), dtype=np.float32)
    return pred_scores, pred_boxes

  pred_scores, pred_boxes = im_refine_bbox(model, im, im_scale, boxes)
  return pred_scores, pred_boxes


def im_detect_bbox_aug(model, im, refine=False):
  all_pred_scores = []
  all_pred_boxes = []

  def add_preds(pred_scores_t, pred_boxes_t):
    all_pred_scores.append(pred_scores_t)
    all_pred_boxes.append(pred_boxes_t)

  if cfg.TEST.BBOX_AUG.H_FLIP:
    pred_scores, pred_boxes, _ = im_detect_bbox_hflip(model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)
    add_preds(pred_scores, pred_boxes)

  for scale in cfg.TEST.BBOX_AUG.SCALES:
    max_size = cfg.TEST.BBOX_AUG.MAX_SIZE
    pred_scores, pred_boxes = im_detect_bbox_scale(model, im, scale, max_size)
    add_preds(pred_scores, pred_boxes)
    if cfg.TEST.BBOX_AUG.SCALE_H_FLIP:
      pred_scores, pred_boxes = im_detect_bbox_scale(model, im, scale, max_size, hflip=True)
      add_preds(pred_scores, pred_boxes)

  for aspect_ratio in cfg.TEST.BBOX_AUG.ASPECT_RATIOS:
    pred_scores, pred_boxes = im_detect_bbox_aspect_ratio(model, im, aspect_ratio)
    add_preds(pred_scores, pred_boxes)
    if cfg.TEST.BBOX_AUG.ASPECT_RATIO_H_FLIP:
      pred_scores, pred_boxes = im_detect_bbox_aspect_ratio(model, im, aspect_ratio, hflip=True)
      add_preds(pred_scores, pred_boxes)

  pred_scores, pred_boxes, im_scale = im_detect_bbox(model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)
  add_preds(pred_scores, pred_boxes)

  pred_scores = np.vstack(all_pred_scores)  # safe
  pred_boxes = np.vstack(all_pred_boxes)  # safe

  if refine:
    pred_scores, pred_boxes = im_refine_pred_boxes(model, im, im_scale, pred_scores, pred_boxes)

  return pred_scores, pred_boxes, im_scale


def im_detect_bbox_hflip(model, im, target_scale, target_max_size):
  im_hf = im[:, ::-1, :]
  im_width = im.shape[1]
  scores_hf, boxes_hf, im_scale = im_detect_bbox(model, im_hf, target_scale, target_max_size)
  boxes_inv = box_utils.flip_boxes(boxes_hf, im_width)
  return scores_hf, boxes_inv, im_scale


def im_detect_bbox_scale(model, im, target_scale, target_max_size, hflip=False):
  if hflip:
    scores_scl, boxes_scl, _ = im_detect_bbox_hflip(model, im, target_scale, target_max_size)
  else:
    scores_scl, boxes_scl, _ = im_detect_bbox(model, im, target_scale, target_max_size)
  return scores_scl, boxes_scl


def im_detect_bbox_aspect_ratio(model, im, aspect_ratio, hflip=False):
  def aspect_ratio_rel(im, aspect_ratio):
    im_h, im_w = im.shape[:2]
    im_ar_w = int(round(aspect_ratio * im_w))
    im_ar = cv2.resize(im, dsize=(im_ar_w, im_h))
    return im_ar

  im_ar = aspect_ratio_rel(im, aspect_ratio)
  if hflip:
    scores_ar, boxes_ar, _ = im_detect_bbox_hflip(model, im_ar, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)
  else:
    scores_ar, boxes_ar, _ = im_detect_bbox(model, im_ar, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)
  boxes_inv = box_utils.aspect_ratio(boxes_ar, 1.0 / aspect_ratio)
  return scores_ar, boxes_inv
