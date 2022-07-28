# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Inference functionality for most Detectron models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2
import numpy as np

from caffe2.python import core
from caffe2.python import workspace

from detectron.core.config import cfg
import detectron.modeling.FPN as fpn
import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils


def cls_boxes_to_boxes(cls_boxes):
    boxes = []
    labels = []
    for c in range(1, len(cls_boxes)):
        if len(cls_boxes[c]) > 0:
            boxes.append(cls_boxes[c])
            labels.extend([c] * len(cls_boxes[c]))

    if len(boxes) > 0:
      boxes = np.vstack(boxes)  # safe
      labels = np.hstack(labels)  # safe
    else:
      boxes = np.zeros((0, 7), dtype=np.float32)  # x1, y1, x2, y2, score, uncertainty, bg_score
      labels = np.zeros((0,), dtype=np.int)

    return boxes, labels


def box_results_with_argmax(pred_scores, pred_boxes):
    if len(pred_scores) == 0:
      labels = np.zeros((0,), dtype=np.int)
      scores = np.zeros((0,), dtype=np.float32)
      boxes = np.zeros((0, 4), dtype=np.float32)
      return scores, boxes, labels

    num_rois, num_classes = pred_scores.shape
    pred_boxes = pred_boxes.reshape((num_rois, num_classes, 4))
    labels = np.argmax(pred_scores[:, 1:], axis=1) + 1
    scores = np.hstack([s[c] for s, c in zip(pred_scores, labels)])  # safe
    boxes = np.vstack([b[c] for b, c in zip(pred_boxes, labels)])  # safe
    return scores, boxes, labels


def resize_boxes(boxes, ratio, im):
    im_h, im_w = im.shape[:2]
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
    widths *= ratio
    heights *= ratio
    new_boxes = boxes.copy()
    new_boxes[:, 0] = ctr_x - 0.5 * widths
    new_boxes[:, 1] = ctr_y - 0.5 * heights
    new_boxes[:, 2] = ctr_x + 0.5 * widths - 1
    new_boxes[:, 3] = ctr_y + 0.5 * heights - 1
    new_boxes = box_utils.clip_boxes_to_image(new_boxes, im_h, im_w)
    return new_boxes


def test_im_detect_with_cls_boxes(model, im, cls_boxes):
    rois, rois_labels = cls_boxes_to_boxes(cls_boxes)
    if len(rois) == 0:
        pred_scores = np.zeros((0, len(cls_boxes)), dtype=np.float32)
        pred_boxes = np.zeros((0, len(cls_boxes) * 4), dtype=np.float32)
        return pred_scores, pred_boxes

    pred_scores, pred_boxes, _ = im_detect_bbox(model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes=resize_boxes(rois[:, :4], 1.1, im))
    scores, boxes, labels = box_results_with_argmax(pred_scores, pred_boxes)

    print('refinement results:')
    for rc, roi, bc, box, bs in zip(rois_labels, rois, labels, boxes, scores):
        rx1, ry1, rx2, ry2, rs = roi[:5]
        bx1, by1, bx2, by2 = box
        print('{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.3f}, {:d} -> {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.3f}, {:d}'.format(rx1, ry1, rx2, ry2, rs, rc, bx1, by1, bx2, by2, bs, bc))

    return pred_scores, pred_boxes


def im_detect_all(model, im, refine=True):
    if cfg.TEST.BBOX_AUG.ENABLED:
        scores, boxes, im_scale = im_detect_bbox_aug(model, im)
    else:
        scores, boxes, im_scale = im_detect_bbox(model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)

    if refine:
        scores2 = scores[:, 1:].reshape((-1,))
        boxes2 = boxes.reshape((scores.shape[0], scores.shape[1], 4))[:, 1:, :].reshape((-1, 4))
        keep = np.where(scores > cfg.TEST.SCORE_THRESH)[0]
        scores2, boxes2, _ = im_detect_bbox(model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes=boxes2[keep])
        scores = np.vstack([scores, scores2])  # safe
        boxes = np.vstack([boxes, boxes2])  # safe

    _, _, cls_boxes = box_results_with_nms_and_limit(scores, boxes)

    #scores, boxes = test_im_detect_with_cls_boxes(model, im, cls_boxes)
    #_, _, cls_boxes = box_results_with_nms_and_limit(scores, boxes)

    cls_boxes = class_agnostic_nms(cls_boxes)
    return cls_boxes


def im_detect_bbox(model, im, target_scale, target_max_size, boxes=None):
    """Bounding box object detection for an image with given box proposals.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals in 0-indexed
            [x1, y1, x2, y2] format, or None if using RPN

    Returns:
        scores (ndarray): R x K array of object class scores for K classes
            (K includes background as object category 0)
        boxes (ndarray): R x 4*K array of predicted bounding boxes
        im_scales (list): list of image scales used in the input blob (as
            returned by _get_blobs and for use with im_detect_mask, etc.)
    """
    inputs, im_scale = _get_blobs(im, boxes, target_scale, target_max_size)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(inputs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(
            hashes, return_index=True, return_inverse=True
        )
        inputs['rois'] = inputs['rois'][index, :]
        boxes = boxes[index, :]

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS and not cfg.MODEL.FASTER_RCNN:
        _add_multilevel_rois_for_test(inputs, 'rois')

    if boxes is None or not cfg.MODEL.FASTER_RCNN:
        for k, v in inputs.items():
            workspace.FeedBlob(core.ScopedName(k), v)
        workspace.RunNet(model.net.Proto().name)

    else:
        for k, v in inputs.items():
            workspace.FeedBlob(core.ScopedName(k), v)
        workspace.RunNet(model.conv_body_net.Proto().name)

        if cfg.FPN.MULTILEVEL_ROIS:
            _add_multilevel_rois_for_test(inputs, 'rois')
        for k, v in inputs.items():
            if k.startswith('rois'):
                workspace.FeedBlob(core.ScopedName(k), v)
        workspace.RunNet(model.rcnn_net.Proto().name)

    scores, pred_boxes = compute_score_bbox(im, im_scale)

    if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    return scores, pred_boxes, im_scale


def compute_score_bbox(im, im_scale):
    # Read out blobs
    if cfg.MODEL.FASTER_RCNN:
        rois = workspace.FetchBlob(core.ScopedName('rois'))
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scale

    # Softmax class probabilities
    scores = workspace.FetchBlob(core.ScopedName('cls_prob')).squeeze()
    # In case there is 1 proposal
    scores = scores.reshape([-1, scores.shape[-1]])

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = workspace.FetchBlob(core.ScopedName('bbox_pred')).squeeze()
        # In case there is 1 proposal
        box_deltas = box_deltas.reshape([-1, box_deltas.shape[-1]])
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            # Remove predictions for bg class (compat with MSRA code)
            box_deltas = box_deltas[:, -4:]
        pred_boxes = box_utils.bbox_transform(
            boxes, box_deltas, cfg.MODEL.BBOX_REG_WEIGHTS
        )
        pred_boxes = box_utils.clip_tiled_boxes(pred_boxes, im.shape)
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            pred_boxes = np.tile(pred_boxes, (1, scores.shape[1]))
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.MODEL.SUBCLASS_FOR_TEST and cfg.MODEL.NUM_SUBCLASSES > 0:
        pred_boxes_sup = pred_boxes
        pred_boxes = np.zeros((pred_boxes_sup.shape[0], cfg.MODEL.NUM_SUBCLASSES * 4), dtype=np.float32)
        subclass_map = np.array(cfg.MODEL.SUBCLASS_MAP, dtype=np.int)
        for idx_sub in range(cfg.MODEL.NUM_SUBCLASSES):
            idx_sup = subclass_map[idx_sub]
            pred_boxes[:, idx_sub*4:(idx_sub+1)*4] = pred_boxes_sup[:, idx_sup*4:(idx_sup+1)*4]
        scores = workspace.FetchBlob(core.ScopedName('cls_sub_prob')).squeeze()
        # In case there is 1 proposal
        scores = scores.reshape([-1, scores.shape[-1]])

    return scores, pred_boxes


def im_detect_bbox_aug(model, im, box_proposals=None):
    # Collect detections computed under different transformations
    scores_ts = []
    boxes_ts = []

    def add_preds_t(scores_t, boxes_t):
        scores_ts.append(scores_t)
        boxes_ts.append(boxes_t)

    # Perform detection on the horizontally flipped image
    if cfg.TEST.BBOX_AUG.H_FLIP:
        scores_hf, boxes_hf, _ = im_detect_bbox_hflip(
            model,
            im,
            cfg.TEST.SCALE,
            cfg.TEST.MAX_SIZE,
            box_proposals=box_proposals
        )
        add_preds_t(scores_hf, boxes_hf)
        #print('h_flip', boxes_hf.reshape(-1, 4)[:, 2].max(), boxes_hf.reshape(-1, 4)[:, 3].max())

    # Compute detections at different scales
    for scale in cfg.TEST.BBOX_AUG.SCALES:
        max_size = cfg.TEST.BBOX_AUG.MAX_SIZE
        scores_scl, boxes_scl = im_detect_bbox_scale(
            model, im, scale, max_size, box_proposals
        )
        add_preds_t(scores_scl, boxes_scl)
        #print('scale {}'.format(scale), boxes_scl.reshape(-1, 4)[:, 2].max(), boxes_scl.reshape(-1, 4)[:, 3].max())

        if cfg.TEST.BBOX_AUG.SCALE_H_FLIP:
            scores_scl_hf, boxes_scl_hf = im_detect_bbox_scale(
                model, im, scale, max_size, box_proposals, hflip=True
            )
            add_preds_t(scores_scl_hf, boxes_scl_hf)

    # Perform detection at different aspect ratios
    for aspect_ratio in cfg.TEST.BBOX_AUG.ASPECT_RATIOS:
        scores_ar, boxes_ar = im_detect_bbox_aspect_ratio(
            model, im, aspect_ratio, box_proposals
        )
        add_preds_t(scores_ar, boxes_ar)

        if cfg.TEST.BBOX_AUG.ASPECT_RATIO_H_FLIP:
            scores_ar_hf, boxes_ar_hf = im_detect_bbox_aspect_ratio(
                model, im, aspect_ratio, box_proposals, hflip=True
            )
            add_preds_t(scores_ar_hf, boxes_ar_hf)

    # Compute detections for the original image (identity transform) last to
    # ensure that the Caffe2 workspace is populated with blobs corresponding
    # to the original image on return (postcondition of im_detect_bbox)
    scores_i, boxes_i, im_scale_i = im_detect_bbox(
        model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes=box_proposals
    )
    add_preds_t(scores_i, boxes_i)
    #print('normal', boxes_i.reshape(-1, 4)[:, 2].max(), boxes_i.reshape(-1, 4)[:, 3].max())

    scores_c = np.vstack(scores_ts)  # safe
    boxes_c = np.vstack(boxes_ts)  # safe
    return scores_c, boxes_c, im_scale_i


def im_detect_bbox_hflip(
    model, im, target_scale, target_max_size, box_proposals=None
):
    """Performs bbox detection on the horizontally flipped image.
    Function signature is the same as for im_detect_bbox.
    """
    # Compute predictions on the flipped image
    im_hf = im[:, ::-1, :]
    im_width = im.shape[1]

    if box_proposals is not None:
        box_proposals_hf = box_utils.flip_boxes(box_proposals, im_width)
    else:
        box_proposals_hf = None

    scores_hf, boxes_hf, im_scale = im_detect_bbox(
        model, im_hf, target_scale, target_max_size, boxes=box_proposals_hf
    )

    # Invert the detections computed on the flipped image
    boxes_inv = box_utils.flip_boxes(boxes_hf, im_width)

    return scores_hf, boxes_inv, im_scale


def im_detect_bbox_scale(
    model, im, target_scale, target_max_size, box_proposals=None, hflip=False
):
    """Computes bbox detections at the given scale.
    Returns predictions in the original image space.
    """
    if hflip:
        scores_scl, boxes_scl, _ = im_detect_bbox_hflip(
            model, im, target_scale, target_max_size, box_proposals=box_proposals
        )
    else:
        scores_scl, boxes_scl, _ = im_detect_bbox(
            model, im, target_scale, target_max_size, boxes=box_proposals
        )
    return scores_scl, boxes_scl


def im_detect_bbox_aspect_ratio(
    model, im, aspect_ratio, box_proposals=None, hflip=False
):
    def aspect_ratio_rel(im, aspect_ratio):
        """Performs width-relative aspect ratio transformation."""
        im_h, im_w = im.shape[:2]
        im_ar_w = int(round(aspect_ratio * im_w))
        im_ar = cv2.resize(im, dsize=(im_ar_w, im_h))
        return im_ar

    """Computes bbox detections at the given width-relative aspect ratio.
    Returns predictions in the original image space.
    """
    # Compute predictions on the transformed image
    im_ar = aspect_ratio_rel(im, aspect_ratio)

    if box_proposals is not None:
        box_proposals_ar = box_utils.aspect_ratio(box_proposals, aspect_ratio)
    else:
        box_proposals_ar = None

    if hflip:
        scores_ar, boxes_ar, _ = im_detect_bbox_hflip(
            model,
            im_ar,
            cfg.TEST.SCALE,
            cfg.TEST.MAX_SIZE,
            box_proposals=box_proposals_ar
        )
    else:
        scores_ar, boxes_ar, _ = im_detect_bbox(
            model,
            im_ar,
            cfg.TEST.SCALE,
            cfg.TEST.MAX_SIZE,
            boxes=box_proposals_ar
        )

    # Invert the detected boxes
    boxes_inv = box_utils.aspect_ratio(boxes_ar, 1.0 / aspect_ratio)

    return scores_ar, boxes_inv


def class_agnostic_nms(cls_boxes):
    boxes, labels = cls_boxes_to_boxes(cls_boxes)
    if len(boxes) == 0:
        return cls_boxes

    keep = box_utils.nms(boxes[:, :5], cfg.TEST.CLASS_AGNOSTIC_NMS)
    boxes = boxes[keep]
    labels = labels[keep]

    cls_boxes_new = [boxes[labels == c] for c in range(len(cls_boxes))]
    return cls_boxes_new


def box_results_with_nms_and_limit(scores, boxes, max_num_outputs=cfg.TEST.DETECTIONS_PER_IM):
    """Returns bounding-box detection results by thresholding on scores and
    applying non-maximum suppression (NMS).

    `boxes` has shape (#detections, 4 * #classes), where each row represents
    a list of predicted bounding boxes for each of the object classes in the
    dataset (including the background class). The detections in each row
    originate from the same object proposal.

    `scores` has shape (#detection, #classes), where each row represents a list
    of object detection confidence scores for each of the object classes in the
    dataset (including the background class). `scores[i, j]`` corresponds to the
    box at `boxes[i, j * 4:(j + 1) * 4]`.
    """
    num_classes = cfg.MODEL.NUM_SUBCLASSES if cfg.MODEL.SUBCLASS_FOR_TEST and cfg.MODEL.NUM_SUBCLASSES > 0 else cfg.MODEL.NUM_CLASSES
    cls_boxes = [[] for _ in range(num_classes)]
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > cfg.TEST.SCORE_THRESH)[0]
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, j * 4:(j + 1) * 4]
        dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(
            np.float32, copy=False
        )
        if cfg.TEST.SOFT_NMS.ENABLED:
            nms_dets, _ = box_utils.soft_nms(
                dets_j,
                sigma=cfg.TEST.SOFT_NMS.SIGMA,
                overlap_thresh=cfg.TEST.NMS,
                score_thresh=0.0001,
                method=cfg.TEST.SOFT_NMS.METHOD
            )
        else:
            keep = box_utils.nms(dets_j, cfg.TEST.NMS)
            nms_dets = dets_j[keep, :]
        # Refine the post-NMS boxes using bounding-box voting
        if cfg.TEST.BBOX_VOTE.ENABLED:
            nms_dets = box_utils.box_voting(
                nms_dets,
                dets_j,
                cfg.TEST.BBOX_VOTE.VOTE_TH,
                scoring_method=cfg.TEST.BBOX_VOTE.SCORING_METHOD
            )
        cls_boxes[j] = np.hstack((  # safe
            nms_dets,
            1.0 - scores[inds[keep]].sum(axis=1).reshape((-1, 1)),
            scores[inds[keep], 0].reshape((-1, 1))
        ))

    boxes, labels = cls_boxes_to_boxes(cls_boxes)
    # Limit to max_per_image detections **over all classes**
    if len(boxes) > 0 and max_num_outputs > 0:
        image_scores = np.hstack(  # safe
            [cls_boxes[j][:, 4] for j in range(1, num_classes)]
        )
        if len(image_scores) > max_num_outputs:
            image_thresh = np.sort(image_scores)[-max_num_outputs]
            for j in range(1, num_classes):
                keep = np.where(cls_boxes[j][:, 4] >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]

    return boxes, labels, cls_boxes


def _get_rois_blob(im_rois, im_scale):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid with columns
            [level, x1, y1, x2, y2]
    """
    rois, levels = _project_im_rois(im_rois, im_scale)
    rois_blob = np.hstack((levels, rois))  # safe
    return rois_blob.astype(np.float32, copy=False)


def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (ndarray): image pyramid levels used by each projected RoI
    """
    rois = im_rois.astype(np.float, copy=False) * scales
    levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)
    return rois, levels


def _add_multilevel_rois_for_test(blobs, name):
    """Distributes a set of RoIs across FPN pyramid levels by creating new level
    specific RoI blobs.

    Arguments:
        blobs (dict): dictionary of blobs
        name (str): a key in 'blobs' identifying the source RoI blob

    Returns:
        [by ref] blobs (dict): new keys named by `name + 'fpn' + level`
            are added to dict each with a value that's an R_level x 5 ndarray of
            RoIs (see _get_rois_blob for format)
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL
    lvls = fpn.map_rois_to_fpn_levels(blobs[name][:, 1:5], lvl_min, lvl_max)
    fpn.add_multilevel_roi_blobs(
        blobs, name, blobs[name], lvls, lvl_min, lvl_max
    )


def _get_blobs(im, rois, target_scale, target_max_size):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    blobs['data'], im_scale, blobs['im_info'] = \
        blob_utils.get_image_blob(im, target_scale, target_max_size)
    if rois is not None:
        blobs['rois'] = _get_rois_blob(rois, im_scale)
    return blobs, im_scale
