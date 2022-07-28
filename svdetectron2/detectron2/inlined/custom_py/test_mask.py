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
from detectron.core.test_od import im_detect_bbox_aug, im_detect_bbox, _get_rois_blob, _add_multilevel_rois_for_test


def im_detect_mask_all(model, im, cls_boxes):
    if cfg.TEST.BBOX_AUG.ENABLED:
        _, _, im_scale = im_detect_bbox_aug(model, im)
    else:
        _, _, im_scale = im_detect_bbox(model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)

    boxes = [cls_boxes[j] for j in range(1, len(cls_boxes)) if len(cls_boxes[j]) > 0]
    if len(boxes) == 0:
      return None
    boxes = np.vstack(boxes)[:, :4]  # safe

    if cfg.TEST.MASK_AUG.ENABLED:
        masks = im_detect_mask_aug(model, im, boxes)
    else:
        masks = im_detect_mask(model, im_scale, boxes)

    cls_segms = segm_results(cls_boxes, masks, boxes, im.shape[0], im.shape[1])

    return cls_segms


def im_conv_body_only(model, im, target_scale, target_max_size):
    """Runs `model.conv_body_net` on the given image `im`."""
    im_blob, im_scale, _im_info = blob_utils.get_image_blob(
        im, target_scale, target_max_size
    )
    workspace.FeedBlob(core.ScopedName('data'), im_blob)
    workspace.RunNet(model.conv_body_net.Proto().name)
    return im_scale


def im_detect_mask(model, im_scale, boxes):
    """Infer instance segmentation masks. This function must be called after
    im_detect_bbox as it assumes that the Caffe2 workspace is already populated
    with the necessary blobs.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im_scales (list): image blob scales as returned by im_detect_bbox
        boxes (ndarray): R x 4 array of bounding box detections (e.g., as
            returned by im_detect_bbox)

    Returns:
        pred_masks (ndarray): R x K x M x M array of class specific soft masks
            output by the network (must be processed by segm_results to convert
            into hard masks in the original image coordinate space)
    """
    M = cfg.MRCNN.RESOLUTION
    if boxes.shape[0] == 0:
        pred_masks = np.zeros((0, M, M), np.float32)
        return pred_masks

    inputs = {'mask_rois': _get_rois_blob(boxes, im_scale)}
    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois_for_test(inputs, 'mask_rois')

    for k, v in inputs.items():
        workspace.FeedBlob(core.ScopedName(k), v)
    workspace.RunNet(model.mask_net.Proto().name)

    # Fetch masks
    pred_masks = workspace.FetchBlob(
        core.ScopedName('mask_fcn_probs')
    ).squeeze()

    if cfg.MRCNN.CLS_SPECIFIC_MASK:
        pred_masks = pred_masks.reshape([-1, cfg.MODEL.NUM_CLASSES, M, M])
    else:
        pred_masks = pred_masks.reshape([-1, 1, M, M])

    return pred_masks


def im_detect_mask_aug(model, im, boxes):
    """Performs mask detection with test-time augmentations.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im (ndarray): BGR image to test
        boxes (ndarray): R x 4 array of bounding boxes

    Returns:
        masks (ndarray): R x K x M x M array of class specific soft masks
    """
    assert not cfg.TEST.MASK_AUG.SCALE_SIZE_DEP, \
        'Size dependent scaling not implemented'

    # Collect masks computed under different transformations
    masks_ts = []

    # Compute masks for the original image (identity transform)
    im_scale_i = im_conv_body_only(model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)
    masks_i = im_detect_mask(model, im_scale_i, boxes)
    masks_ts.append(masks_i)

    # Perform mask detection on the horizontally flipped image
    if cfg.TEST.MASK_AUG.H_FLIP:
        masks_hf = im_detect_mask_hflip(
            model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes
        )
        masks_ts.append(masks_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.MASK_AUG.SCALES:
        max_size = cfg.TEST.MASK_AUG.MAX_SIZE
        masks_scl = im_detect_mask_scale(model, im, scale, max_size, boxes)
        masks_ts.append(masks_scl)

        if cfg.TEST.MASK_AUG.SCALE_H_FLIP:
            masks_scl_hf = im_detect_mask_scale(
                model, im, scale, max_size, boxes, hflip=True
            )
            masks_ts.append(masks_scl_hf)

    # Compute masks at different aspect ratios
    for aspect_ratio in cfg.TEST.MASK_AUG.ASPECT_RATIOS:
        masks_ar = im_detect_mask_aspect_ratio(model, im, aspect_ratio, boxes)
        masks_ts.append(masks_ar)

        if cfg.TEST.MASK_AUG.ASPECT_RATIO_H_FLIP:
            masks_ar_hf = im_detect_mask_aspect_ratio(
                model, im, aspect_ratio, boxes, hflip=True
            )
            masks_ts.append(masks_ar_hf)

    # Combine the predicted soft masks
    if cfg.TEST.MASK_AUG.HEUR == 'SOFT_AVG':
        masks_c = np.mean(masks_ts, axis=0)
    elif cfg.TEST.MASK_AUG.HEUR == 'SOFT_MAX':
        masks_c = np.amax(masks_ts, axis=0)
    elif cfg.TEST.MASK_AUG.HEUR == 'LOGIT_AVG':

        def logit(y):
            return -1.0 * np.log((1.0 - y) / np.maximum(y, 1e-20))

        logit_masks = [logit(y) for y in masks_ts]
        logit_masks = np.mean(logit_masks, axis=0)
        masks_c = 1.0 / (1.0 + np.exp(-logit_masks))
    else:
        raise NotImplementedError(
            'Heuristic {} not supported'.format(cfg.TEST.MASK_AUG.HEUR)
        )

    return masks_c


def im_detect_mask_hflip(model, im, target_scale, target_max_size, boxes):
    """Performs mask detection on the horizontally flipped image.
    Function signature is the same as for im_detect_mask_aug.
    """
    # Compute the masks for the flipped image
    im_hf = im[:, ::-1, :]
    boxes_hf = box_utils.flip_boxes(boxes, im.shape[1])

    im_scale = im_conv_body_only(model, im_hf, target_scale, target_max_size)
    masks_hf = im_detect_mask(model, im_scale, boxes_hf)

    # Invert the predicted soft masks
    masks_inv = masks_hf[:, :, :, ::-1]

    return masks_inv


def im_detect_mask_scale(
    model, im, target_scale, target_max_size, boxes, hflip=False
):
    """Computes masks at the given scale."""
    if hflip:
        masks_scl = im_detect_mask_hflip(
            model, im, target_scale, target_max_size, boxes
        )
    else:
        im_scale = im_conv_body_only(model, im, target_scale, target_max_size)
        masks_scl = im_detect_mask(model, im_scale, boxes)
    return masks_scl


def im_detect_mask_aspect_ratio(model, im, aspect_ratio, boxes, hflip=False):
    """Computes mask detections at the given width-relative aspect ratio."""

    # Perform mask detection on the transformed image
    im_ar = image_utils.aspect_ratio_rel(im, aspect_ratio)
    boxes_ar = box_utils.aspect_ratio(boxes, aspect_ratio)

    if hflip:
        masks_ar = im_detect_mask_hflip(
            model, im_ar, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes_ar
        )
    else:
        im_scale = im_conv_body_only(
            model, im_ar, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE
        )
        masks_ar = im_detect_mask(model, im_scale, boxes_ar)

    return masks_ar


def segm_results(cls_boxes, masks, ref_boxes, im_h, im_w):
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_segms = [[] for _ in range(len(cls_boxes))]
    mask_ind = 0
    # To work around an issue with cv2.resize (it seems to automatically pad
    # with repeated border values), we manually zero-pad the masks by 1 pixel
    # prior to resizing back to the original image resolution. This prevents
    # "top hat" artifacts. We therefore need to expand the reference boxes by an
    # appropriate factor.
    M = cfg.MRCNN.RESOLUTION
    ref_boxes = ref_boxes.astype(np.int32)
    scale = (M + 2.0) / M
    padded_ref_boxes = box_utils.expand_boxes(ref_boxes, scale)
    padded_mask = np.zeros((M + 2, M + 2), dtype=np.float32)

    # skip j = 0, because it's the background class
    for j in range(1, num_classes):
        segms = []
        for _ in range(cls_boxes[j].shape[0]):
            if cfg.MRCNN.CLS_SPECIFIC_MASK:
                padded_mask[1:-1, 1:-1] = masks[mask_ind, j, :, :]
            else:
                padded_mask[1:-1, 1:-1] = masks[mask_ind, 0, :, :]

            x1, y1, x2, y2 = ref_boxes[mask_ind, :]
            px1, py1, px2, py2 = padded_ref_boxes[mask_ind, :]
            pw = int(round(px2 - px1 + 1))
            ph = int(round(py2 - py1 + 1))
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            dx = int(round((pw - w) / 2.0))
            dy = int(round((ph - h) / 2.0))
            mask = cv2.resize(padded_mask, (pw, ph))
            mask = np.array(mask > cfg.MRCNN.THRESH_BINARIZE, dtype=np.uint8)
            mask = mask[dy:dy+h, dx:dx+w]
            segms.append(contour_to_text(mask_to_contour(mask, 4)))

            mask_ind += 1

        cls_segms[j] = segms

    if len(cls_boxes) > num_classes:
        # [person, bicycle], [person, motorcycle], [person, bicycle, motorcycle], [person, motorcycle, truck], [truck]
        seg_indices = { 0: [1, 2], 1: [1, 4], 2: [1, 2, 4], 3: [1, 4, 8], 4: [8] }
        padded_masks = []

    for j in range(num_classes, len(cls_boxes)):
        segms = []
        for _ in range(cls_boxes[j].shape[0]):
            if cfg.MRCNN.CLS_SPECIFIC_MASK:
                for seg_idx in seg_indices[j - num_classes]:
                    padded_mask = np.zeros((M + 2, M + 2), dtype=np.float32)
                    padded_mask[1:-1, 1:-1] = masks[mask_ind, seg_idx, :, :]
                    padded_masks.append(padded_mask)
            else:
                padded_mask[1:-1, 1:-1] = masks[mask_ind, 0, :, :]
                padded_masks.append(padded_mask)

            x1, y1, x2, y2 = ref_boxes[mask_ind, :]
            px1, py1, px2, py2 = padded_ref_boxes[mask_ind, :]
            pw = int(round(px2 - px1 + 1))
            ph = int(round(py2 - py1 + 1))
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            dx = int(round((pw - w) / 2.0))
            dy = int(round((ph - h) / 2.0))
            composite_mask = np.zeros((h, w), dtype=np.uint8)
            for padded_mask in padded_masks:
                mask = cv2.resize(padded_mask, (pw, ph))
                mask = np.array(mask > cfg.MRCNN.THRESH_BINARIZE, dtype=np.uint8)
                composite_mask |= mask[dy:dy+h, dx:dx+w]

            segms.append(contour_to_text(mask_to_contour(composite_mask, 4)))

            mask_ind += 1

        cls_segms[j] = segms

    assert mask_ind == masks.shape[0]
    return cls_segms


def mask_to_contour(mask, step):
  if cv2.__version__[0] in ['2', '4']:
    contours, hiers = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  else:
    _, contours, hiers = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  contours = [contour.astype(int) for contour in contours if contour.shape[0] >= 3]
  for idx in range(len(contours)):
    #epsilon = 0.001 * cv2.arcLength(contours[idx], True)
    #contours[idx] = cv2.approxPolyDP(contours[idx], epsilon, True)
    if step > 1 and contours[idx].shape[0] > step*4:
      contours[idx] = contours[idx][::step*2]
    elif step > 1 and contours[idx].shape[0] > step*2:
      contours[idx] = contours[idx][::step]
  return contours


def contour_to_text(contour):
  text = '|'.join([','.join([str(val) for val in cnt_child.reshape(-1)])
                  for cnt_child in contour])
  return text


def contour_from_text(contour_text):
  contour = []
  for cnt_child_text in contour_text.split('|'):
    cnt_child = [int(val) for val in cnt_child_text.split(',')]
    cnt_child = np.array(cnt_child, dtype=np.int32).reshape((-1, 1, 2))
    contour.append(cnt_child)
  return contour
