# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, Boxes3D, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from ..backbone.resnet import BottleneckBlock, make_stage
from ..matcher import Matcher
from ..poolers import ROIPooler
from ..proposal_generator.proposal_utils import add_ground_truth_to_proposals
from ..sampling import subsample_labels
from .box_head import build_box_head
from .fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputLayers_3DCar
from .keypoint_head import build_keypoint_head
from .mask_head import build_mask_head

import sys
import random
import math

ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)


def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


def build_3d_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = '_3DCarROIHeads'
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


def select_foreground_proposals(
    proposals: List[Instances], bg_label: int
) -> Tuple[List[Instances], List[torch.Tensor]]:
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


def select_proposals_with_visible_keypoints(proposals: List[Instances]) -> List[Instances]:
    """
    Args:
        proposals (list[Instances]): a list of N Instances, where N is the
            number of images.

    Returns:
        proposals: only contains proposals with at least one visible keypoint.

    Note that this is still slightly different from Detectron.
    In Detectron, proposals for training keypoint head are re-sampled from
    all the proposals with IOU>threshold & >=1 visible keypoint.

    Here, the proposals are first sampled from all proposals with
    IOU>threshold, then proposals with no visible keypoint are filtered out.
    This strategy seems to make no difference on Detectron and is easier to implement.
    """
    ret = []
    all_num_fg = []
    for proposals_per_image in proposals:
        # If empty/unannotated image (hard negatives), skip filtering for train
        if len(proposals_per_image) == 0:
            ret.append(proposals_per_image)
            continue
        gt_keypoints = proposals_per_image.gt_keypoints.tensor
        # #fg x K x 3
        vis_mask = gt_keypoints[:, :, 2] >= 1
        xs, ys = gt_keypoints[:, :, 0], gt_keypoints[:, :, 1]
        proposal_boxes = proposals_per_image.proposal_boxes.tensor.unsqueeze(dim=1)  # #fg x 1 x 4
        kp_in_box = (
            (xs >= proposal_boxes[:, :, 0])
            & (xs <= proposal_boxes[:, :, 2])
            & (ys >= proposal_boxes[:, :, 1])
            & (ys <= proposal_boxes[:, :, 3])
        )
        selection = (kp_in_box & vis_mask).any(dim=1)
        selection_idxs = torch.nonzero(selection, as_tuple=True)[0]
        all_num_fg.append(selection_idxs.numel())
        ret.append(proposals_per_image[selection_idxs])

    storage = get_event_storage()
    storage.put_scalar("keypoint_head/num_fg_samples", np.mean(all_num_fg))
    return ret


class ROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It typically contains logic to
    1. (in training only) match proposals with ground truth and sample them
    2. crop the regions and extract per-region features using proposals
    3. make per-region predictions with different heads

    It can have many variants, implemented as subclasses of this class.
    This base class contains the logic to match/sample proposals.
    But it is not necessary to inherit this class if the sampling logic is not needed.
    """

    @configurable
    def __init__(
        self,
        *,
        num_classes,
        batch_size_per_image,
        positive_sample_fraction,
        proposal_matcher,
        proposal_append_gt=True
    ):
        """
        NOTE: this interface is experimental.

        Args:
            num_classes (int): number of classes. Used to label background proposals.
            batch_size_per_image (int): number of proposals to use for training
            positive_sample_fraction (float): fraction of positive (foreground) proposals
                to use for training.
            proposal_matcher (Matcher): matcher that matches proposals and ground truth
            proposal_append_gt (bool): whether to include ground truth as proposals as well
        """
        super().__init__()
        self.batch_size_per_image = batch_size_per_image
        self.positive_sample_fraction = positive_sample_fraction
        self.num_classes = num_classes
        self.proposal_matcher = proposal_matcher
        self.proposal_append_gt = proposal_append_gt

    @classmethod
    def from_config(cls, cfg):
        return {
            "batch_size_per_image": cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
            "positive_sample_fraction": cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "proposal_append_gt": cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT,
            # Matcher to assign box proposals to gt boxes
            "proposal_matcher": Matcher(
                cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
                cfg.MODEL.ROI_HEADS.IOU_LABELS,
                allow_low_quality_matches=False,
            ),
        }

    def _sample_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # firstly, removing ignored class input
            gt_classes[gt_classes == self.ignored_class_idx] = -1
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
            
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_sample_fraction, self.num_classes, -1
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_sample_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        Args:
            images (ImageList):
            features (dict[str,Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:

                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.

        Returns:
            list[Instances]: length `N` list of `Instances` containing the
            detected instances. Returned during inference only; may be [] during training.

            dict[str->Tensor]:
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg)

        # fmt: off
        self.in_features  = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / input_shape[self.in_features[0]].stride, )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.mask_on      = cfg.MODEL.MASK_ON
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON
        assert len(self.in_features) == 1

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, out_channels = self._build_res5_block(cfg)
        self.box_predictor = FastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )

        if self.mask_on:
            self.mask_head = build_mask_head(
                cfg,
                ShapeSpec(channels=out_channels, width=pooler_resolution, height=pooler_resolution),
            )

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = make_stage(
            BottleneckBlock,
            3,
            first_stride=2,
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        return self.res5(x)

    def forward(self, images, features, proposals, targets=None):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        predictions = self.box_predictor(box_features.mean(dim=[2, 3]))

        if self.training:
            del features
            losses = self.box_predictor.losses(predictions, proposals)
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        if self.mask_on:
            features = [features[f] for f in self.in_features]
            x = self._shared_roi_transform(features, [x.pred_boxes for x in instances])
            return self.mask_head(x, instances)
        else:
            return instances


@ROI_HEADS_REGISTRY.register()
class StandardROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches (boxes and masks) directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    ignored_class_idx = -1

    def __init__(self, cfg, input_shape):
        super().__init__(cfg)
        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self._init_box_head(cfg, input_shape)
        self._init_mask_head(cfg, input_shape)
        self._init_keypoint_head(cfg, input_shape)
        self.ignored_class_idx = cfg.MODEL.IGNORED_CLASS_INDEX
        if cfg.MODEL.USE_DCR:
            self.use_dcr = True
        else:
            self.use_dcr = False
        if cfg.MODEL.USE_CLASSIFIER_ONLY:
            self.use_classifier_only = True
        else:
            self.use_classifier_only = False

    def _init_box_head(self, cfg, input_shape):
        # fmt: off
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )        
        self.box_predictor = FastRCNNOutputLayers(cfg, self.box_head.output_shape)

    def _init_mask_head(self, cfg, input_shape):
        # fmt: off
        self.mask_on           = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.mask_head = build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def _init_keypoint_head(self, cfg, input_shape):
        # fmt: off
        self.keypoint_on  = cfg.MODEL.KEYPOINT_ON
        if not self.keypoint_on:
            return
        pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)  # noqa
        sampling_ratio    = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.keypoint_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.keypoint_head = build_keypoint_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        dcr_network: Dict[str, torch.Tensor],
        targets: Optional[List[Instances]] = None,
        batched_inputs: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            # losses = self._forward_box(features, proposals)
            losses, predictions = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            # return proposals, losses
            return proposals, losses, predictions
        else:
            pred_instances = self._forward_box(features, proposals, dcr_network, batched_inputs)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)

            if self.use_dcr:
                pred_instances = self._forward_box_with_dcr(features, pred_instances, dcr_network, batched_inputs)

            return pred_instances, {}

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances], dcr_network: Dict[str, torch.Tensor] = None, batched_inputs: Optional[List[Instances]] = None
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            # return self.box_predictor.losses(predictions, proposals)
            return self.box_predictor.losses(predictions, proposals), predictions
        else:
            # pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            if self.use_classifier_only:
                pred_instances, _ = self.box_predictor.inference_classification_only(predictions, proposals)
            else:
                pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            
            # if self.use_dcr:
            #     # pred_instances, _ = self.box_predictor.inference_with_dcr(predictions, proposals, dcr_network, batched_inputs)
            #     pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            # else:
            #     pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            
            return pred_instances

    def _forward_box_with_dcr(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances], dcr_network: Dict[str, torch.Tensor] = None, batched_inputs: Optional[List[Instances]] = None
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.in_features]
        box_features = self.box_pooler(features, [x.pred_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            # return self.box_predictor.losses(predictions, proposals)
            return self.box_predictor.losses(predictions, proposals), predictions
        else:
            if self.use_dcr:
                pred_instances, _ = self.box_predictor.inference_with_dcr(predictions, proposals, dcr_network, batched_inputs)
            else:
                pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            
            return pred_instances

    def _forward_mask(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            return self.mask_head(mask_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            return self.mask_head(mask_features, instances)

    def _forward_keypoint(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]

        if self.training:
            # The loss is defined on positive proposals with at >=1 visible keypoints.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposals = select_proposals_with_visible_keypoints(proposals)
            proposal_boxes = [x.proposal_boxes for x in proposals]

            keypoint_features = self.keypoint_pooler(features, proposal_boxes)
            return self.keypoint_head(keypoint_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            keypoint_features = self.keypoint_pooler(features, pred_boxes)
            return self.keypoint_head(keypoint_features, instances)



#################### New ROI Head ####################

@ROI_HEADS_REGISTRY.register()
class _3DCarROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches (boxes and masks) directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """
    
    def __init__(self, cfg, input_shape):
        super().__init__(cfg)
        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self._init_3dcar_head(cfg, input_shape)

    def _init_3dcar_head(self, cfg, input_shape):
        # fmt: off
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # fmt: on

        self.num_pts = cfg.MODEL.OPT_3DCAR.NUM_POINTS
        self.batch_size_per_image = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE_3DCAR

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        # in_channels = [input_shape[f].channels for f in self.in_features]
        in_channels = [input_shape['p3'].channels] 
        
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        # self.box_pooler = ROIPooler(
        #     output_size=pooler_resolution,
        #     scales=pooler_scales,
        #     sampling_ratio=sampling_ratio,
        #     pooler_type=pooler_type,
        # )
        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=[pooler_scales[1]],
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.box_predictor = FastRCNNOutputLayers_3DCar(cfg, self.box_head.output_shape)
    

    @torch.no_grad()
    def label_and_sample_proposals_3dcar(
        self, targets: List[Instances]
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_sample_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """

        direction_map10 = [-1, 6, 5, 4, 7, 0, 3, 8, 1, 2]
        
        gt_boxes_3d = [x.gt_boxes_3d for x in targets]
        num_gts = [len(x.gt_boxes_3d) for x in targets]
        

        new_targets = []
        for idx in range(len(targets)):
            boxes_pool = []
            targets_3d = []
            gt_directions = []
            nbatchs = self.batch_size_per_image
            if num_gts[idx] > nbatchs:
                nbatchs = num_gts[idx]
            for i in range(nbatchs):
            # for i in range(self.batch_size_per_image):
                directions = direction_map10[targets[idx][i % num_gts[idx]].gt_directions]
                if directions == -1:
                    continue
                # gt_directions.append(directions)                
                shapes     = targets[idx][i % num_gts[idx]].gt_shapes
                points     = targets[idx][i % num_gts[idx]].gt_boxes_3d
                num_pts    = targets[idx][i % num_gts[idx]].num_pts

                min_x = min(points.tensor.cpu().numpy()[0][0::2][0:num_pts])
                min_y = min(points.tensor.cpu().numpy()[0][1::2][0:num_pts])
                max_x = max(points.tensor.cpu().numpy()[0][0::2][0:num_pts])
                max_y = max(points.tensor.cpu().numpy()[0][1::2][0:num_pts])

                w = max_x - min_x + 1
                h = max_y - min_y + 1

                if i / num_gts[idx] > 0:
                    crop_w = w * (0.5 + 0.5 * random.random())
                    crop_h = h * (0.5 + 0.5 * random.random())
                    crop_x = (w - crop_w) * random.random()
                    crop_y = (h - crop_h) * random.random()

                    min_x = min_x + crop_x
                    min_y = min_y + crop_y
                    max_x = min_x + crop_w - 1
                    max_y = min_y + crop_h - 1
                    w = crop_w
                    h = crop_h
                    
                    # clipping                    
                    for j in range(num_pts):
                        points.tensor[0][j] = max(min_x, min(points.tensor[0][j], max_x))
                        points.tensor[0][2 * j + 1] = max(min_y, min(points.tensor[0][2 * j + 1], max_y))
                

                x = points.tensor.cpu().numpy()[0][0::2][0:num_pts]
                y = points.tensor.cpu().numpy()[0][1::2][0:num_pts]
                
               
                if shapes != -1 and points.tensor[0][0] != -1:
                    valid_pts = 0
                    if (self.num_pts == 8 or self.num_pts == 16) and directions != 0:
                        target = [0] * self.num_pts
                        side = 0

                        boxes_pool.append([min_x, min_y, max_x, max_y])

                        if shapes == 0 and directions != 1 and directions != 5:
                            if y[2] >= y[3]:
                                target[0] = x[0]
                                target[1] = y[0]
                                target[2] = x[0]
                                target[3] = y[0]
                                target[4] = x[2]
                                target[5] = y[2]
                                target[6] = x[2]
                                target[7] = y[2]
                                target[8] = x[1]
                                target[9] = y[1]
                                target[10] = x[1]
                                target[11] = y[1]
                                target[12] = x[3]
                                target[13] = y[3]
                                target[14] = x[3]
                                target[15] = y[3]
                                side = 1
                            else:
                                target[0] = x[1]
                                target[1] = y[1]
                                target[2] = x[1]
                                target[3] = y[1]
                                target[4] = x[3]
                                target[5] = y[3]
                                target[6] = x[3]
                                target[7] = y[3]
                                target[8] = x[0]
                                target[9] = y[0]
                                target[10] = x[0]
                                target[11] = y[0]
                                target[12] = x[2]
                                target[13] = y[2]
                                target[14] = x[2]
                                target[15] = y[2]
                                side = 2
                        else:
                            target[0] = x[0]
                            target[1] = y[0]
                            target[2] = x[1]
                            target[3] = y[1]
                            target[4] = x[2]
                            target[5] = y[2]
                            target[6] = x[3]
                            target[7] = y[3]
                            if shapes == 0:
                                target[8] = x[0] + (x[1] - x[0]) * 0.1
                                target[9] = y[0] + (y[2] - y[0]) * 0.1
                                target[10] = x[1] - (x[1] - x[0]) * 0.1
                                target[11] = y[1] + (y[3] - y[1]) * 0.1
                                target[12] = x[2] + (x[3] - x[2]) * 0.1
                                target[13] = y[2] - (y[2] - y[0]) * 0.1
                                target[14] = x[3] - (x[3] - x[2]) * 0.1
                                target[15] = y[3] - (y[3] - y[1]) * 0.1
                            elif shapes == 1:
                                dist1 = math.sqrt((x[4] - x[5]) * (x[4] - x[5]) + (y[4] - y[5]) * (y[4] - y[5]))
                                dist2 = math.sqrt((x[0] - x[2]) * (x[0] - x[2]) + (y[0] - y[2]) * (y[0] - y[2]))                            
                                scale = dist1 / dist2 if dist2 > 0 or dist2 < 0 else 0
                                target[8] = x[4]
                                target[9] = y[4]
                                target[10] = (x[4] + (x[1] - x[0]) * scale + x[5] + (x[1] - x[2]) * scale) / 2
                                target[11] = (y[4] + (y[1] - y[0]) * scale + y[5] + (y[1] - y[2]) * scale) / 2
                                target[12] = x[5]
                                target[13] = y[5]
                                target[14] = (x[4] + (x[3] - x[0]) * scale + x[5] + (x[3] - x[2]) * scale) / 2
                                target[15] = (y[4] + (y[3] - y[0]) * scale + y[5] + (y[3] - y[2]) * scale) / 2
                                side = 2
                            elif shapes == 2:
                                dist1 = math.sqrt((x[4] - x[5]) * (x[4] - x[5]) + (y[4] - y[5]) * (y[4] - y[5]))
                                dist2 = math.sqrt((x[1] - x[3]) * (x[1] - x[3]) + (y[1] - y[3]) * (y[1] - y[3]))                            
                                scale = dist1 / dist2 if dist2 > 0 or dist2 < 0 else 0
                                target[8] = (x[4] + (x[0] - x[1]) * scale + x[5] + (x[0] - x[3]) * scale) / 2
                                target[9] = (y[4] + (y[0] - y[1]) * scale + y[5] + (y[0] - y[3]) * scale) / 2
                                target[10] = x[4]
                                target[11] = y[4]
                                target[12] = (x[4] + (x[2] - x[1]) * scale + x[5] + (x[2] - x[3]) * scale) / 2
                                target[13] = (y[4] + (y[2] - y[1]) * scale + y[5] + (y[2] - y[3]) * scale) / 2
                                target[14] = x[5]
                                target[15] = y[5]
                                side = 1
                            elif shapes == 3:
                                dist1 = math.sqrt((x[4] - x[5]) * (x[4] - x[5]) + (y[4] - y[5]) * (y[4] - y[5]))
                                dist2 = math.sqrt((x[0] - x[1]) * (x[0] - x[1]) + (y[0] - y[1]) * (y[0] - y[1]))                            
                                scale = dist1 / dist2 if dist2 > 0 or dist2 < 0 else 0
                                target[8] = x[4]
                                target[9] = y[4]
                                target[10] = x[5]
                                target[11] = y[5]
                                target[12] = (x[4] + (x[2] - x[0]) * scale + x[5] + (x[2] - x[1]) * scale) / 2
                                target[13] = (y[4] + (y[2] - y[0]) * scale + y[5] + (y[2] - y[1]) * scale) / 2
                                target[14] = (x[4] + (x[3] - x[0]) * scale + x[5] + (x[3] - x[1]) * scale) / 2
                                target[15] = (y[4] + (y[3] - y[0]) * scale + y[5] + (y[3] - y[1]) * scale) / 2
                            elif shapes == 4:
                                dist1 = math.sqrt((x[4] - x[5]) * (x[4] - x[5]) + (y[4] - y[5]) * (y[4] - y[5]))
                                dist2 = math.sqrt((x[0] - x[2]) * (x[0] - x[2]) + (y[0] - y[2]) * (y[0] - y[2]))
                                scale = dist1 / dist2 if dist2 > 0 or dist2 < 0 else 0
                                target[8] = x[4]
                                target[9] = y[4]
                                target[10] = x[6]
                                target[11] = y[6]
                                target[12] = x[5]
                                target[13] = y[5]
                                target[14] = (x[4] + (x[3] - x[0]) * scale + x[5] + (x[3] - x[2]) * scale + x[6] + (x[3] - x[1]) * scale) / 3
                                target[15] = (y[4] + (y[3] - y[0]) * scale + y[5] + (y[3] - y[2]) * scale + y[6] + (y[3] - y[1]) * scale) / 3
                                side = 2
                            elif shapes == 5:
                                dist1 = math.sqrt((x[4] - x[5]) * (x[4] - x[5]) + (y[4] - y[5]) * (y[4] - y[5]))
                                dist2 = math.sqrt((x[1] - x[3]) * (x[1] - x[3]) + (y[1] - y[3]) * (y[1] - y[3]))
                                scale = dist1 / dist2 if dist2 > 0 or dist2 < 0 else 0
                                target[8] = x[6]
                                target[9] = y[6]
                                target[10] = x[4]
                                target[11] = y[4]
                                target[12] = (x[4] + (x[2] - x[1]) * scale + x[5] + (x[2] - x[3]) * scale + x[6] + (x[2] - x[0]) * scale) / 3
                                target[13] = (y[4] + (y[2] - y[1]) * scale + y[5] + (y[2] - y[3]) * scale + y[6] + (y[2] - y[0]) * scale) / 3
                                target[14] = x[5]
                                target[15] = y[5]
                                side = 1

                        # normalize
                        # here
                        mean_ = [0.0, 0.0]
                        std_ = [0.1, 0.1]

                        # setting tops
                        ntarget = [0] * (self.num_pts * 2)

                        ntarget[0] = ((target[0] - min_x) / w - mean_[0]) / std_[0]
                        ntarget[1] = ((target[1] - min_y) / h - mean_[1]) / std_[1]
                        ntarget[2] = ((target[2] - max_x) / w - mean_[0]) / std_[0]
                        ntarget[3] = ((target[3] - min_y) / h - mean_[1]) / std_[1]
                        ntarget[4] = ((target[4] - min_x) / w - mean_[0]) / std_[0]
                        ntarget[5] = ((target[5] - max_y) / h - mean_[1]) / std_[1]
                        ntarget[6] = ((target[6] - max_x) / w - mean_[0]) / std_[0]
                        ntarget[7] = ((target[7] - max_y) / h - mean_[1]) / std_[1]
                        ntarget[8] = ((target[8] - min_x) / w - mean_[0]) / std_[0]
                        ntarget[9] = ((target[9] - min_y) / h - mean_[1]) / std_[1]
                        ntarget[10] = ((target[10] - max_x) / w - mean_[0]) / std_[0]
                        ntarget[11] = ((target[11] - min_y) / h - mean_[1]) / std_[1]
                        ntarget[12] = ((target[12] - min_x) / w - mean_[0]) / std_[0]
                        ntarget[13] = ((target[13] - max_y) / h - mean_[1]) / std_[1]
                        ntarget[14] = ((target[14] - max_x) / w - mean_[0]) / std_[0]
                        ntarget[15] = ((target[15] - max_y) / h - mean_[1]) / std_[1]
                        
                        valid_pts = 8
                        if self.num_pts == 16:
                            if side == 0:
                                ntarget[16] = ((target[10] - min_x) / w - mean_[0]) / std_[0]
                                ntarget[17] = ((target[11] - min_y) / h - mean_[1]) / std_[1]
                                ntarget[18] = ((target[2] - max_x) / w - mean_[0]) / std_[0]
                                ntarget[19] = ((target[3] - min_y) / h - mean_[1]) / std_[1]
                                ntarget[20] = ((target[14] - min_x) / w - mean_[0]) / std_[0]
                                ntarget[21] = ((target[15] - max_y) / h - mean_[1]) / std_[1]
                                ntarget[22] = ((target[6] - max_x) / w - mean_[0]) / std_[0]
                                ntarget[23] = ((target[7] - max_y) / h - mean_[1]) / std_[1]
                                ntarget[24] = ((target[0] - min_x) / w - mean_[0]) / std_[0]
                                ntarget[25] = ((target[1] - min_y) / h - mean_[1]) / std_[1]
                                ntarget[26] = ((target[8] - max_x) / w - mean_[0]) / std_[0]
                                ntarget[27] = ((target[9] - min_y) / h - mean_[1]) / std_[1]
                                ntarget[28] = ((target[4] - min_x) / w - mean_[0]) / std_[0]
                                ntarget[29] = ((target[5] - max_y) / h - mean_[1]) / std_[1]
                                ntarget[30] = ((target[12] - max_x) / w - mean_[0]) / std_[0]
                                ntarget[31] = ((target[13] - max_y) / h - mean_[1]) / std_[1]
                            elif side == 1:
                                ntarget[16] = ((target[2] - min_x) / w - mean_[0]) / std_[0]
                                ntarget[17] = ((target[3] - min_y) / h - mean_[1]) / std_[1]
                                ntarget[18] = ((target[10] - max_x) / w - mean_[0]) / std_[0]
                                ntarget[19] = ((target[11] - min_y) / h - mean_[1]) / std_[1]
                                ntarget[20] = ((target[6] - min_x) / w - mean_[0]) / std_[0]
                                ntarget[21] = ((target[7] - max_y) / h - mean_[1]) / std_[1]
                                ntarget[22] = ((target[14] - max_x) / w - mean_[0]) / std_[0]
                                ntarget[23] = ((target[15] - max_y) / h - mean_[1]) / std_[1]
                                ntarget[24] = ((target[0] - min_x) / w - mean_[0]) / std_[0]
                                ntarget[25] = ((target[1] - min_y) / h - mean_[1]) / std_[1]
                                ntarget[26] = ((target[8] - max_x) / w - mean_[0]) / std_[0]
                                ntarget[27] = ((target[9] - min_y) / h - mean_[1]) / std_[1]
                                ntarget[28] = ((target[4] - min_x) / w - mean_[0]) / std_[0]
                                ntarget[29] = ((target[5] - max_y) / h - mean_[1]) / std_[1]
                                ntarget[30] = ((target[12] - max_x) / w - mean_[0]) / std_[0]
                                ntarget[31] = ((target[13] - max_y) / h - mean_[1]) / std_[1]
                            elif side == 2:
                                ntarget[16] = ((target[10] - min_x) / w - mean_[0]) / std_[0]
                                ntarget[17] = ((target[11] - min_y) / h - mean_[1]) / std_[1]
                                ntarget[18] = ((target[2] - max_x) / w - mean_[0]) / std_[0]
                                ntarget[19] = ((target[3] - min_y) / h - mean_[1]) / std_[1]
                                ntarget[20] = ((target[14] - min_x) / w - mean_[0]) / std_[0]
                                ntarget[21] = ((target[15] - max_y) / h - mean_[1]) / std_[1]
                                ntarget[22] = ((target[6] - max_x) / w - mean_[0]) / std_[0]
                                ntarget[23] = ((target[7] - max_y) / h - mean_[1]) / std_[1]
                                ntarget[24] = ((target[8] - min_x) / w - mean_[0]) / std_[0]
                                ntarget[25] = ((target[9] - min_y) / h - mean_[1]) / std_[1]
                                ntarget[26] = ((target[0] - max_x) / w - mean_[0]) / std_[0]
                                ntarget[27] = ((target[1] - min_y) / h - mean_[1]) / std_[1]
                                ntarget[28] = ((target[12] - min_x) / w - mean_[0]) / std_[0]
                                ntarget[29] = ((target[13] - max_y) / h - mean_[1]) / std_[1]
                                ntarget[30] = ((target[4] - max_x) / w - mean_[0]) / std_[0]
                                ntarget[31] = ((target[5] - max_y) / h - mean_[1]) / std_[1]
                            valid_pts = valid_pts + 8
                    
                        targets_3d.append(ntarget)
                        gt_directions.append(directions)

                        if valid_pts != 16:
                            print('debug')

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            new_target = Instances(targets[idx]._image_size)
            new_target.gt_points_3d = torch.tensor(targets_3d, dtype=torch.float64, device=device)            
            new_target.gt_rois= Boxes(boxes_pool).to(device)
            new_target.gt_directions = torch.tensor(gt_directions, dtype=torch.int64, device=device)

            new_targets.append(new_target)
            
        return new_targets


    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets
            new_targets = self.label_and_sample_proposals_3dcar(targets)
        del targets

        if self.training:
            losses = self._forward_box(features, new_targets)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            # pred_instances = self.forward_with_given_boxes(features, pred_instances)
            # return pred_instances, {}
            return pred_instances

    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        # features = [features[f] for f in self.in_features]
        features = [features['p3']]
        box_features = self.box_pooler(features, [x.gt_rois for x in proposals])
        # # box_features = self.box_pooler(features, [x.pred_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return self.box_predictor.losses(predictions, proposals)
        else:
            # pred_instances = Instances(proposals[0].image_size)
            # pred_instances.pred_3dbox = Boxes3D(predictions[1])
            # pred_instances.dir_scores = predictions[0]
            # pred_instances.pred_dir_classes

            pred_instances = self.box_predictor.inference(predictions, proposals)
            return pred_instances




@ROI_HEADS_REGISTRY.register()
class DcrROIHead(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches (boxes and masks) directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    ignored_class_idx = -1

    def __init__(self, cfg, input_shape):
        super().__init__(cfg)        
        self.global_avg_pool = torch.nn.AvgPool2d(1, stride=1)

        self.in_features = 'res5'
        self._init_box_head(cfg, input_shape)
        self.ignored_class_idx = cfg.MODEL.IGNORED_CLASS_INDEX

    def _init_box_head(self, cfg, input_shape):
        in_channels = input_shape[self.in_features].channels
        # h = int(cfg.MODEL.DCR.IMAGE_SHAPE[0] / input_shape[self.in_features].stride)
        # w = int(cfg.MODEL.DCR.IMAGE_SHAPE[1] / input_shape[self.in_features].stride)
        h = 1
        w = 1

        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=h, width=w)
        )
        self.box_predictor = FastRCNNOutputLayers(cfg, self.box_head.output_shape)
    
    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}


    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return self.box_predictor.losses(predictions, proposals)
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances    