"""Faster R-CNN decoder for object detection tasks."""

import collections
from typing import Any

import torch
import torchvision


class NoopTransform(torch.nn.Module):
    """A placeholder transform used with torchvision detection model."""

    def __init__(self) -> None:
        """Create a new NoopTransform."""
        super().__init__()

        self.transform = (
            torchvision.models.detection.transform.GeneralizedRCNNTransform(
                min_size=800,
                max_size=800,
                image_mean=[],
                image_std=[],
            )
        )

    def forward(
        self, images: list[torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> tuple[
        torchvision.models.detection.image_list.ImageList, dict[str, torch.Tensor]
    ]:
        """Transform the specified images and targets.

        Simply creates an ImageList object wrapping the provided images.

        Args:
            images: the images.
            targets: the targets (unmodified).

        Returns:
            wrapped images and unmodified targets
        """
        images = self.transform.batch_images(images, size_divisible=32)
        image_sizes = [(image.shape[1], image.shape[2]) for image in images]
        image_list = torchvision.models.detection.image_list.ImageList(
            images, image_sizes
        )
        return image_list, targets


class FasterRCNN(torch.nn.Module):
    """Faster R-CNN head for predicting bounding boxes.

    It inputs multi-scale features, using each feature map to predict ROIs and then
    processing the features within each ROI prediction to get final bounding box
    predictions.
    """

    def __init__(
        self,
        downsample_factors: list[int],
        num_channels: int,
        num_classes: int,
        anchor_sizes: list[list[int]],
        instance_segmentation: bool = False,
        box_score_thresh: float = 0.05,
    ) -> None:
        """Create a new FasterRCNN.

        Args:
            downsample_factors: list indicating the resolution of each feature map in
                the multi-scale features that this module will input. downsample_factor
                indicates that the resolution of that feature map is
                1/downsample_factor.
            num_channels: number of channels in each feature map (all the feature maps
                must have same number of channels, can use Fpn for this).
            num_classes: number of classes to predict.
            anchor_sizes: the anchor sizes to use for the different prediction heads.
            instance_segmentation: whether to predict segmentation mask in addition to
                bounding box for each object instance.
            box_score_thresh: during inference, only return bounding boxes with score
                greater than this threshold.
        """
        super().__init__()
        featmap_names = [f"feat{i}" for i in range(len(downsample_factors))]
        self.noop_transform = NoopTransform()

        # RPN
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = (
            torchvision.models.detection.anchor_utils.AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        )
        rpn_head = torchvision.models.detection.rpn.RPNHead(
            num_channels, rpn_anchor_generator.num_anchors_per_location()[0]
        )
        rpn_fg_iou_thresh = 0.7
        rpn_bg_iou_thresh = 0.3
        rpn_batch_size_per_image = 256
        rpn_positive_fraction = 0.5
        rpn_pre_nms_top_n = dict(training=2000, testing=2000)
        rpn_post_nms_top_n = dict(training=2000, testing=2000)
        rpn_nms_thresh = 0.7
        self.rpn = torchvision.models.detection.rpn.RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
        )

        # ROI
        box_roi_pool = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=featmap_names, output_size=7, sampling_ratio=2
        )
        box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead(
            num_channels * box_roi_pool.output_size[0] ** 2, 1024
        )
        box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            1024, num_classes
        )
        box_fg_iou_thresh = 0.5
        box_bg_iou_thresh = 0.5
        box_batch_size_per_image = 512
        box_positive_fraction = 0.25
        bbox_reg_weights = None
        box_nms_thresh = 0.5
        box_detections_per_img = 100
        self.roi_heads = torchvision.models.detection.roi_heads.RoIHeads(
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )

        if instance_segmentation:
            # Use Mask R-CNN stuff.
            self.roi_heads.mask_roi_pool = torchvision.ops.MultiScaleRoIAlign(
                featmap_names=featmap_names, output_size=14, sampling_ratio=2
            )

            mask_layers = (256, 256, 256, 256)
            mask_dilation = 1
            self.roi_heads.mask_head = (
                torchvision.models.detection.mask_rcnn.MaskRCNNHeads(
                    num_channels, mask_layers, mask_dilation
                )
            )

            mask_predictor_in_channels = 256
            mask_dim_reduced = 256
            self.roi_heads.mask_predictor = (
                torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
                    mask_predictor_in_channels, mask_dim_reduced, num_classes
                )
            )

    def forward(
        self,
        features: list[torch.Tensor],
        inputs: list[dict[str, Any]],
        targets: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Compute the detection outputs and loss from features.

        Args:
            features: multi-scale feature maps.
            inputs: original inputs, should cotnain image key for original image size.
            targets: should contain class key that stores the class label.

        Returns:
            tuple of outputs and loss dict
        """
        # Fix target labels to be 1 size in case it's empty.
        # For some reason this is needed.
        if targets:
            for i, target_dict in enumerate(targets):
                if len(target_dict["labels"]) != 0:
                    continue
                targets[i] = dict(
                    target_dict,
                    labels=torch.zeros(
                        (1,), dtype=torch.int64, device=target_dict["labels"].device
                    ),
                )

        image_list = [inp["image"] for inp in inputs]
        images, targets = self.noop_transform(image_list, targets)

        feature_dict = collections.OrderedDict()
        for i, feat_map in enumerate(features):
            feature_dict[f"feat{i}"] = feat_map

        proposals, proposal_losses = self.rpn(images, feature_dict, targets)
        detections, detector_losses = self.roi_heads(
            feature_dict, proposals, images.image_sizes, targets
        )

        losses = {}
        losses.update(proposal_losses)
        losses.update(detector_losses)

        return detections, losses
