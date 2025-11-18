"""SatlasPretrain models."""

from typing import Any

import satlaspretrain_models
import torch
import torch.nn.functional as F


class SatlasPretrain(torch.nn.Module):
    """SatlasPretrain backbones."""

    def __init__(
        self, model_identifier: str, fpn: bool = False, resize_to_pretrain: bool = False
    ) -> None:
        """Instantiate a new SatlasPretrain instance.

        Args:
            model_identifier: the checkpoint name from the table at
                https://github.com/allenai/satlaspretrain_models
            fpn: whether to include the feature pyramid network, otherwise only the
                Swin-v2-Transformer is used.
            resize_to_pretrain: whether to resize inputs to the pretraining input
                size (512 x 512)
        """
        super().__init__()
        weights_manager = satlaspretrain_models.Weights()
        self.model = weights_manager.get_pretrained_model(
            model_identifier=model_identifier, fpn=fpn, device="cpu"
        )

        if "SwinB" in model_identifier:
            self.backbone_channels = [
                [4, 128],
                [8, 256],
                [16, 512],
                [32, 1024],
            ]
        elif "SwinT" in model_identifier:
            self.backbone_channels = [
                [4, 96],
                [8, 192],
                [16, 384],
                [32, 768],
            ]
        elif "Resnet" in model_identifier:
            self.backbone_channels = [
                [4, 256],
                [8, 512],
                [16, 1024],
                [32, 2048],
            ]
        self.resize_to_pretrain = resize_to_pretrain

    def maybe_resize(self, data: torch.Tensor) -> list[torch.Tensor]:
        """Resize to pretraining sizes if resize_to_pretrain == True."""
        if self.resize_to_pretrain:
            return F.interpolate(
                data,
                size=(512, 512),
                mode="bilinear",
                align_corners=False,
            )
        else:
            return data

    def forward(self, inputs: list[dict[str, Any]]) -> list[torch.Tensor]:
        """Compute feature maps from the SatlasPretrain backbone.

        Inputs:
            inputs: input dicts that must include "image" key containing the image to
                process.
        """
        images = torch.stack([inp["image"] for inp in inputs], dim=0)
        return self.model(self.maybe_resize(images))

    def get_backbone_channels(self) -> list:
        """Returns the output channels of this model when used as a backbone.

        The output channels is a list of (downsample_factor, depth) that corresponds
        to the feature maps that the backbone returns. For example, an element [2, 32]
        indicates that the corresponding feature map is 1/2 the input resolution and
        has 32 channels.

        Returns:
            the output channels of the backbone as a list of (downsample_factor, depth)
            tuples.
        """
        return self.backbone_channels
