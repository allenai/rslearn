"""SatlasPretrain models."""

from typing import Any, Optional

import satlaspretrain_models
import torch
import torchvision


class SatlasPretrain(torch.nn.Module):
    """SatlasPretrain backbones."""

    def __init__(
        self,
        model_identifier: str,
        fpn: bool = False,
    ):
        """Instantiate a new SatlasPretrain instance.

        Args:
            model_identifier: the checkpoint name from the table at
                https://github.com/allenai/satlaspretrain_models
            fpn: whether to include the feature pyramid network, otherwise only the
                Swin-v2-Transformer is used.
        """
        super().__init__()
        weights_manager = satlaspretrain_models.Weights()
        self.model = weights_manager.get_pretrained_model(model_identifier=model_identifier, fpn=fpn)

    def forward(
        self, inputs: list[dict[str, Any]], targets: list[dict[str, Any]] = None
    ):
        """Compute feature maps from the SatlasPretrain backbone.

        Inputs:
            inputs: input dicts that must include "image" key containing the image to
                process.
            targets: target dicts that are ignored
        """
        images = torch.stack([inp["image"] for inp in inputs], dim=0)
        return self.model(images)
