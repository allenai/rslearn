"""An upsampling layer."""

from typing import Any

import torch

from rslearn.train.model_context import ModelContext

from .component import (
    FeatureMaps,
    IntermediateComponent,
)


class Upsample(IntermediateComponent):
    """Upsamples each input feature map by the same factor."""

    def __init__(
        self,
        scale_factor: int,
        mode: str = "bilinear",
    ):
        """Initialize an Upsample.

        Args:
            scale_factor: the upsampling factor, e.g. 2 to double the size.
            mode: "nearest" or "bilinear".
        """
        super().__init__()
        self.layer = torch.nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, intermediates: Any, context: ModelContext) -> FeatureMaps:
        """Upsample each feature map by scale_factor.

        Args:
            intermediates: the output from the previous component, which must be a FeatureMaps.
            context: the model context.

        Returns:
            upsampled feature maps.
        """
        if not isinstance(intermediates, FeatureMaps):
            raise ValueError("input to Upsample must be a FeatureMaps")

        upsampled_feat_maps = [
            self.layer(feat_map) for feat_map in intermediates.feature_maps
        ]
        return FeatureMaps(upsampled_feat_maps)
