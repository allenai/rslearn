"""A sub-pixel (PixelShuffle) reshaping layer."""

from typing import Any

import torch

from rslearn.train.model_context import ModelContext

from .component import (
    FeatureMaps,
    IntermediateComponent,
)


class PixelShuffle(IntermediateComponent):
    """Rearranges feature maps from (C*r^2, H, W) to (C, H*r, W*r) via sub-pixel shuffle.

    This is the "reshape" alternative to interpolation-based upsampling: place a Conv
    that outputs ``num_classes * upscale_factor^2`` channels before this layer, and each
    1/r-resolution token's prediction is reshaped into its r-by-r block of pixels. For a
    linear segmentation probe on patch_size=r features, this lets a single linear layer
    predict every pixel within a patch (rather than upsampling a coarse prediction).
    """

    def __init__(self, upscale_factor: int):
        """Initialize a PixelShuffle.

        Args:
            upscale_factor: factor to increase spatial resolution by (typically the
                encoder patch size). Input channels must be divisible by its square.
        """
        super().__init__()
        self.layer = torch.nn.PixelShuffle(upscale_factor)

    def forward(self, intermediates: Any, context: ModelContext) -> FeatureMaps:
        """Apply sub-pixel shuffle to each feature map.

        Args:
            intermediates: the previous output, which must be a FeatureMaps.
            context: the model context.

        Returns:
            the reshaped feature maps.
        """
        if not isinstance(intermediates, FeatureMaps):
            raise ValueError("input to PixelShuffle must be a FeatureMaps")
        return FeatureMaps(
            [self.layer(feat_map) for feat_map in intermediates.feature_maps]
        )
