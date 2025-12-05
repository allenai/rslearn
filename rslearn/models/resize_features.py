"""The ResizeFeatures module."""

from typing import Any

import torch

from rslearn.train.model_context import ModelContext

from .component import (
    FeatureMaps,
    IntermediateComponent,
)


class ResizeFeatures(IntermediateComponent):
    """Resize input features to new sizes."""

    def __init__(
        self,
        out_sizes: list[tuple[int, int]],
        mode: str = "bilinear",
    ):
        """Initialize a ResizeFeatures.

        Args:
            out_sizes: the output sizes of the feature maps. There must be one entry
                for each input feature map.
            mode: mode to pass to torch.nn.Upsample, e.g. "bilinear" (default) or
                "nearest".
        """
        super().__init__()
        layers = []
        for size in out_sizes:
            layers.append(
                torch.nn.Upsample(
                    size=size,
                    mode=mode,
                )
            )
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, intermediates: Any, context: ModelContext) -> FeatureMaps:
        """Resize the input feature maps to new sizes.

        Args:
            intermediates: the outputs from the previous component, which must be a FeatureMaps.
            context: the model context.

        Returns:
            resized feature maps
        """
        if not isinstance(intermediates, FeatureMaps):
            raise ValueError("input to ResizeFeatures must be a FeatureMaps")

        feat_maps = intermediates.feature_maps
        resized_feat_maps = [
            self.layers[idx](feat_map) for idx, feat_map in enumerate(feat_maps)
        ]
        return FeatureMaps(resized_feat_maps)
