"""Global pooling decoder for spatial feature maps."""

from typing import Any, Literal

import torch

from rslearn.train.model_context import ModelContext

from .component import FeatureMaps, IntermediateComponent


class GlobalPool(IntermediateComponent):
    """Apply global pooling to reduce spatial dimensions to 1x1.

    This component applies global average or max pooling over the spatial dimensions
    of input feature maps, producing 1x1 feature maps that can be used with
    EmbeddingHead or other downstream components.
    """

    def __init__(
        self,
        mode: Literal["mean", "max"] = "mean",
    ) -> None:
        """Create a new GlobalPool.

        Args:
            mode: the pooling mode, either "mean" for global average pooling or
                "max" for global max pooling. Defaults to "mean".
        """
        super().__init__()
        if mode not in ("mean", "max"):
            raise ValueError(f"mode must be 'mean' or 'max', got '{mode}'")
        self.mode = mode

    def forward(self, intermediates: Any, context: ModelContext) -> FeatureMaps:
        """Apply global pooling on the feature maps.

        Args:
            intermediates: output from the previous model component, which must be
                a FeatureMaps.
            context: the model context.

        Returns:
            globally pooled feature maps with spatial dimensions reduced to 1x1.
        """
        if not isinstance(intermediates, FeatureMaps):
            raise ValueError("input to GlobalPool must be FeatureMaps")

        pooled_features = []
        for feat in intermediates.feature_maps:
            # feat is BCHW
            if self.mode == "mean":
                pooled = feat.mean(dim=(2, 3), keepdim=True)
            else:  # max
                pooled = torch.amax(feat, dim=(2, 3), keepdim=True)
            pooled_features.append(pooled)

        return FeatureMaps(pooled_features)
