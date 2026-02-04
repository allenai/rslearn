"""Global pooling decoder for spatial feature maps."""

from typing import Any, Literal

import torch

from rslearn.train.model_context import ModelContext

from .component import FeatureMaps, FeatureVector, IntermediateComponent


class GlobalPool(IntermediateComponent):
    """Apply global pooling to reduce spatial dimensions.

    This component applies global average or max pooling over the spatial dimensions
    of input feature maps. By default, it produces FeatureVector (BxC) suitable for
    ClassificationHead or RegressionHead. When keep_spatial_dims=True, it produces
    1x1 FeatureMaps suitable for EmbeddingHead.
    """

    def __init__(
        self,
        mode: Literal["mean", "max"] = "mean",
        keep_spatial_dims: bool = False,
    ) -> None:
        """Create a new GlobalPool.

        Args:
            mode: the pooling mode, either "mean" for global average pooling or
                "max" for global max pooling. Defaults to "mean".
            keep_spatial_dims: if True, returns FeatureMaps with 1x1 spatial dimensions.
                If False (default), returns FeatureVector (BxC). Defaults to False.
        """
        super().__init__()
        if mode not in ("mean", "max"):
            raise ValueError(f"mode must be 'mean' or 'max', got '{mode}'")
        self.mode = mode
        self.keep_spatial_dims = keep_spatial_dims

    def forward(
        self, intermediates: Any, context: ModelContext
    ) -> FeatureMaps | FeatureVector:
        """Apply global pooling on the feature maps.

        Args:
            intermediates: output from the previous model component, which must be
                a FeatureMaps.
            context: the model context.

        Returns:
            If keep_spatial_dims=False (default): FeatureVector (BxC) suitable for
                ClassificationHead or RegressionHead.
            If keep_spatial_dims=True: FeatureMaps with 1x1 spatial dimensions suitable
                for EmbeddingHead.
        """
        if not isinstance(intermediates, FeatureMaps):
            raise ValueError("input to GlobalPool must be FeatureMaps")

        pooled_features = []
        for feat in intermediates.feature_maps:
            if self.mode == "mean":
                pooled = feat.mean(dim=(2, 3), keepdim=self.keep_spatial_dims)
            else:
                pooled = torch.amax(feat, dim=(2, 3), keepdim=self.keep_spatial_dims)
            pooled_features.append(pooled)

        if self.keep_spatial_dims:
            return FeatureMaps(pooled_features)
        else:
            if len(pooled_features) == 1:
                return FeatureVector(pooled_features[0])
            else:
                return FeatureVector(torch.cat(pooled_features, dim=1))
