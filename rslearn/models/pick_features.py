"""PickFeatures module."""

from typing import Any

from rslearn.train.model_context import ModelContext

from .component import (
    FeatureMaps,
    IntermediateComponent,
)


class PickFeatures(IntermediateComponent):
    """Picks a subset of feature maps in a multi-scale feature map list."""

    def __init__(self, indexes: list[int]):
        """Create a new PickFeatures.

        Args:
            indexes: the indexes of the input feature map list to select.
        """
        super().__init__()
        self.indexes = indexes

    def forward(
        self,
        intermediates: Any,
        context: ModelContext,
    ) -> FeatureMaps:
        """Pick a subset of the features.

        Args:
            intermediates: the output from the previous component, which must be a FeatureMaps.
            context: the model context.
        """
        if not isinstance(intermediates, FeatureMaps):
            raise ValueError("input to PickFeatures must be FeatureMaps")

        new_features = [intermediates.feature_maps[idx] for idx in self.indexes]
        return FeatureMaps(new_features)
