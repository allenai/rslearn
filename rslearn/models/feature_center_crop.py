"""Apply center cropping on a feature map."""

from typing import Any

from rslearn.train.model_context import ModelContext

from .component import FeatureMaps, IntermediateComponent


class FeatureCenterCrop(IntermediateComponent):
    """Apply center cropping on the input feature maps."""

    def __init__(
        self,
        sizes: list[tuple[int, int]],
    ) -> None:
        """Create a new FeatureCenterCrop.

        Only the center of each feature map will be retained and passed to the next
        module.

        Args:
            sizes: a list of (height, width) tuples, with one tuple for each input
                feature map.
        """
        super().__init__()
        self.sizes = sizes

    def forward(self, intermediates: Any, context: ModelContext) -> FeatureMaps:
        """Apply center cropping on the feature maps.

        Args:
            intermediates: output from the previous model component, which must be a FeatureMaps.
            context: the model context.

        Returns:
            center cropped feature maps.
        """
        if not isinstance(intermediates, FeatureMaps):
            raise ValueError("input to FeatureCenterCrop must be FeatureMaps")

        new_features = []
        for i, feat in enumerate(intermediates.feature_maps):
            height, width = self.sizes[i]
            if feat.shape[2] < height or feat.shape[3] < width:
                raise ValueError(
                    "feature map is smaller than the desired height and width"
                )
            start_h = feat.shape[2] // 2 - height // 2
            start_w = feat.shape[3] // 2 - width // 2
            feat = feat[:, :, start_h : start_h + height, start_w : start_w + width]
            new_features.append(feat)
        return FeatureMaps(new_features)
