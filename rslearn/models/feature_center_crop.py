"""Apply center cropping on a feature map."""

from typing import Any

import torch


class FeatureCenterCrop(torch.nn.Module):
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

    def forward(
        self, features: list[torch.Tensor], inputs: list[dict[str, Any]]
    ) -> list[torch.Tensor]:
        """Apply center cropping on the feature maps.

        Args:
            features: list of feature maps at different resolutions.
            inputs: original inputs (ignored).

        Returns:
            center cropped feature maps.
        """
        new_features = []
        for i, feat in enumerate(features):
            height, width = self.sizes[i]
            if feat.shape[2] < height or feat.shape[3] < width:
                raise ValueError(
                    "feature map is smaller than the desired height and width"
                )
            start_h = feat.shape[2] // 2 - height // 2
            start_w = feat.shape[3] // 2 - width // 2
            feat = feat[:, :, start_h : start_h + height, start_w : start_w + width]
            new_features.append(feat)
        return new_features
