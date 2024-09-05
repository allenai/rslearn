"""PickFeatures module."""

from typing import Any

import torch


class PickFeatures(torch.nn.Module):
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
        features: list[torch.Tensor],
        inputs: list[dict[str, Any]] | None = None,
        targets: list[dict[str, Any]] | None = None,
    ) -> list[torch.Tensor]:
        """Pick a subset of the features.

        Args:
            features: input features
            inputs: raw inputs, not used
            targets: targets, not used
        """
        return [features[idx] for idx in self.indexes]
