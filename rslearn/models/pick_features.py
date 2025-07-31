"""PickFeatures module."""

from typing import Any

import torch


class PickFeatures(torch.nn.Module):
    """Picks a subset of feature maps in a multi-scale feature map list."""

    def __init__(self, indexes: list[int], collapse: bool = False):
        """Create a new PickFeatures.

        Args:
            indexes: the indexes of the input feature map list to select.
            collapse: return one feature map instead of list. If enabled, indexes must
                consist of one index.
        """
        super().__init__()
        self.indexes = indexes
        self.collapse = collapse

        if self.collapse and len(self.indexes) != 1:
            raise ValueError("if collapse is enabled, must get exactly one index")

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
        new_features = [features[idx] for idx in self.indexes]
        if self.collapse:
            assert len(new_features) == 1
            return new_features[0]
        else:
            return new_features
