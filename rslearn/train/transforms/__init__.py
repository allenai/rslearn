"""rslearn transforms."""

from typing import Any

import torch


class Sequential(torch.nn.Module):
    """Sequentially apply provided transforms.

    Each transform should accept (input_dict, target_dict) and return an updated
    tuple.
    """

    def __init__(self, *args: Any) -> None:
        """Initialize a new Sequential from a list of transforms."""
        super().__init__()
        self.transforms = torch.nn.ModuleList(args)

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply each specified transform."""
        for transform in self.transforms:
            input_dict, target_dict = transform(input_dict, target_dict)
        return input_dict, target_dict
