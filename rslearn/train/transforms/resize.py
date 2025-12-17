"""Resize transform."""

from typing import Any

import torch
import torchvision
from torchvision.transforms import InterpolationMode

from .transform import Transform

INTERPOLATION_MODES = {
    "nearest": InterpolationMode.NEAREST,
    "nearest_exact": InterpolationMode.NEAREST_EXACT,
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
}


class Resize(Transform):
    """Resizes inputs to a target size."""

    def __init__(
        self,
        target_size: tuple[int, int],
        selectors: list[str] = [],
        interpolation: str = "nearest",
    ):
        """Initialize a resize transform.

        Args:
            target_size: the (height, width) to resize to.
            selectors: items to transform.
            interpolation: the interpolation mode to use for resizing.
                Must be one of "nearest", "nearest_exact", "bilinear", or "bicubic".
        """
        super().__init__()
        self.target_size = target_size
        self.selectors = selectors
        self.interpolation = INTERPOLATION_MODES[interpolation]

    def apply_resize(self, image: torch.Tensor) -> torch.Tensor:
        """Apply resizing on the specified image.

        If the image is 2D, it is unsqueezed to 3D and then squeezed
        back after resizing.

        Args:
            image: the image to transform.
        """
        if image.dim() == 2:
            image = image.unsqueeze(0)  # (H, W) -> (1, H, W)
            result = torchvision.transforms.functional.resize(
                image, self.target_size, self.interpolation
            )
            return result.squeeze(0)  # (1, H, W) -> (H, W)

        return torchvision.transforms.functional.resize(
            image, self.target_size, self.interpolation
        )

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply transform over the inputs and targets.

        Args:
            input_dict: the input
            target_dict: the target

        Returns:
            transformed (input_dicts, target_dicts) tuple
        """
        self.apply_fn(self.apply_resize, input_dict, target_dict, self.selectors)
        return input_dict, target_dict
