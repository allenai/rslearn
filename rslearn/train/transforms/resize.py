"""Resize transform."""

from typing import Any

import torchvision
from torchvision.transforms import InterpolationMode

from rslearn.train.model_context import RasterImage

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
        skip_missing: bool = False,
    ):
        """Initialize a resize transform.

        Args:
            target_size: the (height, width) to resize to.
            selectors: items to transform.
            interpolation: the interpolation mode to use for resizing.
                Must be one of "nearest", "nearest_exact", "bilinear", or "bicubic".
            skip_missing: if True, skip selectors that don't exist in the input/target
                dicts. Useful when working with optional inputs.
        """
        super().__init__()
        self.target_size = target_size
        self.selectors = selectors
        self.interpolation = INTERPOLATION_MODES[interpolation]
        self.skip_missing = skip_missing

    def apply_resize(self, image: RasterImage) -> RasterImage:
        """Apply resizing on the specified image.

        Args:
            image: the image to transform.
        """
        image.image = torchvision.transforms.functional.resize(
            image.image, self.target_size, self.interpolation
        )
        return image

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
