"""Transforms related to Sentinel-1 data."""

from typing import Any

import torch

from rslearn.train.model_context import RasterImage

from .transform import Transform


class Sentinel1ToDecibels(Transform):
    """Convert Sentinel-1 data from raw intensity to or from decibels."""

    def __init__(
        self,
        selectors: list[str] = ["image"],
        from_decibels: bool = False,
        epsilon: float = 1e-6,
    ):
        """Initialize a new Sentinel1ToDecibels.

        Args:
            selectors: the input selectors to apply the transform on.
            from_decibels: convert from decibels to intensities instead of intensity to
                decibels.
            epsilon: when converting to decibels, clip the intensities to this minimum
                value to avoid log issues. This is mostly to avoid pixels that have no
                data with no data value being 0.
        """
        super().__init__()
        self.selectors = selectors
        self.from_decibels = from_decibels
        self.epsilon = epsilon

    def apply_image(self, image: RasterImage) -> RasterImage:
        """Normalize the specified image.

        Args:
            image: the image to transform.
        """
        if self.from_decibels:
            # Decibels to linear scale.
            image.image = torch.pow(10.0, image.image / 10.0)
        else:
            # Linear scale to decibels.
            image.image = 10 * torch.log10(torch.clamp(image.image, min=self.epsilon))
        return image

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply normalization over the inputs and targets.

        Args:
            input_dict: the input
            target_dict: the target

        Returns:
            normalized (input_dicts, target_dicts) tuple
        """
        self.apply_fn(self.apply_image, input_dict, target_dict, self.selectors)
        return input_dict, target_dict
