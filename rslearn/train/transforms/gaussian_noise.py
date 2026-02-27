"""Gaussian noise augmentation transform."""

from typing import Any

import torch

from rslearn.train.model_context import RasterImage

from .transform import Transform


class GaussianNoise(Transform):
    """Add Gaussian noise to inputs."""

    def __init__(
        self,
        selectors: list[str] = ["image"],
        std: float = 0.01,
        skip_missing: bool = False,
    ):
        """Initialize GaussianNoise.

        Args:
            selectors: inputs to augment.
            std: standard deviation of the noise.
            skip_missing: skip missing selectors.
        """
        super().__init__()
        self.selectors = selectors
        self.std = std
        self.skip_missing = skip_missing

    def apply_image(self, image: RasterImage) -> RasterImage:
        """Add noise."""
        image.image = image.image + torch.randn_like(image.image) * self.std
        return image

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply transform."""
        self.apply_fn(self.apply_image, input_dict, target_dict, self.selectors)
        return input_dict, target_dict
