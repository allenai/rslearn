"""Normalization transforms."""

import warnings
from typing import Any

import torch

from rslearn.train.model_context import RasterImage

from .transform import Transform


class Normalize(Transform):
    """Normalize one or more input images with mean and standard deviation."""

    def __init__(
        self,
        mean: float | list[float],
        std: float | list[float],
        valid_range: (
            tuple[float, float] | tuple[list[float], list[float]] | None
        ) = None,
        selectors: list[str] = ["image"],
        bands: list[int] | None = None,
        num_bands: int | None = None,
    ) -> None:
        """Initialize a new Normalize.

        Result will be (input - mean) / std.

        Args:
            mean: a single value or one mean per channel
            std: a single value or one std per channel (must match the shape of mean)
            valid_range: optionally clip to a minimum and maximum value
            selectors: image items to transform
            bands: optionally restrict the normalization to these band indices. If set,
                mean and std must either be one value, or have length equal to the
                number of band indices passed here.
            num_bands: deprecated, no longer used. Will be removed after 2026-04-01.
        """
        super().__init__()

        if num_bands is not None:
            warnings.warn(
                "num_bands is deprecated and no longer used. "
                "It will be removed after 2026-04-01.",
                FutureWarning,
            )

        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

        if valid_range:
            self.valid_min = torch.tensor(valid_range[0])
            self.valid_max = torch.tensor(valid_range[1])
        else:
            self.valid_min = None
            self.valid_max = None

        self.selectors = selectors
        self.bands = torch.tensor(bands) if bands is not None else None

    def apply_image(self, image: RasterImage) -> RasterImage:
        """Normalize the specified image.

        Args:
            image: the image to transform.
        """
        # Get mean/std with singleton dims for broadcasting over CTHW.
        if len(self.mean.shape) == 0:
            # Scalar - broadcasts naturally.
            mean, std = self.mean, self.std
        else:
            # Vector of length C - add singleton dims for T, H, W.
            mean = self.mean[:, None, None, None]
            std = self.std[:, None, None, None]

        if self.bands is not None:
            # Normalize only specific band indices.
            image.image[self.bands] = (image.image[self.bands] - mean) / std
            if self.valid_min is not None:
                image.image[self.bands] = torch.clamp(
                    image.image[self.bands],
                    min=self.valid_min,
                    max=self.valid_max,
                )
        else:
            image.image = (image.image - mean) / std
            if self.valid_min is not None:
                image.image = torch.clamp(
                    image.image, min=self.valid_min, max=self.valid_max
                )
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
