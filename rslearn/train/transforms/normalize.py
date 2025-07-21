"""Normalization transforms."""

from typing import Any

import torch

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
            std: a single value or one std per channel
            valid_range: optionally clip to a minimum and maximum value
            selectors: image items to transform
            bands: optionally restrict the normalization to these bands
            num_bands: the number of bands per image, to distinguish different images
                in a time series. If set, then the bands list is repeated for each
                image, e.g. if bands=[2] then we apply normalization on images[2],
                images[2+num_bands], images[2+num_bands*2], etc.
        """
        super().__init__()
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
        self.num_bands = num_bands

    def apply_image(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize the specified image.

        Args:
            image: the image to transform.
        """
        if self.bands is not None:
            # User has provided band indices to normalize.
            # If num_bands is set, then we repeat these for each image in the input
            # image time series.
            band_indices = self.bands
            if self.num_bands:
                num_images = image.shape[0] // self.num_bands
                band_indices = torch.cat(
                    [
                        band_indices + image_idx * self.num_bands
                        for image_idx in range(num_images)
                    ],
                    dim=0,
                )

            image[band_indices] = (image[band_indices] - self.mean) / self.std
            if self.valid_min is not None:
                image[band_indices] = torch.clamp(
                    image[band_indices], min=self.valid_min, max=self.valid_max
                )
        else:
            image = (image - self.mean) / self.std
            if self.valid_min is not None:
                image = torch.clamp(image, min=self.valid_min, max=self.valid_max)
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
