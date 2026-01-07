"""Concatenate bands across multiple image inputs."""

from datetime import datetime
from enum import Enum
from typing import Any

import torch

from rslearn.train.model_context import RasterImage

from .transform import Transform, read_selector, write_selector


class ConcatenateDim(Enum):
    """Enum for concatenation dimensions."""

    CHANNEL = 0
    TIME = 1


class Concatenate(Transform):
    """Concatenate bands across multiple image inputs."""

    def __init__(
        self,
        selections: dict[str, list[int]],
        output_selector: str,
        concatenate_dim: ConcatenateDim | int = ConcatenateDim.TIME,
    ):
        """Initialize a new Concatenate.

        Args:
            selections: map from selector to list of band indices in that input to
                retain, or empty list to use all bands.
            output_selector: the output selector under which to save the concatenate image.
            concatenate_dim: the dimension against which to concatenate the inputs
        """
        super().__init__()
        self.selections = selections
        self.output_selector = output_selector
        self.concatenate_dim = (
            concatenate_dim.value
            if isinstance(concatenate_dim, ConcatenateDim)
            else concatenate_dim
        )

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply concatenation over the inputs and targets.

        Args:
            input_dict: the input
            target_dict: the target

        Returns:
            normalized (input_dicts, target_dicts) tuple
        """
        images = []
        return_raster_image: bool = False
        timestamps: list[tuple[datetime, datetime]] | None = None
        for selector, wanted_bands in self.selections.items():
            print(wanted_bands)
            image = read_selector(input_dict, target_dict, selector)
            if isinstance(image, torch.Tensor):
                if wanted_bands:
                    image = image[wanted_bands, :, :]
                images.append(image)
            elif isinstance(image, RasterImage):
                return_raster_image = True
                if wanted_bands:
                    images.append(image.image[wanted_bands, :, :])
                else:
                    images.append(image.image)
                if timestamps is None:
                    if image.timestamps is not None:
                        # assume all concatenated modalities have the same
                        # number of timestamps
                        timestamps = image.timestamps
        if return_raster_image:
            result = RasterImage(
                torch.concatenate(images, dim=self.concatenate_dim),
                timestamps=timestamps,
            )
        else:
            result = torch.concatenate(images, dim=self.concatenate_dim)
        write_selector(input_dict, target_dict, self.output_selector, result)
        return input_dict, target_dict
