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
            (input_dicts, target_dicts) where the entry corresponding to
            output_selector contains the concatenated RasterImage.
        """
        tensors: list[torch.Tensor] = []
        timestamps: list[tuple[datetime, datetime]] | None = None

        for selector, wanted_bands in self.selections.items():
            image = read_selector(input_dict, target_dict, selector)
            if wanted_bands:
                tensors.append(image.image[wanted_bands, :, :])
            else:
                tensors.append(image.image)
            if timestamps is None and image.timestamps is not None:
                # assume all concatenated modalities have the same
                # number of timestamps
                timestamps = image.timestamps

        result = RasterImage(
            torch.concatenate(tensors, dim=self.concatenate_dim),
            timestamps=timestamps,
        )
        write_selector(input_dict, target_dict, self.output_selector, result)
        return input_dict, target_dict
