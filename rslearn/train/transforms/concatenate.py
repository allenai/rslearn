"""Concatenate bands across multiple image inputs."""

from datetime import datetime
from enum import Enum
from typing import Any

import torch

from rslearn.train.model_context import RasterImage

from .transform import Transform, read_selector, selector_exists, write_selector


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
        skip_missing: bool = False,
    ):
        """Initialize a new Concatenate.

        Args:
            selections: map from selector to list of band indices in that input to
                retain, or empty list to use all bands.
            output_selector: the output selector under which to save the concatenate image.
            concatenate_dim: the dimension against which to concatenate the inputs
            skip_missing: if True, selectors that are absent from the input/target dicts
                are silently skipped instead of raising. Useful when concatenating
                optional inputs (e.g. per-month layers where some months are missing).
        """
        super().__init__(skip_missing=skip_missing)
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
        concatenate_time = self.concatenate_dim == ConcatenateDim.TIME.value

        tensors: list[torch.Tensor] = []
        # For CHANNEL concatenation, all inputs share the same timesteps, so we keep the
        # first available timestamps. For TIME concatenation, timestamps are collected
        # across all inputs so that len(timestamps) matches the concatenated time
        # dimension (required by models that consume real per-timestep timestamps).
        channel_timestamps: list[tuple[datetime, datetime]] | None = None
        time_timestamps: list[tuple[datetime, datetime]] = []
        time_has_all_timestamps = True

        for selector, wanted_bands in self.selections.items():
            if self.skip_missing and not selector_exists(
                input_dict, target_dict, selector
            ):
                continue
            image = read_selector(input_dict, target_dict, selector)
            if wanted_bands:
                tensors.append(image.image[wanted_bands, :, :])
            else:
                tensors.append(image.image)

            if concatenate_time:
                if image.timestamps is not None:
                    time_timestamps.extend(image.timestamps)
                else:
                    time_has_all_timestamps = False
            elif channel_timestamps is None and image.timestamps is not None:
                channel_timestamps = image.timestamps

        if not tensors:
            raise ValueError(
                f"Concatenate produced no inputs for output_selector "
                f"'{self.output_selector}' (all selectors missing with skip_missing)."
            )

        if concatenate_time:
            timestamps = time_timestamps if time_has_all_timestamps else None
        else:
            timestamps = channel_timestamps

        result = RasterImage(
            torch.concatenate(tensors, dim=self.concatenate_dim),
            timestamps=timestamps,
        )
        write_selector(input_dict, target_dict, self.output_selector, result)
        return input_dict, target_dict
