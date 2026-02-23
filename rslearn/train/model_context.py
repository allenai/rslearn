"""Data classes to provide various context to models."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import torch

from rslearn.utils.geometry import PixelBounds, Projection
from rslearn.utils.raster_array import RasterArray


@dataclass
class RasterImage:
    """A raster image is a torch.tensor containing the images and their associated timestamps."""

    # image is a 4D CTHW tensor
    image: torch.Tensor
    # if timestamps is not None, len(timestamps) must match the T dimension of the tensor
    timestamps: list[tuple[datetime, datetime]] | None = None
    # expected_timestamps specifies the full set of expected timestamps for temporal alignment.
    # This is computed from layer config (window time_range + query config parameters) and this
    # is set only when PER_PERIOD_MOSAIC mode is used with max_matches == number of periods in
    # the window time_range.
    expected_timestamps: list[tuple[datetime, datetime]] | None = None

    def __post_init__(self) -> None:
        """Validate that the image tensor is 4D (CTHW)."""
        if self.image.dim() != 4:
            raise ValueError(
                f"RasterImage expects a 4D CTHW tensor, got {self.image.dim()}D "
                f"with shape {tuple(self.image.shape)}"
            )

    @property
    def shape(self) -> torch.Size:
        """The shape of the image."""
        return self.image.shape

    def dim(self) -> int:
        """The dim of the image."""
        return self.image.dim()

    @property
    def dtype(self) -> torch.dtype:
        """The image dtype."""
        return self.image.dtype

    def single_ts_to_chw_tensor(self) -> torch.Tensor:
        """Single timestep models expect single timestep inputs.

        This function (1) checks this raster image only has 1 timestep and
        (2) returns the tensor for that (single) timestep (going from CTHW to CHW).
        """
        if self.image.shape[1] != 1:
            raise ValueError(f"Expected a single timestep, got {self.image.shape[1]}")
        return self.image[:, 0]

    def get_hw_tensor(self) -> torch.Tensor:
        """Get a 2D HW tensor from a single-channel, single-timestep RasterImage.

        This function checks that C=1 and T=1, then returns the HW tensor.
        Useful for per-pixel labels like segmentation masks.
        """
        if self.image.shape[0] != 1:
            raise ValueError(
                f"Expected single channel (C=1), got {self.image.shape[0]}"
            )
        if self.image.shape[1] != 1:
            raise ValueError(
                f"Expected single timestep (T=1), got {self.image.shape[1]}"
            )
        return self.image[0, 0]

    @classmethod
    def from_raster_array(cls, ra: RasterArray) -> RasterImage:
        """Create a RasterImage from a numpy-based RasterArray.

        Converts the numpy CTHW array to a torch tensor and copies timestamps.

        Args:
            ra: the source RasterArray.

        Returns:
            a new RasterImage with the same data.
        """
        return cls(
            image=torch.as_tensor(ra.array.copy()),
            timestamps=ra.timestamps,
        )


@dataclass
class SampleMetadata:
    """Metadata pertaining to an example."""

    window_group: str
    window_name: str
    window_bounds: PixelBounds
    crop_bounds: PixelBounds
    crop_idx: int
    num_crops_in_window: int
    time_range: tuple[datetime, datetime] | None
    projection: Projection

    # Task name to differentiate different tasks.
    dataset_source: str | None


@dataclass
class ModelContext:
    """Context to pass to all model components."""

    # One input dict per example in the batch.
    inputs: list[dict[str, torch.Tensor | RasterImage]]
    # One SampleMetadata per example in the batch.
    metadatas: list[SampleMetadata]
    # Arbitrary dict that components can add to.
    context_dict: dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class ModelOutput:
    """The output from the Predictor.

    Args:
        outputs: output compatible with the configured Task.
        loss_dict: map from loss names to scalar tensors.
        metadata: arbitrary dict that can be used to store other outputs.
    """

    outputs: Iterable[Any]
    loss_dict: dict[str, torch.Tensor]
    metadata: dict[str, Any] = field(default_factory=lambda: {})
