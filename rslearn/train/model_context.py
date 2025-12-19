"""Data classes to provide various context to models."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import torch

from rslearn.utils.geometry import PixelBounds, Projection
from rslearn.utils.raster_format import RasterImage


@dataclass
class SampleMetadata:
    """Metadata pertaining to an example."""

    window_group: str
    window_name: str
    window_bounds: PixelBounds
    patch_bounds: PixelBounds
    patch_idx: int
    num_patches_in_window: int
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
