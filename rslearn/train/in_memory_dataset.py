"""In-memory dataset wrappers that cache full windows and crop on access.

Suitable for small datasets that fit in memory. The cache avoids re-reading
from disk on every epoch while still allowing fresh random crops and
augmentations each time.
"""

import random
from dataclasses import replace
from typing import Any

import shapely
import torch

from rslearn.dataset import Window
from rslearn.log_utils import get_logger
from rslearn.train.all_crops_dataset import (
    crop_tensor_or_rasterimage,
    get_window_crop_options,
    pad_slice_protect,
)
from rslearn.train.dataset import ModelDataset
from rslearn.train.model_context import RasterImage, SampleMetadata
from rslearn.utils.geometry import STGeometry

logger = get_logger(__name__)


class InMemoryDataset(torch.utils.data.Dataset):
    """Base class for in-memory dataset wrappers.

    Caches full windows loaded via ModelDataset.get_raw_inputs so that
    repeated access (across epochs or crops) does not hit disk.

    Subclasses define __len__ and __getitem__ to control how crops are
    selected from the cached windows.
    """

    def __init__(
        self,
        dataset: ModelDataset,
        crop_size: tuple[int, int] | None = None,
    ):
        """Create a new InMemoryDataset.

        Args:
            dataset: the ModelDataset to wrap. Should be configured with
                crop_size=None (load entire windows) so that we can cache
                the full window and crop later.
            crop_size: the size of the crops to extract, or None to return
                full windows without cropping.
        """
        super().__init__()
        self.dataset = dataset
        self.crop_size = crop_size
        self.windows = self.dataset.get_dataset_examples()
        self.inputs = dataset.inputs
        self.window_cache: dict[
            int, tuple[dict[str, Any], dict[str, Any], SampleMetadata]
        ] = {}

    def get_raw_inputs(
        self, window_id: int
    ) -> tuple[dict[str, Any], dict[str, Any], SampleMetadata]:
        """Load a full window's raw inputs, with caching.

        Also pads raster inputs by crop size to protect slicing near
        right/bottom edges.

        Args:
            window_id: index into self.windows.

        Returns:
            a tuple of (raw_inputs, passthrough_inputs, metadata).
        """
        if window_id in self.window_cache:
            return self.window_cache[window_id]

        raw_inputs, passthrough_inputs, metadata = self.dataset.get_raw_inputs(
            window_id
        )
        # Pad rasters so that edge crops extending past the window are safe.
        # Not needed when returning full windows (crop_size is None).
        if self.crop_size is not None:
            pad_slice_protect(
                raw_inputs, passthrough_inputs, self.crop_size, self.inputs
            )

        self.window_cache[window_id] = (raw_inputs, passthrough_inputs, metadata)
        return self.window_cache[window_id]

    def _crop_input_dict(
        self,
        d: dict[str, Any],
        start_offset: tuple[int, int],
        end_offset: tuple[int, int],
        cur_geom: STGeometry,
    ) -> dict[str, Any]:
        """Crop a dictionary of inputs to the given bounds.

        Crop coordinates are scaled based on each input's resolution_factor.
        """
        cropped = {}
        for input_name, value in d.items():
            if isinstance(value, torch.Tensor | RasterImage):
                rf = self.inputs[input_name].resolution_factor
                scale = rf.numerator / rf.denominator
                scaled_start = (
                    int(start_offset[0] * scale),
                    int(start_offset[1] * scale),
                )
                scaled_end = (
                    int(end_offset[0] * scale),
                    int(end_offset[1] * scale),
                )
                cropped[input_name] = crop_tensor_or_rasterimage(
                    value, scaled_start, scaled_end
                )
            elif isinstance(value, list):
                cropped[input_name] = [
                    feat for feat in value if cur_geom.intersects(feat.geometry)
                ]
            else:
                raise ValueError("got input that is neither tensor nor feature list")
        return cropped

    def _finalize_sample(
        self,
        window_id: int,
        crop_bounds: tuple[int, int, int, int],
        crop_idx: int,
        num_crops: int,
    ) -> tuple[dict[str, Any], dict[str, Any], SampleMetadata]:
        """Produce a final (input_dict, target_dict, metadata) for one crop.

        Loads the cached window, slices the crop, runs task.process_inputs
        and transforms.

        Args:
            window_id: index into self.windows.
            crop_bounds: (col_min, row_min, col_max, row_max) of the crop.
            crop_idx: ordinal index of this crop within the window.
            num_crops: total number of crops in this window.

        Returns:
            a tuple of (input_dict, target_dict, metadata).
        """
        raw_inputs, passthrough_inputs, metadata = self.get_raw_inputs(window_id)
        bounds = metadata.crop_bounds

        cur_geom = STGeometry(metadata.projection, shapely.box(*crop_bounds), None)
        start_offset = (crop_bounds[0] - bounds[0], crop_bounds[1] - bounds[1])
        end_offset = (crop_bounds[2] - bounds[0], crop_bounds[3] - bounds[1])

        cur_raw_inputs = self._crop_input_dict(
            raw_inputs, start_offset, end_offset, cur_geom
        )
        cur_passthrough_inputs = self._crop_input_dict(
            passthrough_inputs, start_offset, end_offset, cur_geom
        )

        cur_metadata = replace(
            metadata,
            crop_bounds=crop_bounds,
            crop_idx=crop_idx,
            num_crops_in_window=num_crops,
        )

        input_dict, target_dict = self.dataset.task.process_inputs(
            cur_raw_inputs,
            metadata=cur_metadata,
            load_targets=not self.dataset.split_config.get_skip_targets(),
        )
        input_dict.update(cur_passthrough_inputs)
        input_dict, target_dict = self.dataset.transforms(input_dict, target_dict)

        return input_dict, target_dict, cur_metadata

    def get_dataset_examples(self) -> list[Window]:
        """Returns a list of windows in this dataset."""
        return self.dataset.get_dataset_examples()

    def set_name(self, name: str) -> None:
        """Sets dataset name.

        Args:
            name: dataset name
        """
        self.dataset.set_name(name)


class InMemoryAllCropsDataset(InMemoryDataset):
    """In-memory dataset that enumerates all sliding-window crops.

    This should be used when SplitConfig.load_all_crops is enabled.
    Precomputes the full list of crop positions so that __len__ and
    random-access __getitem__ are available (unlike IterableAllCropsDataset).
    """

    def __init__(
        self,
        dataset: ModelDataset,
        crop_size: tuple[int, int],
        overlap_pixels: int = 0,
    ):
        """Create a new InMemoryAllCropsDataset.

        Args:
            dataset: the ModelDataset to wrap.
            crop_size: the size of the crops to extract.
            overlap_pixels: the number of pixels shared between adjacent crops.
        """
        super().__init__(dataset=dataset, crop_size=crop_size)

        self.crops: list[tuple[int, tuple[int, int, int, int], tuple[int, int]]] = []
        for window_id, window in enumerate(self.windows):
            window_crop_bounds = get_window_crop_options(
                crop_size, (overlap_pixels, overlap_pixels), window.bounds
            )
            for i, crop_bound in enumerate(window_crop_bounds):
                self.crops.append((window_id, crop_bound, (i, len(window_crop_bounds))))

    def __len__(self) -> int:
        """Return the total number of crops in the dataset."""
        return len(self.crops)

    def __getitem__(
        self, index: int
    ) -> tuple[dict[str, Any], dict[str, Any], SampleMetadata]:
        """Return (input_dict, target_dict, metadata) for a single flattened crop."""
        window_id, crop_bounds, (crop_idx, num_crops) = self.crops[index]
        return self._finalize_sample(window_id, crop_bounds, crop_idx, num_crops)


class InMemoryRandomCropDataset(InMemoryDataset):
    """In-memory dataset that picks one random crop per window per access.

    Each __getitem__ call selects a fresh random crop from the cached window,
    so different epochs see different crops while avoiding repeated disk reads.
    """

    def __init__(
        self,
        dataset: ModelDataset,
        crop_size: tuple[int, int] | None = None,
        fix_crop_pick: bool = False,
    ):
        """Create a new InMemoryRandomCropDataset.

        Args:
            dataset: the ModelDataset to wrap.
            crop_size: the size of the crops to extract, or None to return
                full windows without cropping.
            fix_crop_pick: if True, the random crop for a given window index
                is deterministic (seeded by the index). Useful for val/test.
        """
        super().__init__(dataset=dataset, crop_size=crop_size)
        self.fix_crop_pick = fix_crop_pick

    def __len__(self) -> int:
        """Return the number of windows (one sample per window)."""
        return len(self.windows)

    def __getitem__(
        self, index: int
    ) -> tuple[dict[str, Any], dict[str, Any], SampleMetadata]:
        """Return (input_dict, target_dict, metadata) for a random crop of the window."""
        window = self.windows[index]

        if self.crop_size is None:
            crop_bounds = window.bounds
        else:
            rng = random.Random(index if self.fix_crop_pick else None)
            window_size = (
                window.bounds[2] - window.bounds[0],
                window.bounds[3] - window.bounds[1],
            )

            def get_crop_range(n_crop: int, n_window: int) -> tuple[int, int]:
                if n_crop > n_window:
                    start = rng.randint(n_window - n_crop, 0)
                else:
                    start = rng.randint(0, n_window - n_crop)
                return start, start + n_crop

            col_start, col_end = get_crop_range(self.crop_size[0], window_size[0])
            row_start, row_end = get_crop_range(self.crop_size[1], window_size[1])

            crop_bounds = (
                window.bounds[0] + col_start,
                window.bounds[1] + row_start,
                window.bounds[0] + col_end,
                window.bounds[1] + row_end,
            )

        return self._finalize_sample(index, crop_bounds, crop_idx=0, num_crops=1)
