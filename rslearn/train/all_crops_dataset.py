"""Wrapper around ModelDataset to load all crops in a window."""

import itertools
from collections.abc import Iterable, Iterator
from dataclasses import replace
from typing import Any

import shapely
import torch

from rslearn.dataset import Window
from rslearn.train.dataset import DataInput, ModelDataset
from rslearn.train.model_context import RasterImage, SampleMetadata
from rslearn.utils.geometry import PixelBounds, STGeometry


def get_window_crop_options(
    crop_size: tuple[int, int],
    overlap_size: tuple[int, int],
    bounds: PixelBounds,
) -> list[PixelBounds]:
    """Get the bounds of each input crop within the window bounds.

    This is used when running inference on all crops of a large window, to
    compute the position of each crop.

    Args:
        crop_size: the size of the crops to extract.
        overlap_size: the size of the overlap between crops.
        bounds: the window bounds to divide up into smaller crops.

    Returns:
        a list of crop bounds within the overall bounds. The rightmost and
            bottommost crops may extend beyond the provided bounds.
    """
    # We stride the crops by (crop_size - overlap_size) until the last crop.
    # The first crop always starts at bounds[0]/bounds[1]. It's okay if it extends
    # beyond the window bounds since pad_slice_protect pads raster inputs.
    # We handle the last crop with a special case to ensure it does not exceed the
    # window bounds. Instead, it may overlap the previous crop.
    # Here is a simple 1D example:
    # - Suppose bounds is [0, 15] with crop_size=8, overlap_size=2
    # - Then the first crop should be [0, 8] (from first crop special case)
    # - There will only be one crop in the middle, [6, 14]
    # - And the last crop will be at [7, 15]
    # - Note that, if the bounds was [0, 14], we will only have the first/last crop
    #   special cases with no crops in the middle from the range(...).
    cols = [bounds[0]] + list(
        range(
            bounds[0] + crop_size[0] - overlap_size[0],
            bounds[2] - crop_size[0],
            crop_size[0] - overlap_size[0],
        )
    )
    rows = [bounds[1]] + list(
        range(
            bounds[1] + crop_size[1] - overlap_size[1],
            bounds[3] - crop_size[1],
            crop_size[1] - overlap_size[1],
        )
    )
    # Add last crops only if the input is larger than one crop.
    if bounds[2] - crop_size[0] > bounds[0]:
        cols.append(bounds[2] - crop_size[0])
    if bounds[3] - crop_size[1] > bounds[1]:
        rows.append(bounds[3] - crop_size[1])

    crop_bounds: list[PixelBounds] = []
    for col in cols:
        for row in rows:
            crop_bounds.append((col, row, col + crop_size[0], row + crop_size[1]))
    return crop_bounds


def pad_slice_protect(
    raw_inputs: dict[str, Any],
    passthrough_inputs: dict[str, Any],
    crop_size: tuple[int, int],
    inputs: dict[str, DataInput],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pad raster inputs in-place by crop size to protect slicing near right/bottom edges.

    The padding is scaled based on each input's resolution_factor.

    Args:
        raw_inputs: the raw inputs to pad.
        passthrough_inputs: the passthrough inputs to pad.
        crop_size: the size of the crops to extract (at window resolution).
        inputs: the DataInput definitions, used to get resolution_factor per input.

    Returns:
        a tuple of (raw_inputs, passthrough_inputs).
    """
    for input_name in set(raw_inputs.keys()) | set(passthrough_inputs.keys()):
        data_input = inputs.get(input_name)
        if data_input is None or data_input.data_type != "raster":
            continue

        value = raw_inputs.get(input_name, passthrough_inputs.get(input_name))
        if not isinstance(value, RasterImage):
            raise TypeError(
                f"expected raster input '{input_name}' to be a RasterImage, got {type(value)}"
            )

        rf = data_input.resolution_factor
        scale = rf.numerator / rf.denominator
        scaled_pad_x = int(crop_size[0] * scale)
        scaled_pad_y = int(crop_size[1] * scale)

        padded = torch.nn.functional.pad(
            value.image, pad=(0, scaled_pad_x, 0, scaled_pad_y)
        )
        padded_value = RasterImage(padded, value.timestamps)
        if input_name in raw_inputs:
            raw_inputs[input_name] = padded_value
        if input_name in passthrough_inputs:
            passthrough_inputs[input_name] = padded_value
    return raw_inputs, passthrough_inputs


def crop_tensor_or_rasterimage(
    x: torch.Tensor | RasterImage, start: tuple[int, int], end: tuple[int, int]
) -> torch.Tensor | RasterImage:
    """Crop a tensor or a RasterImage."""
    if isinstance(x, torch.Tensor):
        # Crop the CHW tensor with scaled coordinates.
        return x[
            :,
            start[1] : end[1],
            start[0] : end[0],
        ].clone()
    else:
        # Crop the CTHW tensor with scaled coordinates.
        return RasterImage(
            x.image[
                :,
                :,
                start[1] : end[1],
                start[0] : end[0],
            ].clone(),
            x.timestamps,
        )


class IterableAllCropsDataset(torch.utils.data.IterableDataset):
    """This wraps a ModelDataset to iterate over all crops in that dataset.

    This should be used when SplitConfig.load_all_crops is enabled. The ModelDataset
    is configured with no crop size (load entire windows), and the dataset is wrapped
    in an AllCropsDataset.

    Similar to DistributedSampler, we add extra samples at each rank to ensure
    consistent number of batches across all ranks.
    """

    def __init__(
        self,
        dataset: ModelDataset,
        crop_size: tuple[int, int],
        overlap_pixels: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ):
        """Create a new IterableAllCropsDataset.

        Args:
            dataset: the ModelDataset to wrap.
            crop_size: the size of the crops to extract.
            overlap_pixels: the number of pixels shared between adjacent crops. Note
                that the right/bottom-most crops may still overlap with other crops even
                if overlap_pixels=0 since we ensure that all crops are contained in the
                window bounds.
            rank: the global rank of this train worker process.
            world_size: the total number of train worker processes.
        """
        super().__init__()
        self.dataset = dataset
        self.crop_size = crop_size
        self.overlap_size = (overlap_pixels, overlap_pixels)
        self.rank = rank
        self.world_size = world_size
        self.windows = self.dataset.get_dataset_examples()
        self.inputs = dataset.inputs

    def set_name(self, name: str) -> None:
        """Sets dataset name.

        Args:
            name: dataset name
        """
        self.dataset.set_name(name)

    def get_window_num_crops(self, bounds: PixelBounds) -> int:
        """Get the number of crops for these bounds.

        This corresponds to the length of the list returned by get_window_crop_options.
        """
        num_cols = (
            len(
                range(
                    bounds[0],
                    bounds[2] - self.crop_size[0],
                    self.crop_size[0] - self.overlap_size[0],
                )
            )
            + 1
        )
        num_rows = (
            len(
                range(
                    bounds[1],
                    bounds[3] - self.crop_size[1],
                    self.crop_size[1] - self.overlap_size[1],
                )
            )
            + 1
        )
        return num_cols * num_rows

    def _get_worker_iteration_data(self) -> tuple[Iterable[int], int]:
        """Get the windows we should iterate over.

        This is split both by training worker (self.rank) and data loader worker (via
        get_worker_info).

        We also compute the total number of samples that each data loader worker should
        yield. This is important for DDP to ensure that all ranks see the same number
        of batches.

        Returns:
            a tuple (window_ids, num_samples_per_worker).
        """
        # Figure out the total number of data loader workers and our worker ID.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        global_worker_id = self.rank * num_workers + worker_id
        global_num_workers = self.world_size * num_workers

        # Split up the windows evenly among the workers.
        # We compute this for all workers since we will need to see the maximum number
        # of samples under this assignment across workers.
        window_indexes = range(len(self.windows))
        windows_by_worker = [
            window_indexes[cur_rank :: self.world_size][cur_worker_id::num_workers]
            for cur_rank in range(self.world_size)
            for cur_worker_id in range(num_workers)
        ]

        # Now compute the maximum number of samples across workers.
        max_num_crops = 0
        for worker_windows in windows_by_worker:
            worker_num_crops = 0
            for window_id in worker_windows:
                worker_num_crops += self.get_window_num_crops(
                    self.windows[window_id].bounds
                )
            max_num_crops = max(max_num_crops, worker_num_crops)

        # Each worker needs at least one window, otherwise it won't be able to pad.
        # Unless there are zero windows total, which is fine.
        # Previously we would address this by borrowing the windows from another
        # worker, but this causes issues with RslearnWriter: if we yield the same
        # window from parallel workers, it may end up writing an empty output for that
        # window in the end.
        # So now we raise an error instead, and require the number of workers to be
        # less than the number of windows.
        if len(windows_by_worker[global_worker_id]) == 0 and max_num_crops > 0:
            raise ValueError(
                f"the number of workers {global_num_workers} must be <= the number of windows {len(self.windows)}"
            )

        return (windows_by_worker[global_worker_id], max_num_crops)

    def __iter__(
        self,
    ) -> Iterator[tuple[dict[str, Any], dict[str, Any], SampleMetadata]]:
        """Iterate over all crops in each element of the underlying ModelDataset."""
        # Iterate over the window IDs until we have returned enough samples.
        window_ids, num_samples_needed = self._get_worker_iteration_data()
        num_samples_returned = 0

        for iteration_idx in itertools.count():
            for window_id in window_ids:
                raw_inputs, passthrough_inputs, metadata = self.dataset.get_raw_inputs(
                    window_id
                )
                bounds = metadata.crop_bounds

                # For simplicity, pad raster inputs by crop size to ensure that any crop
                # bounds extending outside the window bounds will not have issues when
                # we slice later. Padding is scaled per-input based on resolution_factor.
                pad_slice_protect(
                    raw_inputs, passthrough_inputs, self.crop_size, self.inputs
                )

                # Now iterate over the crops and extract/yield them.
                # Note that, in case user is leveraging RslearnWriter, it is important that
                # the crop_idx be increasing (as we iterate) within one window.
                crops = get_window_crop_options(
                    self.crop_size, self.overlap_size, bounds
                )
                for crop_idx, crop_bounds in enumerate(crops):
                    cur_geom = STGeometry(
                        metadata.projection, shapely.box(*crop_bounds), None
                    )
                    start_offset = (
                        crop_bounds[0] - bounds[0],
                        crop_bounds[1] - bounds[1],
                    )
                    end_offset = (
                        crop_bounds[2] - bounds[0],
                        crop_bounds[3] - bounds[1],
                    )

                    # Define a helper function to handle each input dict.
                    # Crop coordinates are scaled based on each input's resolution_factor.
                    def crop_input_dict(d: dict[str, Any]) -> dict[str, Any]:
                        cropped = {}
                        for input_name, value in d.items():
                            if isinstance(value, torch.Tensor | RasterImage):
                                # Scale crop coordinates via multiply_bounds
                                # to stay consistent with the training path.
                                rf = self.inputs[input_name].resolution_factor
                                relative_bounds = (
                                    start_offset[0],
                                    start_offset[1],
                                    end_offset[0],
                                    end_offset[1],
                                )
                                scaled_bounds = rf.multiply_bounds(relative_bounds)
                                scaled_start = (
                                    scaled_bounds[0],
                                    scaled_bounds[1],
                                )
                                scaled_end = (
                                    scaled_bounds[2],
                                    scaled_bounds[3],
                                )
                                cropped[input_name] = crop_tensor_or_rasterimage(
                                    value, scaled_start, scaled_end
                                )
                            elif isinstance(value, list):
                                cropped[input_name] = [
                                    feat
                                    for feat in value
                                    if cur_geom.intersects(feat.geometry)
                                ]
                            else:
                                raise ValueError(
                                    "got input that is neither tensor nor feature list"
                                )
                        return cropped

                    cur_raw_inputs = crop_input_dict(raw_inputs)
                    cur_passthrough_inputs = crop_input_dict(passthrough_inputs)

                    # Adjust the metadata as well.
                    cur_metadata = replace(
                        metadata,
                        crop_bounds=crop_bounds,
                        crop_idx=crop_idx,
                        num_crops_in_window=len(crops),
                    )

                    # Now we can compute input and target dicts via the task.
                    input_dict, target_dict = self.dataset.task.process_inputs(
                        cur_raw_inputs,
                        metadata=cur_metadata,
                        load_targets=not self.dataset.split_config.get_skip_targets(),
                    )
                    input_dict.update(cur_passthrough_inputs)
                    input_dict, target_dict = self.dataset.transforms(
                        input_dict, target_dict
                    )

                    if num_samples_returned < num_samples_needed:
                        yield input_dict, target_dict, cur_metadata
                        num_samples_returned += 1
                    else:
                        assert iteration_idx > 0

            if num_samples_returned >= num_samples_needed:
                break

    def get_dataset_examples(self) -> list[Window]:
        """Returns a list of windows in this dataset."""
        return self.dataset.get_dataset_examples()


class InMemoryAllCropsDataset(torch.utils.data.Dataset):
    """This wraps a ModelDataset to iterate over all crops in that dataset.

    This should be used when SplitConfig.load_all_crops is enabled.

    This is a simpler version of IterableAllCropsDataset that caches all windows in memory.
    This is useful for small datasets that fit in memory.
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
            overlap_pixels: the number of pixels shared between adjacent crops. Note
                that the right/bottom-most crops may still overlap since we ensure that
                all crops are contained in the window bounds.
        """
        super().__init__()
        self.dataset = dataset
        self.crop_size = crop_size
        self.overlap_size = (overlap_pixels, overlap_pixels)
        self.windows = self.dataset.get_dataset_examples()
        self.inputs = dataset.inputs
        self.window_cache: dict[
            int, tuple[dict[str, Any], dict[str, Any], SampleMetadata]
        ] = {}

        # Precompute the batch boundaries for each window
        self.crops = []
        for window_id, window in enumerate(self.windows):
            window_crop_bounds = get_window_crop_options(
                self.crop_size, self.overlap_size, window.bounds
            )
            for i, crop_bound in enumerate(window_crop_bounds):
                self.crops.append((window_id, crop_bound, (i, len(window_crop_bounds))))

    def get_raw_inputs(
        self, index: int
    ) -> tuple[dict[str, Any], dict[str, Any], SampleMetadata]:
        """Get the raw inputs for a single crop. Retrieve from cache if possible.

        Also pads raster inputs by crop size to protect slicing near right/bottom edges.

        Args:
            index: the index of the crop.

        Returns:
            a tuple of (raw_inputs, passthrough_inputs, metadata).
        """
        if index in self.window_cache:
            return self.window_cache[index]

        raw_inputs, passthrough_inputs, metadata = self.dataset.get_raw_inputs(index)
        pad_slice_protect(raw_inputs, passthrough_inputs, self.crop_size, self.inputs)

        self.window_cache[index] = (raw_inputs, passthrough_inputs, metadata)
        return self.window_cache[index]

    def _crop_input_dict(
        self,
        d: dict[str, Any],
        start_offset: tuple[int, int],
        end_offset: tuple[int, int],
        cur_geom: STGeometry,
    ) -> dict[str, Any]:
        """Crop a dictionary of inputs to the given bounds.

        Crop coordinates are scaled via multiply_bounds to stay consistent
        with the training path.
        """
        cropped = {}
        for input_name, value in d.items():
            if isinstance(value, torch.Tensor | RasterImage):
                rf = self.inputs[input_name].resolution_factor
                relative_bounds = (
                    start_offset[0],
                    start_offset[1],
                    end_offset[0],
                    end_offset[1],
                )
                scaled_bounds = rf.multiply_bounds(relative_bounds)
                scaled_start = (
                    scaled_bounds[0],
                    scaled_bounds[1],
                )
                scaled_end = (
                    scaled_bounds[2],
                    scaled_bounds[3],
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

    def __len__(self) -> int:
        """Return the total number of crops in the dataset."""
        return len(self.crops)

    def __getitem__(
        self, index: int
    ) -> tuple[dict[str, Any], dict[str, Any], SampleMetadata]:
        """Return (input_dict, target_dict, metadata) for a single flattened crop."""
        (window_id, crop_bounds, (crop_idx, num_crops)) = self.crops[index]
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

        # Adjust the metadata as well.
        cur_metadata = replace(
            metadata,
            crop_bounds=crop_bounds,
            crop_idx=crop_idx,
            num_crops_in_window=num_crops,
        )

        # Now we can compute input and target dicts via the task.
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
