"""Wrapper around ModelDataset to load all patches (crops) in a window."""

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


def get_window_patch_options(
    patch_size: tuple[int, int],
    overlap_size: tuple[int, int],
    bounds: PixelBounds,
) -> list[PixelBounds]:
    """Get the bounds of each input patch within the window bounds.

    This is used when running inference on all patches (crops) of a large window, to
    compute the position of each patch.

    Args:
        patch_size: the size of the patches to extract.
        overlap_size: the size of the overlap between patches.
        bounds: the window bounds to divide up into smaller patches.

    Returns:
        a list of patch bounds within the overall bounds. The rightmost and
            bottommost patches may extend beyond the provided bounds.
    """
    # We stride the patches by patch_size - overlap_size until the last patch.
    # We handle the first patch with a special case to ensure it is always used.
    # We handle the last patch with a special case to ensure it does not exceed the
    # window bounds. Instead, it may overlap the previous patch.
    cols = [bounds[0]] + list(
        range(
            bounds[0] + patch_size[0],
            bounds[2] - patch_size[0],
            patch_size[0] - overlap_size[0],
        )
    )
    rows = [bounds[1]] + list(
        range(
            bounds[1] + patch_size[1],
            bounds[3] - patch_size[1],
            patch_size[1] - overlap_size[1],
        )
    )
    # Add last patches only if the input is larger than one patch.
    if bounds[2] - patch_size[0] > bounds[0]:
        cols.append(bounds[2] - patch_size[0])
    if bounds[3] - patch_size[1] > bounds[1]:
        rows.append(bounds[3] - patch_size[1])

    patch_bounds: list[PixelBounds] = []
    for col in cols:
        for row in rows:
            patch_bounds.append((col, row, col + patch_size[0], row + patch_size[1]))
    return patch_bounds


def pad_slice_protect(
    raw_inputs: dict[str, Any],
    passthrough_inputs: dict[str, Any],
    patch_size: tuple[int, int],
    inputs: dict[str, DataInput],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pad tensors in-place by patch size to protect slicing near right/bottom edges.

    The padding is scaled based on each input's resolution_factor.

    Args:
        raw_inputs: the raw inputs to pad.
        passthrough_inputs: the passthrough inputs to pad.
        patch_size: the size of the patches to extract (at window resolution).
        inputs: the DataInput definitions, used to get resolution_factor per input.

    Returns:
        a tuple of (raw_inputs, passthrough_inputs).
    """
    for d in [raw_inputs, passthrough_inputs]:
        for input_name, value in list(d.items()):
            if not isinstance(value, torch.Tensor):
                continue
            # Get resolution scale for this input
            rf = inputs[input_name].resolution_factor
            scale = rf.numerator / rf.denominator
            # Scale the padding amount
            scaled_pad_x = int(patch_size[0] * scale)
            scaled_pad_y = int(patch_size[1] * scale)
            d[input_name] = torch.nn.functional.pad(
                value, pad=(0, scaled_pad_x, 0, scaled_pad_y)
            )
    return raw_inputs, passthrough_inputs


class IterableAllPatchesDataset(torch.utils.data.IterableDataset):
    """This wraps a ModelDataset to iterate over all patches in that dataset.

    This should be used when SplitConfig.load_all_patches is enabled. The ModelDataset
    is configured with no patch size (load entire windows), and the dataset is wrapped
    in an AllPatchesDataset.

    Similar to DistributedSampler, we add extra samples at each rank to ensure
    consistent number of batches across all ranks.
    """

    def __init__(
        self,
        dataset: ModelDataset,
        patch_size: tuple[int, int],
        overlap_ratio: float = 0.0,
        rank: int = 0,
        world_size: int = 1,
    ):
        """Create a new IterableAllPatchesDataset.

        Args:
            dataset: the ModelDataset to wrap.
            patch_size: the size of the patches to extract.
            overlap_ratio: whether to include overlap between the patches. Note that
                the right/bottom-most patches may still overlap since we ensure that
                all patches are contained in the window bounds.
            rank: the global rank of this train worker process.
            world_size: the total number of train worker processes.
        """
        super().__init__()
        self.dataset = dataset
        self.patch_size = patch_size
        self.overlap_size = (
            round(self.patch_size[0] * overlap_ratio),
            round(self.patch_size[1] * overlap_ratio),
        )
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

    def get_window_num_patches(self, bounds: PixelBounds) -> int:
        """Get the number of patches for these bounds.

        This corresponds to the length of the list returned by get_patch_options.
        """
        num_cols = (
            len(
                range(
                    bounds[0],
                    bounds[2] - self.patch_size[0],
                    self.patch_size[0] - self.overlap_size[0],
                )
            )
            + 1
        )
        num_rows = (
            len(
                range(
                    bounds[1],
                    bounds[3] - self.patch_size[1],
                    self.patch_size[1] - self.overlap_size[1],
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
        max_num_patches = 0
        for worker_windows in windows_by_worker:
            worker_num_patches = 0
            for window_id in worker_windows:
                worker_num_patches += self.get_window_num_patches(
                    self.windows[window_id].bounds
                )
            max_num_patches = max(max_num_patches, worker_num_patches)

        # Each worker needs at least one window, otherwise it won't be able to pad.
        # Unless there are zero windows total, which is fine.
        # Previously we would address this by borrowing the windows from another
        # worker, but this causes issues with RslearnWriter: if we yield the same
        # window from parallel workers, it may end up writing an empty output for that
        # window in the end.
        # So now we raise an error instead, and require the number of workers to be
        # less than the number of windows.
        if len(windows_by_worker[global_worker_id]) == 0 and max_num_patches > 0:
            raise ValueError(
                f"the number of workers {global_num_workers} must be <= the number of windows {len(self.windows)}"
            )

        return (windows_by_worker[global_worker_id], max_num_patches)

    def __iter__(
        self,
    ) -> Iterator[tuple[dict[str, Any], dict[str, Any], SampleMetadata]]:
        """Iterate over all patches in each element of the underlying ModelDataset."""
        # Iterate over the window IDs until we have returned enough samples.
        window_ids, num_samples_needed = self._get_worker_iteration_data()
        num_samples_returned = 0

        for iteration_idx in itertools.count():
            for window_id in window_ids:
                raw_inputs, passthrough_inputs, metadata = self.dataset.get_raw_inputs(
                    window_id
                )
                bounds = metadata.patch_bounds

                # For simplicity, pad tensors by patch size to ensure that any patch bounds
                # extending outside the window bounds will not have issues when we slice
                # the tensors later. Padding is scaled per-input based on resolution_factor.
                pad_slice_protect(
                    raw_inputs, passthrough_inputs, self.patch_size, self.inputs
                )

                # Now iterate over the patches and extract/yield the crops.
                # Note that, in case user is leveraging RslearnWriter, it is important that
                # the patch_idx be increasing (as we iterate) within one window.
                patches = get_window_patch_options(
                    self.patch_size, self.overlap_size, bounds
                )
                for patch_idx, patch_bounds in enumerate(patches):
                    cur_geom = STGeometry(
                        metadata.projection, shapely.box(*patch_bounds), None
                    )
                    start_offset = (
                        patch_bounds[0] - bounds[0],
                        patch_bounds[1] - bounds[1],
                    )
                    end_offset = (
                        patch_bounds[2] - bounds[0],
                        patch_bounds[3] - bounds[1],
                    )

                    # Define a helper function to handle each input dict.
                    # Crop coordinates are scaled based on each input's resolution_factor.
                    def crop_input_dict(d: dict[str, Any]) -> dict[str, Any]:
                        cropped = {}
                        for input_name, value in d.items():
                            if isinstance(value, torch.Tensor | RasterImage):
                                # Get resolution scale for this input
                                rf = self.inputs[input_name].resolution_factor
                                scale = rf.numerator / rf.denominator
                                # Scale the crop coordinates
                                scaled_start = (
                                    int(start_offset[0] * scale),
                                    int(start_offset[1] * scale),
                                )
                                scaled_end = (
                                    int(end_offset[0] * scale),
                                    int(end_offset[1] * scale),
                                )
                                if isinstance(value, torch.Tensor):
                                    # Crop the CHW tensor with scaled coordinates.
                                    cropped[input_name] = value[
                                        :,
                                        scaled_start[1] : scaled_end[1],
                                        scaled_start[0] : scaled_end[0],
                                    ].clone()
                                else:
                                    # Crop the CTHW tensor with scaled coordinates.
                                    cropped[input_name] = RasterImage(
                                        value.image[
                                            :,
                                            :,
                                            scaled_start[1] : scaled_end[1],
                                            scaled_start[0] : scaled_end[0],
                                        ].clone(),
                                        value.timestamps,
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
                        patch_bounds=patch_bounds,
                        patch_idx=patch_idx,
                        num_patches_in_window=len(patches),
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


class InMemoryAllPatchesDataset(torch.utils.data.Dataset):
    """This wraps a ModelDataset to iterate over all patches in that dataset.

    This should be used when SplitConfig.load_all_patches is enabled.

    This is a simpler version of IterableAllPatchesDataset that caches all windows in memory.
    This is useful for small datasets that fit in memory.
    """

    def __init__(
        self,
        dataset: ModelDataset,
        patch_size: tuple[int, int],
        overlap_ratio: float = 0.0,
    ):
        """Create a new InMemoryAllPatchesDataset.

        Args:
            dataset: the ModelDataset to wrap.
            patch_size: the size of the patches to extract.
            overlap_ratio: whether to include overlap between the patches. Note that
                the right/bottom-most patches may still overlap since we ensure that
                all patches are contained in the window bounds.
        """
        super().__init__()
        self.dataset = dataset
        self.patch_size = patch_size
        self.overlap_size = (
            round(self.patch_size[0] * overlap_ratio),
            round(self.patch_size[1] * overlap_ratio),
        )
        self.windows = self.dataset.get_dataset_examples()
        self.inputs = dataset.inputs
        self.window_cache: dict[
            int, tuple[dict[str, Any], dict[str, Any], SampleMetadata]
        ] = {}

        # Precompute the batch boundaries for each window
        self.patches = []
        for window_id, window in enumerate(self.windows):
            patch_bounds = get_window_patch_options(
                self.patch_size, self.overlap_size, window.bounds
            )
            for i, patch_bound in enumerate(patch_bounds):
                self.patches.append((window_id, patch_bound, (i, len(patch_bounds))))

    def get_raw_inputs(
        self, index: int
    ) -> tuple[dict[str, Any], dict[str, Any], SampleMetadata]:
        """Get the raw inputs for a single patch. Retrieve from cache if possible.

        Also crops/pads the tensors by patch size to protect slicing near right/bottom edges.

        Args:
            index: the index of the patch.

        Returns:
            a tuple of (raw_inputs, passthrough_inputs, metadata).
        """
        if index in self.window_cache:
            return self.window_cache[index]

        raw_inputs, passthrough_inputs, metadata = self.dataset.get_raw_inputs(index)
        pad_slice_protect(raw_inputs, passthrough_inputs, self.patch_size, self.inputs)

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

        Crop coordinates are scaled based on each input's resolution_factor.
        """
        cropped = {}
        for input_name, value in d.items():
            if isinstance(value, torch.Tensor | RasterImage):
                # Get resolution scale for this input
                rf = self.inputs[input_name].resolution_factor
                scale = rf.numerator / rf.denominator
                # Scale the crop coordinates
                scaled_start = (
                    int(start_offset[0] * scale),
                    int(start_offset[1] * scale),
                )
                scaled_end = (
                    int(end_offset[0] * scale),
                    int(end_offset[1] * scale),
                )
                if isinstance(value, torch.Tensor):
                    cropped[input_name] = value[
                        :,
                        scaled_start[1] : scaled_end[1],
                        scaled_start[0] : scaled_end[0],
                    ].clone()
                else:
                    # Crop the CTHW tensor with scaled coordinates.
                    cropped[input_name] = RasterImage(
                        value.image[
                            :,
                            :,
                            scaled_start[1] : scaled_end[1],
                            scaled_start[0] : scaled_end[0],
                        ].clone(),
                        value.timestamps,
                    )
            elif isinstance(value, list):
                cropped[input_name] = [
                    feat for feat in value if cur_geom.intersects(feat.geometry)
                ]
            else:
                raise ValueError("got input that is neither tensor nor feature list")
        return cropped

    def __len__(self) -> int:
        """Return the total number of patches in the dataset."""
        return len(self.patches)

    def __getitem__(
        self, index: int
    ) -> tuple[dict[str, Any], dict[str, Any], SampleMetadata]:
        """Return (input_dict, target_dict, metadata) for a single flattened patch."""
        (window_id, patch_bounds, (patch_idx, num_patches)) = self.patches[index]
        raw_inputs, passthrough_inputs, metadata = self.get_raw_inputs(window_id)
        bounds = metadata.patch_bounds

        cur_geom = STGeometry(metadata.projection, shapely.box(*patch_bounds), None)
        start_offset = (patch_bounds[0] - bounds[0], patch_bounds[1] - bounds[1])
        end_offset = (patch_bounds[2] - bounds[0], patch_bounds[3] - bounds[1])

        cur_raw_inputs = self._crop_input_dict(
            raw_inputs, start_offset, end_offset, cur_geom
        )
        cur_passthrough_inputs = self._crop_input_dict(
            passthrough_inputs, start_offset, end_offset, cur_geom
        )

        # Adjust the metadata as well.
        cur_metadata = replace(
            metadata,
            patch_bounds=patch_bounds,
            patch_idx=patch_idx,
            num_patches_in_window=num_patches,
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
