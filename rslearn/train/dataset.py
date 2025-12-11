"""Default Dataset for rslearn."""

import hashlib
import json
import multiprocessing
import os
import random
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Any

import torch
import tqdm
from rasterio.warp import Resampling

import rslearn.train.transforms.transform
from rslearn.config import (
    DType,
    LayerConfig,
)
from rslearn.dataset.dataset import Dataset
from rslearn.dataset.storage.file import FileWindowStorage
from rslearn.dataset.window import Window, get_layer_and_group_from_dir_name
from rslearn.log_utils import get_logger
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import PixelBounds, ResolutionFactor
from rslearn.utils.mp import star_imap_unordered

from .model_context import SampleMetadata
from .tasks import Task
from .transforms import Sequential

logger = get_logger(__name__)


def get_torch_dtype(dtype: DType) -> torch.dtype:
    """Convert rslearn DType to torch dtype."""
    if dtype == DType.INT32:
        return torch.int32
    elif dtype == DType.FLOAT32:
        return torch.float32
    else:
        raise ValueError(f"unable to handle {dtype} as a torch dtype")


class SamplerFactory:
    """Factory to produce a Sampler.

    This enables configuring a sampler without needing to pass the dataset.
    """

    def get_sampler(self, dataset: "ModelDataset") -> torch.utils.data.Sampler:
        """Create a sampler for the given dataset.

        Args:
            dataset: the dataset

        Returns:
            a sampler
        """
        raise NotImplementedError


class RandomSamplerFactory(SamplerFactory):
    """A sampler factory for RandomSampler."""

    def __init__(
        self, replacement: bool = False, num_samples: int | None = None
    ) -> None:
        """Initialize a RandomSamplerFactory.

        Args:
            replacement: whether to pick with replacement, default false
            num_samples: optional number of dataset samples to limit iteration to,
                otherwise picks random samples equal to the dataset size
        """
        self.replacement = replacement
        self.num_samples = num_samples

    def get_sampler(self, dataset: "ModelDataset") -> torch.utils.data.Sampler:
        """Create a sampler for the given dataset.

        Args:
            dataset: the dataset

        Returns:
            a RandomSampler
        """
        return torch.utils.data.RandomSampler(
            dataset, replacement=self.replacement, num_samples=self.num_samples
        )


class WeightedRandomSamplerFactory(SamplerFactory):
    """A sampler factory for WeightedRandomSampler."""

    def __init__(
        self, option_key: str, num_samples: int, replacement: bool = True
    ) -> None:
        """Initialize a WeightedRandomSamplerFactory.

        Args:
            option_key: the key in the window option dict containing the weights
            num_samples: number of examples to sample per epoch
            replacement: whether to pick with replacement, default true
        """
        self.option_key = option_key
        self.num_samples = num_samples
        self.replacement = replacement

    def get_sampler(self, dataset: "ModelDataset") -> torch.utils.data.Sampler:
        """Create a sampler for the given dataset.

        Args:
            dataset: the dataset

        Returns:
            a RandomSampler
        """
        weights = []
        for window in dataset.get_dataset_examples():
            weights.append(window.options[self.option_key])
        return torch.utils.data.WeightedRandomSampler(
            weights, self.num_samples, replacement=self.replacement
        )


@dataclass
class DataInput:
    """Specification of a piece of data from a window that is needed for training.

    The DataInput includes which layer(s) the data can be obtained from for each window.

    Args:
        data_type: either "raster" or "vector"
        layers: list of layer names that this input can be read from.
        bands: the bands to read, if this is a raster.
        required: whether examples lacking one of these layers should be skipped
        passthrough: whether to expose this to the model even if it isn't returned
            by any task
        is_target: whether this DataInput represents a target for the task. Targets
            are not read during prediction phase.
        dtype: data type to load the raster as
        load_all_layers: whether to load all of the layers specified in the list of
            layer names. By default, we randomly pick one layer to read. When
            reading multiple layers, the images are stacked on the channel
            dimension. This option will also cause the dataset to only include
            windows where all of the layers are materialized (by default, only
            windows with none of the layers materialized would be excluded).
        load_all_item_groups: whether to load all item groups in the layer(s) we
            are reading from. By default, we assume the specified layer name is of
            the form "{layer_name}.{group_idx}" and read that item group only. With
            this option enabled, we ignore the group_idx and read all item groups.
        resolution_factor: raster inputs are read by default at the window
            resolution. This is a multiplier to read at a different resolution,
            e.g. if the window resolution is 64x64 at 10 m/pixel, and
            resolution_factor=2, then the input is read as 32x32 at 20 m/pixel.
        resampling: resampling method (default nearest neighbor).
    """

    data_type: str
    layers: list[str]
    bands: list[str] | None = None
    required: bool = True
    passthrough: bool = False
    is_target: bool = False
    dtype: DType = DType.FLOAT32
    load_all_layers: bool = False
    load_all_item_groups: bool = False
    resolution_factor: ResolutionFactor = ResolutionFactor()
    resampling: Resampling = Resampling.nearest


def read_raster_layer_for_data_input(
    window: Window,
    bounds: PixelBounds,
    layer_name: str,
    group_idx: int,
    layer_config: LayerConfig,
    data_input: DataInput,
) -> torch.Tensor:
    """Read a raster layer for a DataInput.

    This scans the available rasters for the layer at the window to determine which
    ones are needed to get all of the configured bands.

    Args:
        window: the window to read from.
        bounds: the bounds to read.
        layer_name: the layer.
        group_idx: the item group.
        layer_config: the layer configuration.
        data_input: the DataInput that specifies the bands and dtype.

    Returns:
        tensor containing raster data.
    """
    # See what different sets of bands we need to read to get all the
    # configured bands.
    needed_bands = data_input.bands
    if needed_bands is None:
        raise ValueError(f"No bands specified for {layer_name}")
    needed_band_indexes = {}
    for i, band in enumerate(needed_bands):
        needed_band_indexes[band] = i
    needed_sets_and_indexes = []
    for band_set in layer_config.band_sets:
        needed_src_indexes = []
        needed_dst_indexes = []
        if band_set.bands is None:
            continue
        for i, band in enumerate(band_set.bands):
            if band not in needed_band_indexes:
                continue
            needed_src_indexes.append(i)
            needed_dst_indexes.append(needed_band_indexes[band])
            del needed_band_indexes[band]
        if len(needed_src_indexes) == 0:
            continue
        needed_sets_and_indexes.append(
            (band_set, needed_src_indexes, needed_dst_indexes)
        )
    if len(needed_band_indexes) > 0:
        raise ValueError(
            "could not get all the needed bands from "
            + f"window {window.name} layer {layer_name} group {group_idx}"
        )

    # Get the projection and bounds to read under (multiply window resolution # by
    # the specified resolution factor).
    final_projection = data_input.resolution_factor.multiply_projection(
        window.projection
    )
    final_bounds = data_input.resolution_factor.multiply_bounds(bounds)

    image = torch.zeros(
        (
            len(needed_bands),
            final_bounds[3] - final_bounds[1],
            final_bounds[2] - final_bounds[0],
        ),
        dtype=get_torch_dtype(data_input.dtype),
    )

    for band_set, src_indexes, dst_indexes in needed_sets_and_indexes:
        if band_set.format is None:
            raise ValueError(f"No format specified for {layer_name}")
        raster_format = band_set.instantiate_raster_format()
        raster_dir = window.get_raster_dir(
            layer_name, band_set.bands, group_idx=group_idx
        )

        # TODO: previously we try to read based on band_set.zoom_offset when possible,
        # and handle zooming in with torch.repeat (if resampling method is nearest
        # neighbor). However, we have not benchmarked whether this actually improves
        # data loading speed, so for simplicity, for now we let rasterio handle the
        # resampling. If it really is much faster to handle it via torch, then it may
        # make sense to bring back that functionality.

        src = raster_format.decode_raster(
            raster_dir, final_projection, final_bounds, resampling=Resampling.nearest
        )
        image[dst_indexes, :, :] = torch.as_tensor(
            src[src_indexes, :, :].astype(data_input.dtype.get_numpy_dtype())
        )

    return image


def read_data_input(
    dataset: Dataset,
    window: Window,
    bounds: PixelBounds,
    data_input: DataInput,
    rng: random.Random,
) -> torch.Tensor | list[Feature]:
    """Read the data specified by the DataInput from the window.

    Args:
        dataset: the dataset, to get layer configs.
        window: the window to read from.
        bounds: the bounds of the patch we are reading.
        data_input: the DataInput that specifies what layers to read.
        rng: random number generator

    Returns:
        the raster or vector data.
    """
    # We first enumerate which layers are available.
    # If load_all_item_groups is set, we need to check each item group within the
    # layer.
    layer_options: list[tuple[str, int]] = []
    if data_input.load_all_item_groups:
        wanted_layers = set(data_input.layers)
        for layer_name, group_idx in window.list_completed_layers():
            if layer_name not in wanted_layers:
                continue
            layer_options.append((layer_name, group_idx))
    else:
        for option in data_input.layers:
            layer_name, group_idx = get_layer_and_group_from_dir_name(option)
            if not window.is_layer_completed(layer_name, group_idx):
                continue
            layer_options.append((layer_name, group_idx))

    # Now determine the layers that we should actually read.
    # We randomly pick one, unless load_all_layers is set, in which case we read all of
    # them.
    layers_to_read: list[tuple[str, int]]
    if data_input.load_all_layers:
        # We assume that the user has ensured the layers are compatible, e.g. raster
        # layers will need to have the same number of bands.
        layers_to_read = layer_options
    else:
        layers_to_read = [rng.choice(layer_options)]

    if data_input.data_type == "raster":
        images: list[torch.Tensor] = []
        for layer_name, group_idx in layers_to_read:
            layer_config = dataset.layers[layer_name]
            images.append(
                read_raster_layer_for_data_input(
                    window, bounds, layer_name, group_idx, layer_config, data_input
                )
            )
        return torch.cat(images, dim=0)

    elif data_input.data_type == "vector":
        # We don't really support time series for vector data currently, we just
        # concatenate the features together.
        features: list[Feature] = []
        for layer_name, group_idx in layers_to_read:
            layer_config = dataset.layers[layer_name]
            vector_format = layer_config.instantiate_vector_format()
            layer_dir = window.get_layer_dir(layer_name, group_idx=group_idx)
            cur_features = vector_format.decode_vector(
                layer_dir, window.projection, window.bounds
            )
            features.extend(cur_features)

        return features

    else:
        raise ValueError(f"unknown data type {data_input.data_type}")


class SplitConfig:
    """Configuration that can be specified separately for train, val, and test."""

    def __init__(
        self,
        groups: list[str] | None = None,
        names: list[str] | None = None,
        tags: dict[str, Any] | None = None,
        num_samples: int | None = None,
        num_patches: int | None = None,
        transforms: list[torch.nn.Module] | None = None,
        sampler: SamplerFactory | None = None,
        patch_size: int | tuple[int, int] | None = None,
        overlap_ratio: float | None = None,
        load_all_patches: bool | None = None,
        skip_targets: bool | None = None,
    ) -> None:
        """Initialize a new SplitConfig.

        Args:
            groups: for this split, only read windows in one of these groups
            names: for this split, read windows with these specific names
            tags: only select windows that have options matching these tags. If key and
                value are set, then window must have an option with the same key and
                value. If value is empty, then only the existince of the key in the
                window options is checked.
            num_samples: limit this split to this many examples
            num_patches: limit this split to this many patches
            transforms: transforms to apply
            sampler: SamplerFactory for this split
            patch_size: an optional square size or (width, height) tuple. If set, read
                crops of this size rather than entire windows.
            overlap_ratio: an optional float between 0 and 1. If set, read patches with
                this ratio of overlap.
            load_all_patches: with patch_size set, rather than sampling a random patch
                for each window, read all patches as separate sequential items in the
                dataset.
            skip_targets: whether to skip targets when loading inputs
        """
        self.groups = groups
        self.names = names
        self.tags = tags
        self.num_samples = num_samples
        self.num_patches = num_patches
        self.transforms = transforms
        self.sampler = sampler
        self.patch_size = patch_size
        self.skip_targets = skip_targets

        # Note that load_all_patches are handled by the RslearnDataModule rather than
        # the ModelDataset.
        self.load_all_patches = load_all_patches
        self.overlap_ratio = overlap_ratio

        if self.overlap_ratio is not None and not (0 < self.overlap_ratio < 1):
            raise ValueError("overlap_ratio must be between 0 and 1 (exclusive)")

    def update(self, other: "SplitConfig") -> "SplitConfig":
        """Override settings in this SplitConfig with those in another.

        Returns:
            the resulting SplitConfig combining the settings.
        """
        result = SplitConfig(
            groups=self.groups,
            names=self.names,
            tags=self.tags,
            num_samples=self.num_samples,
            num_patches=self.num_patches,
            transforms=self.transforms,
            sampler=self.sampler,
            patch_size=self.patch_size,
            overlap_ratio=self.overlap_ratio,
            load_all_patches=self.load_all_patches,
            skip_targets=self.skip_targets,
        )
        if other.groups:
            result.groups = other.groups
        if other.names:
            result.names = other.names
        if other.tags:
            result.tags = other.tags
        if other.num_samples:
            result.num_samples = other.num_samples
        if other.num_patches:
            result.num_patches = other.num_patches
        if other.transforms:
            result.transforms = other.transforms
        if other.sampler:
            result.sampler = other.sampler
        if other.patch_size:
            result.patch_size = other.patch_size
        if other.overlap_ratio is not None:
            result.overlap_ratio = other.overlap_ratio
        if other.load_all_patches is not None:
            result.load_all_patches = other.load_all_patches
        if other.skip_targets is not None:
            result.skip_targets = other.skip_targets
        return result

    def get_patch_size(self) -> tuple[int, int] | None:
        """Get patch size normalized to int tuple."""
        if self.patch_size is None:
            return None
        if isinstance(self.patch_size, int):
            return (self.patch_size, self.patch_size)
        return self.patch_size

    def get_overlap_ratio(self) -> float:
        """Get the overlap ratio (default 0)."""
        return self.overlap_ratio if self.overlap_ratio is not None else 0.0

    def get_load_all_patches(self) -> bool:
        """Returns whether loading all patches is enabled (default False)."""
        return True if self.load_all_patches is True else False

    def get_skip_targets(self) -> bool:
        """Returns whether skip_targets is enabled (default False)."""
        return True if self.skip_targets is True else False


def check_window(inputs: dict[str, DataInput], window: Window) -> Window | None:
    """Verify that the window has the required layers based on the specified inputs.

    Args:
        inputs: the inputs to the dataset.
        window: the window to check.

    Returns:
        the window if it has all the required inputs or None otherwise
    """

    # Make sure window has all the needed layers.
    def is_available(data_input: DataInput) -> bool:
        # If load_all_layers is enabled, we should check that all the layers are
        # present. Otherwise, we just need one layer.
        is_any_layer_available = False
        are_all_layers_available = True
        for layer_name in data_input.layers:
            if window.is_layer_completed(layer_name):
                is_any_layer_available = True
            else:
                are_all_layers_available = False
        if data_input.load_all_layers:
            return are_all_layers_available
        else:
            return is_any_layer_available

    for data_input in inputs.values():
        if not data_input.required:
            continue
        if not is_available(data_input):
            logger.debug(
                "Skipping window %s since check for layers %s failed",
                window.name,
                data_input.layers,
            )
            return None

    return window


class ModelDataset(torch.utils.data.Dataset):
    """The default pytorch dataset implementation for rslearn."""

    def __init__(
        self,
        dataset: Dataset,
        split_config: SplitConfig,
        inputs: dict[str, DataInput],
        task: Task,
        workers: int,
        name: str | None = None,
        fix_patch_pick: bool = False,
    ) -> None:
        """Instantiate a new ModelDataset.

        Args:
            dataset: underlying rslearn dataset to read data from
            split_config: configuration specific to this split
            inputs: data to read from the dataset for training
            task: the task to train on
            workers: number of workers to use for initializing the dataset
            name: name of the dataset (default: None)
            fix_patch_pick: if True, fix the patch pick to be the same every time
                for a given window. Useful for testing (default: False)
        """
        self.dataset = dataset
        self.split_config = split_config
        self.inputs = inputs
        self.task = task
        self.name = name
        self.fix_patch_pick = fix_patch_pick
        if split_config.transforms:
            self.transforms = Sequential(*split_config.transforms)
        else:
            self.transforms = rslearn.train.transforms.transform.Identity()

        # Get normalized patch size from the SplitConfig.
        # But if load all patches is enabled, this is handled by AllPatchesDataset, so
        # here we instead load the entire windows.
        if split_config.get_load_all_patches():
            self.patch_size = None
        else:
            self.patch_size = split_config.get_patch_size()

        windows = self._get_initial_windows(split_config, workers)

        # If targets are not needed, remove them from the inputs.
        if split_config.get_skip_targets():
            for k in list(self.inputs.keys()):
                if self.inputs[k].is_target:
                    del self.inputs[k]

        # Eliminate windows that are missing either a requisite input layer, or missing
        # all target layers.
        new_windows = []
        if workers == 0:
            for window in windows:
                if check_window(self.inputs, window) is None:
                    continue
                new_windows.append(window)
        else:
            p = multiprocessing.Pool(workers)
            outputs = star_imap_unordered(
                p,
                check_window,
                [
                    dict(
                        inputs=self.inputs,
                        window=window,
                    )
                    for window in windows
                ],
            )
            for window in tqdm.tqdm(
                outputs, total=len(windows), desc="Checking available layers in windows"
            ):
                if window is None:
                    continue
                new_windows.append(window)
            p.close()
        windows = new_windows

        # Sort the windows to ensure that the dataset is consistent across GPUs.
        # Inconsistent ordering can lead to a subset of windows being processed during
        # "model test" / "model predict" when using multiple GPUs.
        # We use a hash so that functionality like num_samples limit gets a random
        # subset of windows (with respect to the hash function choice).
        windows.sort(
            key=lambda window: hashlib.sha256(window.name.encode()).hexdigest()
        )

        # Limit windows to num_samples if requested.
        if split_config.num_samples:
            # The windows are sorted by hash of window name so this distribution should
            # be representative of the population.
            windows = windows[0 : split_config.num_samples]

        # Write dataset_examples to a file so that we can load it lazily in the worker
        # processes. Otherwise it takes a long time to transmit it when spawning each
        # process.
        self.dataset_examples_fname = os.path.join(
            tempfile.gettempdir(),
            "rslearn_dataset_examples",
            f"{os.getpid()}_{uuid.uuid4()}.json",
        )
        self.num_dataset_examples = len(windows)
        self.dataset_examples: list[Window] | None = None
        logger.info(
            f"Writing {len(windows)} dataset examples to {self.dataset_examples_fname}"
        )
        os.makedirs(os.path.dirname(self.dataset_examples_fname), exist_ok=True)
        with open(self.dataset_examples_fname, "w") as f:
            json.dump([self._serialize_item(example) for example in windows], f)

    def _get_initial_windows(
        self, split_config: SplitConfig, workers: int
    ) -> list[Window]:
        """Get the initial windows before input layer filtering.

        The windows are filtered based on configured window names, groups, and tags.

        This is a helper for the init function.

        Args:
            split_config: the split configuration.
            workers: number of worker processes.

        Returns:
            list of windows from the dataset after applying the aforementioned filters.
        """
        # Load windows from dataset.
        # If the window storage is FileWindowStorage, we pass the workers/show_progress arguments.
        kwargs: dict[str, Any] = {}
        if isinstance(self.dataset.storage, FileWindowStorage):
            kwargs["workers"] = workers
            kwargs["show_progress"] = True
        # We also add the name/group filters to the kwargs.
        if split_config.names:
            kwargs["names"] = split_config.names
        if split_config.groups:
            kwargs["groups"] = split_config.groups

        windows = self.dataset.load_windows(**kwargs)

        # Filter by tags (if provided) using the window.options.
        if split_config.tags:
            new_windows = []
            num_removed: dict[str, int] = {}
            for window in windows:
                for k, v in split_config.tags.items():
                    if k not in window.options or (v and window.options[k] != v):
                        num_removed[k] = num_removed.get(k, 0) + 1
                        break
                else:
                    new_windows.append(window)
            logger.info(
                f"Started with {len(windows)} windows, ended with {len(new_windows)} windows for {self.dataset.path}"
            )
            for k, v in num_removed.items():
                logger.info(f"Removed {v} windows due to tag {k}")
            windows = new_windows

        return windows

    def _serialize_item(self, example: Window) -> dict[str, Any]:
        return example.get_metadata()

    def _deserialize_item(self, d: dict[str, Any]) -> Window:
        return Window.from_metadata(
            self.dataset.storage,
            d,
        )

    def get_dataset_examples(self) -> list[Window]:
        """Get a list of examples in the dataset.

        If load_all_patches is False, this is a list of Windows. Otherwise, this is a
        list of (window, patch_bounds, (patch_idx, # patches)) tuples.
        """
        if self.dataset_examples is None:
            logger.debug(
                f"Loading dataset examples from {self.dataset_examples_fname} in process {os.getpid()}"
            )
            with open(self.dataset_examples_fname) as f:
                self.dataset_examples = [
                    self._deserialize_item(d) for d in json.load(f)
                ]
            logger.debug(f"Finished loading dataset examples in process {os.getpid()}")
        return self.dataset_examples

    def __len__(self) -> int:
        """Returns the dataset length."""
        return self.num_dataset_examples

    def get_raw_inputs(
        self, idx: int
    ) -> tuple[dict[str, Any], dict[str, Any], SampleMetadata]:
        """Get the raw inputs and base metadata for this example.

        This is the raster or vector data before being processed by the Task. So it
        should be a Tensor for raster and list[Feature] for vector.

        Args:
            idx: the index in the dataset.

        Returns:
            a tuple (raw_inputs, passthrough_inputs, metadata).
        """
        dataset_examples = self.get_dataset_examples()
        example = dataset_examples[idx]
        rng = random.Random(idx if self.fix_patch_pick else None)

        # Select bounds to read.
        if self.patch_size:
            window = example

            def get_patch_range(n_patch: int, n_window: int) -> list[int]:
                if n_patch > n_window:
                    # Select arbitrary range containing the entire window.
                    # Basically arbitrarily padding the window to get to patch size.
                    start = rng.randint(n_window - n_patch, 0)
                    return [start, start + n_patch]

                else:
                    # Select arbitrary patch within the window.
                    start = rng.randint(0, n_window - n_patch)
                    return [start, start + n_patch]

            window_size = (
                window.bounds[2] - window.bounds[0],
                window.bounds[3] - window.bounds[1],
            )
            patch_ranges = [
                get_patch_range(self.patch_size[0], window_size[0]),
                get_patch_range(self.patch_size[1], window_size[1]),
            ]
            bounds = (
                window.bounds[0] + patch_ranges[0][0],
                window.bounds[1] + patch_ranges[1][0],
                window.bounds[0] + patch_ranges[0][1],
                window.bounds[1] + patch_ranges[1][1],
            )

        else:
            window = example
            bounds = window.bounds

        assert isinstance(window, Window)

        raw_inputs = {}
        passthrough_inputs = {}
        for name, data_input in self.inputs.items():
            raw_inputs[name] = read_data_input(
                self.dataset, window, bounds, data_input, rng
            )
            if data_input.passthrough:
                passthrough_inputs[name] = raw_inputs[name]

        metadata = SampleMetadata(
            window_group=window.group,
            window_name=window.name,
            window_bounds=window.bounds,
            patch_bounds=bounds,
            patch_idx=0,
            num_patches_in_window=1,
            time_range=window.time_range,
            projection=window.projection,
            dataset_source=self.name,
        )

        return raw_inputs, passthrough_inputs, metadata

    def __getitem__(
        self, idx: int
    ) -> tuple[dict[str, Any], dict[str, Any], SampleMetadata]:
        """Read one training example.

        Args:
            idx: the index in the dataset.

        Returns:
            a tuple (input_dict, target_dict, metadata)
        """
        logger.debug("__getitem__ start pid=%d item_idx=%d", os.getpid(), idx)

        raw_inputs, passthrough_inputs, metadata = self.get_raw_inputs(idx)

        input_dict, target_dict = self.task.process_inputs(
            raw_inputs,
            metadata=metadata,
            load_targets=not self.split_config.get_skip_targets(),
        )
        input_dict.update(passthrough_inputs)
        input_dict, target_dict = self.transforms(input_dict, target_dict)

        logger.debug("__getitem__ finish pid=%d item_idx=%d", os.getpid(), idx)

        return input_dict, target_dict, metadata

    def set_name(self, name: str) -> None:
        """Set the name of the dataset.

        Args:
            name: the name to set.
        """
        self.name = name


class RetryDataset(torch.utils.data.Dataset):
    """A dataset wrapper that retries getitem upon encountering error."""

    def __init__(
        self, dataset: ModelDataset, retries: int = 3, delay: float = 5
    ) -> None:
        """Create a new RetryDataset.

        Args:
            dataset: the dataset to wrap.
            retries: the maximum number of tries before raising error.
            delay: how many seconds to sleep before retrying
        """
        self.dataset = dataset
        self.retries = retries
        self.delay = delay

    def set_name(self, name: str) -> None:
        """Set the name of the dataset.

        Args:
            name: the name to set.
        """
        self.dataset.set_name(name)

    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Any:
        """Get item from the dataset.

        The get operation is performed on the underlying dataset multiple times up to
        the configured maximum number of retries.

        Args:
            idx: the item index.

        Returns:
            the item data.
        """
        for _ in range(self.retries):
            try:
                return self.dataset[idx]
            except Exception as e:
                logger.warning("warning: caught exception loading item %d: %s", idx, e)
            time.sleep(self.delay)

        # One last try -- but don't catch any more errors.
        return self.dataset[idx]

    def get_dataset_examples(self) -> list[Window]:
        """Returns a list of windows in this dataset."""
        return self.dataset.get_dataset_examples()


class MultiDataset(torch.utils.data.Dataset):
    """A dataset that combines multiple datasets."""

    def __init__(self, datasets: dict[str, RetryDataset]) -> None:
        """Create a new MultiDataset.

        Args:
            datasets: map of dataset name to dataset.
        """
        self.datasets = datasets
        self.buckets = {}
        curr_offset = 0
        for name, ds in datasets.items():
            self.buckets[name] = range(curr_offset, curr_offset + len(ds))
            curr_offset += len(ds)

    def __len__(self) -> int:
        """Return length of the dataset."""
        return sum(len(ds) for ds in self.datasets.values())

    def __getitem__(self, idx: int) -> Any:
        """Get item from the dataset.

        Args:
            idx: the item index.

        Returns:
            the item data.
        """
        for name, bucket in self.buckets.items():
            if idx in bucket:
                return self.datasets[name][idx - bucket.start]
        raise IndexError(f"Index {idx} out of range (len={len(self)})")
