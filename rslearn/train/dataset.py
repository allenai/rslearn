"""Default Dataset for rslearn."""

import hashlib
import json
import multiprocessing
import os
import random
import tempfile
import time
import uuid
from typing import Any

import torch
import tqdm

import rslearn.train.transforms.transform
from rslearn.config import (
    DType,
    RasterFormatConfig,
    RasterLayerConfig,
    VectorLayerConfig,
)
from rslearn.dataset.dataset import Dataset
from rslearn.dataset.window import Window, get_layer_and_group_from_dir_name
from rslearn.log_utils import get_logger
from rslearn.train.tasks import Task
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import PixelBounds
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import load_raster_format
from rslearn.utils.vector_format import load_vector_format

from .transforms import Sequential

logger = get_logger(__name__)


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


class DataInput:
    """Specification of a piece of data from a window that is needed for training.

    The DataInput includes which layer(s) the data can be obtained from for each window.
    """

    def __init__(
        self,
        data_type: str,
        layers: list[str],
        bands: list[str] | None = None,
        required: bool = True,
        passthrough: bool = False,
        is_target: bool = False,
        dtype: DType = DType.FLOAT32,
        load_all_layers: bool = False,
        load_all_item_groups: bool = False,
    ) -> None:
        """Initialize a new DataInput.

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
                dimension.
            load_all_item_groups: whether to load all item groups in the layer(s) we
                are reading from. By default, we assume the specified layer name is of
                the form "{layer_name}.{group_idx}" and read that item group only. With
                this option enabled, we ignore the group_idx and read all item groups.
        """
        self.data_type = data_type
        self.layers = layers
        self.bands = bands
        self.required = required
        self.passthrough = passthrough
        self.is_target = is_target
        self.dtype = dtype
        self.load_all_layers = load_all_layers
        self.load_all_item_groups = load_all_item_groups


def read_raster_layer_for_data_input(
    window: Window,
    bounds: PixelBounds,
    layer_name: str,
    group_idx: int,
    layer_config: RasterLayerConfig,
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

    image = torch.zeros(
        (len(needed_bands), bounds[3] - bounds[1], bounds[2] - bounds[0]),
        dtype=data_input.dtype.get_torch_dtype(),
    )

    for band_set, src_indexes, dst_indexes in needed_sets_and_indexes:
        final_projection, final_bounds = band_set.get_final_projection_and_bounds(
            window.projection, bounds
        )
        if band_set.format is None:
            raise ValueError(f"No format specified for {layer_name}")
        raster_format = load_raster_format(
            RasterFormatConfig(band_set.format["name"], band_set.format)
        )
        raster_dir = window.get_raster_dir(
            layer_name, band_set.bands, group_idx=group_idx
        )
        src = raster_format.decode_raster(raster_dir, final_projection, final_bounds)
        if src is None:
            raise ValueError(f"Source is None for {data_input}")
        # Resize to patch size if needed.
        # This is for band sets that are stored at a lower resolution.
        # Here we assume that it is a multiple.
        if src.shape[1:3] != image.shape[1:3]:
            if src.shape[1] < image.shape[1]:
                factor = image.shape[1] // src.shape[1]
                src = src.repeat(repeats=factor, axis=1).repeat(repeats=factor, axis=2)
            else:
                factor = src.shape[1] // image.shape[1]
                src = src[:, ::factor, ::factor]

        image[dst_indexes, :, :] = torch.as_tensor(
            src[src_indexes, :, :].astype(data_input.dtype.get_numpy_dtype())
        )

    return image


def read_data_input(
    dataset: Dataset, window: Window, bounds: PixelBounds, data_input: DataInput
) -> torch.Tensor | list[Feature]:
    """Read the data specified by the DataInput from the window.

    Args:
        dataset: the dataset, to get layer configs.
        window: the window to read from.
        bounds: the bounds of the patch we are reading.
        data_input: the DataInput that specifies what layers to read.

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
        layers_to_read = [random.choice(layer_options)]

    if data_input.data_type == "raster":
        images: list[torch.Tensor] = []
        for layer_name, group_idx in layers_to_read:
            layer_config = dataset.layers[layer_name]
            assert isinstance(layer_config, RasterLayerConfig)
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
            assert isinstance(layer_config, VectorLayerConfig)
            vector_format = load_vector_format(layer_config.format)
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
        self.load_all_patches = load_all_patches
        self.skip_targets = skip_targets
        self.overlap_ratio = overlap_ratio
        if self.overlap_ratio is not None:
            if not (0 < self.overlap_ratio < 1):
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
    def is_any_layer_available(data_input: DataInput) -> bool:
        for layer_name in data_input.layers:
            if window.is_layer_completed(layer_name):
                return True
        return False

    for data_input in inputs.values():
        if not data_input.required:
            continue
        if not is_any_layer_available(data_input):
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
    ) -> None:
        """Instantiate a new ModelDataset.

        Args:
            dataset: underlying rslearn dataset to read data from
            split_config: configuration specific to this split
            inputs: data to read from the dataset for training
            task: the task to train on
            workers: number of workers to use for initializing the dataset
            name: name of the dataset (default: None)
        """
        self.dataset = dataset
        self.split_config = split_config
        self.inputs = inputs
        self.task = task
        self.name = name
        if split_config.transforms:
            self.transforms = Sequential(*split_config.transforms)
        else:
            self.transforms = rslearn.train.transforms.transform.Identity()

        # Convert patch size to (width, height) format if needed.
        if not split_config.patch_size:
            self.patch_size = None
        elif isinstance(split_config.patch_size, int):
            self.patch_size = (split_config.patch_size, split_config.patch_size)
        else:
            self.patch_size = split_config.patch_size

        if split_config.names:
            windows = self.dataset.load_windows(
                groups=split_config.groups,
                names=split_config.names,
                show_progress=True,
                workers=workers,
            )
        elif split_config.groups:
            windows = self.dataset.load_windows(
                groups=split_config.groups, show_progress=True, workers=workers
            )
        else:
            windows = self.dataset.load_windows(show_progress=True, workers=workers)

        if split_config.tags:
            # Filter the window.options.
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

        # If targets are not needed, remove them from the inputs.
        if split_config.get_skip_targets():
            for k in list(self.inputs.keys()):
                if self.inputs[k].is_target:
                    del self.inputs[k]

        # Eliminate windows that are missing either a requisite input layer, or missing
        # all target layers.
        # We use only main thread if the index is set, since that can take a long time
        # to send to the worker threads, it may get serialized for each window.
        new_windows = []
        if workers == 0 or (len(windows) >= 1 and windows[0].index is not None):
            for window in windows:
                if check_window(self.inputs, window) is None:
                    continue
                # The index may be set, but now that this check is done, from here on
                # we no longer need it. We set it None so that we don't end up passing
                # it later to the dataloader workers.
                window.index = None
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

        dataset_examples: list = windows

        # If we're loading all patches, we need to include the patch details.
        # In this case, dataset_examples is a list of (window, bounds, (patch_idx, # patches)).
        if split_config.get_load_all_patches() and self.patch_size is not None:
            patches = []
            overlap_size = int(
                self.patch_size[0] * split_config.overlap_ratio
                if split_config.overlap_ratio
                else 0
            )
            for window in dataset_examples:
                cur_patches = []
                if window is None:
                    raise ValueError("Window is None in load_all_patches")
                for col in range(
                    window.bounds[0],
                    window.bounds[2],
                    self.patch_size[0] - overlap_size,
                ):
                    for row in range(
                        window.bounds[1],
                        window.bounds[3],
                        self.patch_size[1] - overlap_size,
                    ):
                        cur_patches.append(
                            (
                                col,
                                row,
                                col + self.patch_size[0],
                                row + self.patch_size[1],
                            )
                        )
                for i, patch_bounds in enumerate(cur_patches):
                    patches.append((window, patch_bounds, (i, len(cur_patches))))

            if split_config.num_patches:
                patches = patches[0 : split_config.num_patches]

            dataset_examples = patches

        # Write dataset_examples to a file so that we can load it lazily in the worker
        # processes. Otherwise it takes a long time to transmit it when spawning each
        # process.
        self.dataset_examples_fname = os.path.join(
            tempfile.gettempdir(),
            "rslearn_dataset_examples",
            f"{os.getpid()}_{uuid.uuid4()}.json",
        )
        self.num_dataset_examples = len(dataset_examples)
        self.dataset_examples: list | None = None
        logger.info(
            f"Writing {len(dataset_examples)} dataset examples to {self.dataset_examples_fname}"
        )
        os.makedirs(os.path.dirname(self.dataset_examples_fname), exist_ok=True)
        with open(self.dataset_examples_fname, "w") as f:
            json.dump(
                [self._serialize_item(example) for example in dataset_examples], f
            )

    def _serialize_item(self, example: Window | tuple) -> dict[str, Any]:
        if isinstance(example, Window):
            return {
                "window": example.get_metadata(),
            }
        else:
            window, patch_bounds, indices = example
            return {
                "window": window.get_metadata(),
                "patch_bounds": patch_bounds,
                "indices": indices,
            }

    def _deserialize_item(self, d: dict[str, Any]) -> Window | tuple:
        window_metadata = d["window"]
        window = Window.from_metadata(
            Window.get_window_root(
                self.dataset.path, window_metadata["group"], window_metadata["name"]
            ),
            window_metadata,
        )
        if "patch_bounds" in d:
            return (window, tuple(d["patch_bounds"]), tuple(d["indices"]))
        else:
            return window

    def get_dataset_examples(self) -> list:
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

    def __getitem__(
        self, idx: int
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Read one training example.

        Args:
            idx: the index in the dataset.

        Returns:
            a tuple (input_dict, target_dict)
        """
        dataset_examples = self.get_dataset_examples()
        logger.debug("__getitem__ start pid=%d item_idx=%d", os.getpid(), idx)
        example = dataset_examples[idx]

        # Select bounds to read.
        if self.split_config.get_load_all_patches():
            window, bounds, (patch_idx, num_patches) = example

        elif self.patch_size:
            window = example

            def get_patch_range(n_patch: int, n_window: int) -> list[int]:
                if n_patch > n_window:
                    # Select arbitrary range containing the entire window.
                    # Basically arbitrarily padding the window to get to patch size.
                    start = random.randint(n_window - n_patch, 0)
                    return [start, start + n_patch]

                else:
                    # Select arbitrary patch within the window.
                    start = random.randint(0, n_window - n_patch)
                    return [start, start + n_patch]

            window_size = (
                window.bounds[2] - window.bounds[0],
                window.bounds[3] - window.bounds[1],
            )
            patch_ranges = [
                get_patch_range(self.patch_size[0], window_size[0]),
                get_patch_range(self.patch_size[1], window_size[1]),
            ]
            bounds = [
                window.bounds[0] + patch_ranges[0][0],
                window.bounds[1] + patch_ranges[1][0],
                window.bounds[0] + patch_ranges[0][1],
                window.bounds[1] + patch_ranges[1][1],
            ]

        else:
            window = example
            bounds = window.bounds

        assert isinstance(window, Window)

        raw_inputs = {}
        passthrough_inputs = {}
        for name, data_input in self.inputs.items():
            raw_inputs[name] = read_data_input(self.dataset, window, bounds, data_input)
            if data_input.passthrough:
                passthrough_inputs[name] = raw_inputs[name]

        metadata = {
            "group": window.group,
            "window_name": window.name,
            "window_bounds": window.bounds,
            "bounds": bounds,
            "time_range": window.time_range,
            "projection": window.projection,
            "dataset_source": self.name,
        }
        if self.split_config.get_load_all_patches():
            metadata["patch_idx"] = patch_idx
            metadata["num_patches"] = num_patches
        else:
            metadata["patch_idx"] = 0
            metadata["num_patches"] = 1

        input_dict, target_dict = self.task.process_inputs(
            raw_inputs,
            metadata=metadata,
            load_targets=not self.split_config.get_skip_targets(),
        )
        input_dict.update(passthrough_inputs)
        input_dict, target_dict = self.transforms(input_dict, target_dict)
        input_dict["dataset_source"] = self.name

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

    def get_dataset_examples(self) -> list:
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
