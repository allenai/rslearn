"""Default Dataset for rslearn."""

import multiprocessing
import random
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
from rslearn.dataset import Dataset, Window
from rslearn.train.tasks import Task
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import load_raster_format
from rslearn.utils.vector_format import load_vector_format

from .transforms import Sequential


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

    def __init__(self, replacement: bool = False, num_samples: int | None = None):
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

    def __init__(self, option_key: str, num_samples: int, replacement: bool = True):
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
        for window in dataset.windows:
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
    ):
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
        """
        self.data_type = data_type
        self.layers = layers
        self.bands = bands
        self.required = required
        self.passthrough = passthrough
        self.is_target = is_target
        self.dtype = dtype


class SplitConfig:
    """Configuration that can be specified separately for train, val, and test."""

    def __init__(
        self,
        groups: list[str] | None = None,
        names: list[str] | None = None,
        tags: dict[str, str] | None = None,
        num_samples: int | None = None,
        transforms: list[torch.nn.Module] | None = None,
        sampler: SamplerFactory | None = None,
        patch_size: int | tuple[int, int] | None = None,
        load_all_patches: bool | None = None,
        skip_targets: bool | None = None,
    ):
        """Initialize a new SplitConfig.

        Args:
            groups: for this split, only read windows in one of these groups
            names: for this split, read windows with these specific names
            tags: only select windows that have options matching these tags. If key and
                value are set, then window must have an option with the same key and
                value. If value is empty, then only the existince of the key in the
                window options is checked.
            num_samples: limit this split to this many examples
            transforms: transforms to apply
            sampler: SamplerFactory for this split
            patch_size: an optional square size or (width, height) tuple. If set, read
                crops of this size rather than entire windows.
            load_all_patches: with patch_size set, rather than sampling a random patch
                for each window, read all patches as separate sequential items in the
                dataset.
            skip_targets: whether to skip targets when loading inputs
        """
        self.groups = groups
        self.names = names
        self.tags = tags
        self.num_samples = num_samples
        self.transforms = transforms
        self.sampler = sampler
        self.patch_size = patch_size
        self.load_all_patches = load_all_patches
        self.skip_targets = skip_targets

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
            transforms=self.transforms,
            sampler=self.sampler,
            patch_size=self.patch_size,
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
        if other.transforms:
            result.transforms = other.transforms
        if other.sampler:
            result.sampler = other.sampler
        if other.patch_size:
            result.patch_size = other.patch_size
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


def check_window(inputs: dict[str, DataInput], window: Window) -> bool:
    """Verify that the window has the required layers based on the specified inputs.

    Args:
        inputs: the inputs to the dataset.
        window: the window to check.

    Returns:
        the window if it has all the required inputs or None otherwise
    """

    # Make sure window has all the needed layers.
    def is_any_layer_available(data_input):
        for layer_name in data_input.layers:
            completed_fname = window.path / "layers" / layer_name / "completed"
            if completed_fname.exists():
                return True
        return False

    for data_input in inputs.values():
        if not data_input.required:
            continue
        if not is_any_layer_available(data_input):
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
    ):
        """Instantiate a new ModelDataset.

        Args:
            dataset: underlying rslearn dataset to read data from
            split_config: configuration specific to this split
            inputs: data to read from the dataset for training
            task: the task to train on
            workers: number of workers to use for initializing the dataset
        """
        self.dataset = dataset
        self.split_config = split_config
        self.inputs = inputs
        self.task = task

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
            for window in windows:
                for k, v in split_config.tags.items():
                    if k not in window.options:
                        continue
                    if v and window.options[k] != v:
                        continue
                    new_windows.append(window)
            windows = new_windows

        # If targets are not needed, remove them from the inputs.
        if split_config.get_skip_targets():
            for k in list(self.inputs.keys()):
                if self.inputs[k].is_target:
                    del self.inputs[k]

        # Eliminate windows that are missing either a requisite input layer, or missing
        # all target layers.
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
        new_windows = []
        for window in tqdm.tqdm(
            outputs, total=len(windows), desc="Checking available layers in windows"
        ):
            if window is None:
                continue
            new_windows.append(window)
        p.close()
        windows = new_windows

        # Limit windows to num_samples if requested.
        if split_config.num_samples:
            # TODO: use hash of window names so this is deterministic and not arbitrarily ordered according to load_windows
            windows = windows[0 : split_config.num_samples]

        self.windows = windows

        # If we're loading all patches, we need to include the patch details.
        if split_config.get_load_all_patches():
            patches = []
            for window in self.windows:
                cur_patches = []
                for col in range(
                    window.bounds[0], window.bounds[2], self.patch_size[0]
                ):
                    for row in range(
                        window.bounds[1], window.bounds[3], self.patch_size[1]
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
            self.windows = patches

    def __len__(self) -> int:
        """Returns the dataset length."""
        return len(self.windows)

    def __getitem__(self, idx) -> tuple[dict[str, Any], dict[str, Any]]:
        """Read one training example.

        Args:
            idx: the index in the dataset.

        Returns:
            a tuple (input_dict, target_dict)
        """
        window = self.windows[idx]

        # Select bounds to read.
        if self.split_config.get_load_all_patches():
            window, bounds, (patch_idx, num_patches) = window
        elif self.patch_size:

            def get_patch_range(n_patch, n_window):
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
            bounds = window.bounds

        # Read the inputs and targets.
        def read_input(data_input: DataInput):
            # First enumerate all options of individual layers to read.
            layer_options = []
            for layer_name in data_input.layers:
                completed_fname = window.path / "layers" / layer_name / "completed"
                if not completed_fname.exists():
                    continue
                layer_options.append(layer_name)

            # For now we just randomly pick one option.
            # In the future we need to support different configuration for how to pick
            # the options, as well as picking multiple for series inputs.
            layer = random.choice(layer_options)
            layer_dir = window.path / "layers" / layer
            layer_config = self.dataset.layers[layer]

            if data_input.data_type == "raster":
                assert isinstance(layer_config, RasterLayerConfig)

                # See what different sets of bands we need to read to get all the
                # configured bands.
                needed_bands = data_input.bands
                needed_band_indexes = {}
                for i, band in enumerate(needed_bands):
                    needed_band_indexes[band] = i
                needed_sets_and_indexes = []
                for band_set in layer_config.band_sets:
                    needed_src_indexes = []
                    needed_dst_indexes = []
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
                    raise Exception(
                        "could not get all the needed bands from "
                        + f"window {window.name} layer {layer}"
                    )

                image = torch.zeros(
                    (len(needed_bands), bounds[3] - bounds[1], bounds[2] - bounds[0]),
                    dtype=data_input.dtype.get_torch_dtype(),
                )

                for band_set, src_indexes, dst_indexes in needed_sets_and_indexes:
                    _, final_bounds = band_set.get_final_projection_and_bounds(
                        window.projection, bounds
                    )
                    raster_format = load_raster_format(
                        RasterFormatConfig(band_set.format["name"], band_set.format)
                    )
                    cur_path = layer_dir / "_".join(band_set.bands)
                    src = raster_format.decode_raster(cur_path, final_bounds)

                    # Resize to patch size if needed.
                    # This is for band sets that are stored at a lower resolution.
                    # Here we assume that it is a multiple.
                    if src.shape[1:3] != image.shape[1:3]:
                        if src.shape[1] < image.shape[1]:
                            factor = image.shape[1] // src.shape[1]
                            src = src.repeat(repeats=factor, axis=1).repeat(
                                repeats=factor, axis=2
                            )
                        else:
                            factor = src.shape[1] // image.shape[1]
                            src = src[:, ::factor, ::factor]

                    image[dst_indexes, :, :] = torch.as_tensor(
                        src[src_indexes, :, :].astype(
                            data_input.dtype.get_numpy_dtype()
                        )
                    )

                return image

            elif data_input.data_type == "vector":
                assert isinstance(layer_config, VectorLayerConfig)
                vector_format = load_vector_format(layer_config.format)
                features = vector_format.decode_vector(layer_dir, bounds)
                return features

            else:
                raise Exception(f"unknown data type {data_input.data_type}")

        raw_inputs = {}
        passthrough_inputs = {}
        for name, data_input in self.inputs.items():
            raw_inputs[name] = read_input(data_input)
            if data_input.passthrough:
                passthrough_inputs[name] = raw_inputs[name]

        metadata = {
            "group": window.group,
            "window_name": window.name,
            "window_bounds": window.bounds,
            "bounds": bounds,
            "time_range": window.time_range,
            "projection": window.projection,
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

        return input_dict, target_dict, metadata
