"""Default Dataset for rslearn."""

import os
import random
from typing import Any, Optional, Union

import torch

from rslearn.config import RasterFormatConfig, RasterLayerConfig, VectorLayerConfig
from rslearn.dataset import Dataset, Window
from rslearn.train.tasks import Task
from rslearn.utils import LocalFileAPI
from rslearn.utils.raster_format import load_raster_format
from rslearn.utils.vector_format import load_vector_format

from .transforms import Sequential


class SamplerFactory:
    """Factory to produce a Sampler.

    This enables configuring a sampler without needing to pass the dataset.
    """

    def get_sampler(
        self, dataset: torch.utils.data.Dataset
    ) -> torch.utils.data.Sampler:
        """Create a sampler for the given dataset.

        Args:
            dataset: the dataset

        Returns:
            a sampler
        """
        raise NotImplementedError


class RandomSamplerFactory(SamplerFactory):
    """A sampler factory for RandomSampler."""

    def __init__(self, replacement: bool = False, num_samples: Optional[int] = None):
        """Initialize a RandomSamplerFactory.

        Args:
            replacement: whether to pick with replacement, default false
            num_samples: optional number of dataset samples to limit iteration to,
                otherwise picks random samples equal to the dataset size
        """
        self.replacement = replacement
        self.num_samples = num_samples

    def get_sampler(
        self, dataset: torch.utils.data.Dataset
    ) -> torch.utils.data.Sampler:
        """Create a sampler for the given dataset.

        Args:
            dataset: the dataset

        Returns:
            a RandomSampler
        """
        return torch.utils.data.RandomSampler(
            dataset, replacement=self.replacement, num_samples=self.num_samples
        )


class DataInput:
    """Specification of a piece of data from a window that is needed for training.

    The DataInput includes which layer(s) the data can be obtained from for each window.
    """

    def __init__(
        self,
        data_type: str,
        layers: list[str],
        bands: Optional[list[str]] = None,
        required: bool = True,
        passthrough: bool = False,
    ):
        """Initialize a new DataInput.

        Args:
            data_type: either "raster" or "vector"
            layers: list of layer names that this input can be read from.
            bands: the bands to read, if this is a raster.
            required: whether examples lacking one of these layers should be skipped
            passthrough: whether to expose this to the model even if it isn't returned
                by any task
        """
        self.data_type = data_type
        self.layers = layers
        self.bands = bands
        self.required = required
        self.passthrough = passthrough


class SplitConfig:
    """Configuration that can be specified separately for train, val, and test."""

    def __init__(
        self,
        groups: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        num_samples: Optional[int] = None,
        transforms: Optional[list[torch.nn.Module]] = None,
        sampler: Optional[SamplerFactory] = None,
        patch_size: Optional[Union[int, tuple[int, int]]] = None,
    ):
        """Initialize a new SplitConfig.

        Args:
            groups: for this split, only read windows in one of these groups
            tags: todo
            num_samples: limit this split to this many examples
            transforms: transforms to apply
            sampler: SamplerFactory for this split
            patch_size: an optional square size or (width, height) tuple. If set, read
                crops of this size rather than entire windows.
        """
        self.groups = groups
        self.tags = tags
        self.num_samples = num_samples
        self.transforms = transforms
        self.sampler = sampler
        self.patch_size = patch_size

    def update(self, other: "SplitConfig") -> "SplitConfig":
        """Override settings in this SplitConfig with those in another.

        Returns:
            the resulting SplitConfig combining the settings.
        """
        result = SplitConfig(
            groups=self.groups,
            tags=self.tags,
            num_samples=self.num_samples,
            transforms=self.transforms,
            sampler=self.sampler,
            patch_size=self.patch_size,
        )
        if other.groups:
            result.groups = other.groups
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
        return result


class ModelDataset(torch.utils.data.Dataset):
    """The default pytorch dataset implementation for rslearn."""

    def __init__(
        self,
        root_dir: str,
        split_config: SplitConfig,
        inputs: dict[str, DataInput],
        task: Task,
    ):
        """Instantiate a new ModelDataset.

        Args:
            root_dir: the root directory of the dataset
            split_config: configuration specific to this split
            inputs: data to read from the dataset for training
            task: the task to train on
        """
        self.split_config = split_config
        self.inputs = inputs
        self.task = task

        if split_config.transforms:
            self.transforms = Sequential(*split_config.transforms)
        else:
            self.transforms = torch.nn.Identity()

        # Convert patch size to (width, height) format if needed.
        if not split_config.patch_size:
            self.patch_size = None
        elif isinstance(split_config.patch_size, int):
            self.patch_size = (split_config.patch_size, split_config.patch_size)
        else:
            self.patch_size = split_config.patch_size

        self.dataset = Dataset(root_dir)

        if split_config.groups:
            windows = self.dataset.load_windows(groups=split_config.groups)
        else:
            windows = self.dataset.load_windows()

        if split_config.tags:
            # TODO: some kind of window tagging system, can filter by it
            pass

        # Eliminate windows that are missing either a requisite input layer, or missing
        # all target layers.
        def check_window(window: Window) -> bool:
            def is_any_layer_available(data_input):
                for layer_name in data_input.layers:
                    if os.path.exists(
                        os.path.join(window.window_root, "layers", layer_name)
                    ):
                        return True
                return False

            for data_input in self.inputs.values():
                if not data_input.required:
                    continue
                if not is_any_layer_available(data_input):
                    return False

            return True

        windows = [window for window in windows if check_window(window)]

        # Limit windows to num_samples if requested.
        if split_config.num_samples:
            # TODO: use hash of window names so this is deterministic and not arbitrarily ordered according to load_windows
            windows = windows[0 : split_config.num_samples]

        self.windows = windows

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
        window_size = (
            window.bounds[2] - window.bounds[0],
            window.bounds[3] - window.bounds[1],
        )

        # Select bounds to read.
        if self.patch_size:

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
                if not os.path.exists(
                    os.path.join(window.window_root, "layers", layer_name)
                ):
                    continue
                layer_options.append(layer_name)

            # For now we just randomly pick one option.
            # In the future we need to support different configuration for how to pick
            # the options, as well as picking multiple for series inputs.
            layer = random.choice(layer_options)
            layer_dir = os.path.join(window.window_root, "layers", layer)
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
                    dtype=torch.float32,
                )

                for band_set, src_indexes, dst_indexes in needed_sets_and_indexes:
                    raster_format = load_raster_format(
                        RasterFormatConfig(band_set.format["name"], band_set.format)
                    )
                    file_api = LocalFileAPI(
                        os.path.join(layer_dir, "_".join(band_set.bands))
                    )
                    src = raster_format.decode_raster(file_api, bounds)
                    image[dst_indexes, :, :] = torch.as_tensor(
                        src[src_indexes, :, :]
                    ).float()

                return image

            elif data_input.data_type == "vector":
                assert isinstance(layer_config, VectorLayerConfig)
                vector_format = load_vector_format(layer_config.format)
                file_api = LocalFileAPI(layer_dir)
                features = vector_format.decode_vector(file_api, bounds)
                return features

            else:
                raise Exception(f"unknown data type {data_input.data_type}")

        raw_inputs = {}
        passthrough_inputs = {}
        for name, data_input in self.inputs.items():
            raw_inputs[name] = read_input(data_input)
            if data_input.passthrough:
                passthrough_inputs[name] = raw_inputs[name]

        input_dict, target_dict = self.task.process_inputs(raw_inputs)
        input_dict.update(passthrough_inputs)
        return self.transforms(input_dict, target_dict)
