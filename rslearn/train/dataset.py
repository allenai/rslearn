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


class TaskInput:
    """Specification of one input for a task."""

    def __init__(self, name: str, data_type: str):
        """Initialize a new TaskInput.

        Args:
            name: the name of this task.
            data_type: the data type to read
        """
        self.name = name
        self.data_type = data_type


class TaskConfig:
    """Specification of the inputs to read."""

    def __init__(self, inputs: list[TaskInput], targets: list[TaskInput]):
        """Initialize a new TaskConfig.

        Args:
            inputs: list of TaskInput objects that provide the inputs for the model
            targets: list of TaskInput objects that provide the targets for the model
        """
        self.inputs = inputs
        self.targets = targets


class DatasetInput:
    """Specification of how to read an input from a dataset."""

    def __init__(self, layers: list[str], bands: Optional[list[str]] = None):
        """Initialize a new DatasetInput.

        Args:
            layers: list of layer names that this input can be read from.
            bands: the bands to read, if this is a raster.
        """
        self.layers = layers
        self.bands = bands


class SplitConfig:
    """Configuration of windows for train, val, and test."""

    def __init__(
        self,
        groups: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        num_samples: Optional[int] = None,
    ):
        """Initialize a new SplitConfig.

        Args:
            groups: for this split, only read windows in one of these groups
            tags: todo
            num_samples: limit this split to this many examples
        """
        self.groups = groups
        self.tags = tags
        self.num_samples = num_samples


class DatasetConfig:
    """Specification of how to read data from one dataset."""

    def __init__(
        self,
        root_dir: str,
        inputs: dict[str, DatasetInput],
        targets: dict[str, DatasetInput],
        splits: dict[str, SplitConfig],
        patch_size: Optional[Union[int, tuple[int, int]]] = None,
    ):
        """Initialize a new DatasetConfig.

        Args:
            root_dir: the root directory of the dataset
            inputs: list of DatasetInput objects that provide the inputs for the model
            targets: list of DatasetInput objects that provide the targets for the model
            splits: configuration of train, val, and test splits
            patch_size: an optional square size or (width, height) tuple. If set, read
                crops of this size rather than entire windows.
        """
        self.root_dir = root_dir
        self.inputs = inputs
        self.targets = targets
        self.splits = splits
        self.patch_size = patch_size


class ModelDataset(torch.utils.data.Dataset):
    """The default pytorch dataset implemnetation for rslearn."""

    def __init__(
        self,
        tasks: dict[str, Task],
        task_config: TaskConfig,
        dataset_config: DatasetConfig,
        split: str,
        transforms: torch.nn.Module,
    ):
        """Instantiate a new ModelDataset.

        Args:
            tasks: the tasks to train on
            task_config: specification of the inputs to read
            dataset_config: where and how to read the inputs
            split: usually 'train', 'val', or 'test'
            transforms: transforms to apply
        """
        self.tasks = tasks
        self.task_config = task_config
        self.dataset_config = dataset_config
        self.transforms = transforms

        # Convert patch size to (width, height) format if needed.
        if not dataset_config.patch_size:
            self.patch_size = None
        elif isinstance(dataset_config.patch_size, int):
            self.patch_size = (dataset_config.patch_size, dataset_config.patch_size)
        else:
            self.patch_size = dataset_config.patch_size

        self.dataset = Dataset(dataset_config.root_dir)
        split_config = dataset_config.splits[split]

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
            def is_any_layer_available(layers):
                for layer_name in dataset_input.layers:
                    if os.path.exists(
                        os.path.join(window.window_root, "layers", layer_name)
                    ):
                        return True
                return False

            for dataset_input in self.dataset_config.inputs.values():
                if not is_any_layer_available(dataset_input.layers):
                    return False

            any_target_available = False
            for dataset_input in self.dataset_config.targets.values():
                if is_any_layer_available(dataset_input.layers):
                    any_target_available = True
            if not any_target_available:
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
        def read_section(task_input: TaskInput, dataset_input: DatasetInput):
            # First enumerate all options of individual layers to read.
            layer_options = []
            for layer_name in dataset_input.layers:
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

            if task_input.data_type == "raster":
                assert isinstance(layer_config, RasterLayerConfig)

                # See what different sets of bands we need to read to get all the
                # configured bands.
                needed_bands = dataset_input.bands
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

            elif task_input.data_type == "vector":
                assert isinstance(layer_config, VectorLayerConfig)
                vector_format = load_vector_format(layer_config.format)
                file_api = LocalFileAPI(layer_dir)
                features = vector_format.decode_vector(file_api, bounds)
                return features

            else:
                raise Exception(f"unknown data type {task_input.data_type}")

        def load_dict(
            task_inputs: list[TaskInput], dataset_inputs: dict[str, DatasetInput]
        ):
            task_inputs_by_name = {
                task_input.name: task_input for task_input in task_inputs
            }
            d = {}
            for name, dataset_input in dataset_inputs.items():
                task_input = task_inputs_by_name[name]
                d[name] = read_section(task_input, dataset_input)
            return d

        input_dict = load_dict(self.task_config.inputs, self.dataset_config.inputs)
        target_dict = load_dict(self.task_config.targets, self.dataset_config.targets)

        for name, task in self.tasks.items():
            target_dict[name] = task.get_target(target_dict[name])

        return self.transforms(input_dict, target_dict)
