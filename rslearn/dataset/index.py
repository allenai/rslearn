"""Index about windows in the dataset."""

import json
import multiprocessing
from typing import TYPE_CHECKING

import tqdm
from upath import UPath

from .window import (
    LAYERS_DIRECTORY_NAME,
    Window,
    WindowLayerData,
    get_layer_and_group_from_dir_name,
)

if TYPE_CHECKING:
    from .dataset import Dataset


def get_window_layer_datas(window: Window) -> list[WindowLayerData]:
    """Helper function for multiprocessing to load window layer datas."""
    return list(window.load_layer_datas().values())


def get_window_completed_layers(window: Window) -> list[tuple[str, int]]:
    """Helper function for multiprocessing to load window completed layers."""
    # We don't know the range of possible groups a priori, so we instead look in the
    # layers directory.
    layers_directory = window.path / LAYERS_DIRECTORY_NAME
    if not layers_directory.exists():
        return []

    completed_layers = []
    for layer_dir in layers_directory.iterdir():
        layer_name, group_idx = get_layer_and_group_from_dir_name(layer_dir.name)
        if not window.is_layer_completed(layer_name, group_idx):
            continue
        completed_layers.append((layer_name, group_idx))
    return completed_layers


class DatasetIndex:
    """Manage an index about windows in the dataset.

    The index is just a single file containing information about all windows in the
    dataset, so that this information does not need to be loaded from per-window files.

    The information includes the window metadata, the window layer datas (matching data
    source items), and the completed layers.

    Currently the index must be manually maintained. It can be created for relatively
    static datasets, and updated each time the dataset is modified.
    """

    FNAME = "dataset_index.json"

    def __init__(
        self,
        windows: list[Window],
        layer_datas: dict[str, list[WindowLayerData]],
        completed_layers: dict[str, list[tuple[str, int]]],
    ):
        """Create a new DatasetIndex.

        Args:
            windows: the windows in the dataset.
            layer_datas: map from window name to the layer datas for that window.
            completed_layers: map from window name to its list of completed layers.
                Each element is (layer_name, group_idx).
        """
        self.windows = windows
        self.layer_datas = layer_datas
        self.completed_layers = completed_layers

        for window in self.windows:
            window.index = self

    def get_windows(
        self,
        groups: list[str] | None = None,
        names: list[str] | None = None,
    ) -> list[Window]:
        """Get the windows in the dataset.

        Args:
            groups: an optional list of groups to filter by
            names: an optional list of window names to filter by
        """
        windows = self.windows
        if groups is not None:
            group_set = set(groups)
            windows = [window for window in windows if window.group in group_set]
        if names is not None:
            name_set = set(names)
            windows = [window for window in windows if window.name in name_set]
        return windows

    def save_index(self, ds_path: UPath) -> None:
        """Save the index to the specified file."""
        encoded_windows = [window.get_metadata() for window in self.windows]

        encoded_layer_datas = {}
        for window_name, layer_data_list in self.layer_datas.items():
            encoded_layer_datas[window_name] = [
                layer_data.serialize() for layer_data in layer_data_list
            ]

        encoded_index = {
            "windows": encoded_windows,
            "layer_datas": encoded_layer_datas,
            "completed_layers": self.completed_layers,
        }
        with (ds_path / self.FNAME).open("w") as f:
            json.dump(encoded_index, f)

    @staticmethod
    def load_index(ds_path: UPath) -> "DatasetIndex":
        """Load the DatasetIndex for the specified dataset."""
        with (ds_path / DatasetIndex.FNAME).open() as f:
            encoded_index = json.load(f)

        windows = []
        for encoded_window in encoded_index["windows"]:
            window = Window.from_metadata(
                path=Window.get_window_root(
                    ds_path, encoded_window["group"], encoded_window["name"]
                ),
                metadata=encoded_window,
            )
            windows.append(window)

        layer_datas = {}
        for window_name, encoded_layer_data_list in encoded_index[
            "layer_datas"
        ].items():
            layer_datas[window_name] = [
                WindowLayerData.deserialize(encoded_layer_data)
                for encoded_layer_data in encoded_layer_data_list
            ]

        completed_layers = {}
        for window_name, encoded_layer_list in encoded_index[
            "completed_layers"
        ].items():
            completed_layers[window_name] = [
                (layer_name, group_idx)
                for (layer_name, group_idx) in encoded_layer_list
            ]

        return DatasetIndex(
            windows=windows,
            layer_datas=layer_datas,
            completed_layers=completed_layers,
        )

    @staticmethod
    def build_index(dataset: "Dataset", workers: int) -> "DatasetIndex":
        """Build a new DatasetIndex for the specified dataset."""
        # Load windows.
        windows = dataset.load_windows(workers=workers, no_index=True)

        # Load layer datas.
        p = multiprocessing.Pool(workers)
        layer_data_outputs = p.imap(get_window_layer_datas, windows)
        layer_data_outputs = tqdm.tqdm(
            layer_data_outputs, total=len(windows), desc="Loading window layer datas"
        )
        layer_datas = {}
        for window, cur_layer_datas in zip(windows, layer_data_outputs):
            layer_datas[window.name] = cur_layer_datas

        # Load completed layers.
        completed_layer_outputs = p.imap(get_window_completed_layers, windows)
        completed_layer_outputs = tqdm.tqdm(
            completed_layer_outputs, total=len(windows), desc="Loading completed layers"
        )
        completed_layers = {}  # window name -> list of (layer name, group idx)
        for window, cur_completed_layers in zip(windows, completed_layer_outputs):
            completed_layers[window.name] = cur_completed_layers
        p.close()

        return DatasetIndex(
            windows=windows,
            layer_datas=layer_datas,
            completed_layers=completed_layers,
        )
