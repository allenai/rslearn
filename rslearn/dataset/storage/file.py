"""The default file-based dataset storage backend."""

import json
import multiprocessing

import tqdm
from upath import UPath

from rslearn.dataset.window import (
    LAYERS_DIRECTORY_NAME,
    Window,
    WindowLayerData,
    get_layer_and_group_from_dir_name,
    get_window_layer_dir,
)
from rslearn.log_utils import get_logger
from rslearn.utils.fsspec import open_atomic
from rslearn.utils.mp import star_imap_unordered

from .storage import DatasetStorage, DatasetStorageFactory

logger = get_logger(__name__)


def load_window(storage: "FileDatasetStorage", window_dir: UPath) -> Window:
    """Load the window from its directory by reading metadata.json.

    Args:
        storage: the underlying FileDatasetStorage.
        window_dir: the path where the window is stored.

    Returns:
        the window object.
    """
    metadata_fname = window_dir / "metadata.json"
    with metadata_fname.open() as f:
        metadata = json.load(f)
    return Window.from_metadata(storage, metadata)


class FileDatasetStorage(DatasetStorage):
    """The default file-backed dataset storage."""

    def __init__(self, path: UPath):
        """Create a new FileDatasetStorage.

        Args:
            path: the path to the dataset.
        """
        self.path = path

    def get_window_root(self, group: str, name: str) -> UPath:
        """Get the path where the window should be stored."""
        return Window.get_window_root(self.path, group, name)

    def get_windows(
        self,
        groups: list[str] | None = None,
        names: list[str] | None = None,
        show_progress: bool = False,
        workers: int = 0,
    ) -> list["Window"]:
        """Load the windows in the dataset.

        Args:
            groups: an optional list of groups to filter loading
            names: an optional list of window names to filter loading
            show_progress: whether to show tqdm progress bar
            workers: number of parallel workers, default 0 (use main thread only to load windows)
        """
        # Avoid directory does not exist errors later.
        if not (self.path / "windows").exists():
            return []

        window_dirs = []
        if not groups:
            groups = []
            for p in (self.path / "windows").iterdir():
                groups.append(p.name)
        for group in groups:
            group_dir = self.path / "windows" / group
            if not group_dir.exists():
                logger.warning(
                    f"Skipping group directory {group_dir} since it does not exist"
                )
                continue
            if names:
                cur_names = names
            else:
                cur_names = []
                for p in group_dir.iterdir():
                    cur_names.append(p.name)

            for window_name in cur_names:
                window_dir = group_dir / window_name
                window_dirs.append(window_dir)

        if workers == 0:
            windows = [load_window(self, window_dir) for window_dir in window_dirs]
        else:
            p = multiprocessing.Pool(workers)
            outputs = star_imap_unordered(
                p,
                load_window,
                [
                    dict(storage=self, window_dir=window_dir)
                    for window_dir in window_dirs
                ],
            )
            if show_progress:
                outputs = tqdm.tqdm(
                    outputs, total=len(window_dirs), desc="Loading windows"
                )
            windows = []
            for window in outputs:
                windows.append(window)
            p.close()

        return windows

    def create_or_update_window(self, window: Window) -> None:
        """Create or update the window.

        An existing window is only updated if there is one with the same name and group.

        If there is a window with the same name but a different group, the behavior is
        undefined.
        """
        window_path = self.get_window_root(window.group, window.name)
        window_path.mkdir(parents=True, exist_ok=True)
        metadata_path = window_path / "metadata.json"
        logger.debug(f"Saving window metadata to {metadata_path}")
        with open_atomic(metadata_path, "w") as f:
            json.dump(window.get_metadata(), f)

    def get_layer_datas(self, group: str, name: str) -> dict[str, "WindowLayerData"]:
        """Get the window layer datas for the specified window.

        Args:
            group: the window group.
            name: the window name.

        Returns:
            a dict mapping from the layer name to the layer data for that layer, if one
                was previously saved.
        """
        window_path = self.get_window_root(group, name)
        items_fname = window_path / "items.json"
        if not items_fname.exists():
            return {}

        with items_fname.open() as f:
            layer_datas = [
                WindowLayerData.deserialize(layer_data) for layer_data in json.load(f)
            ]

        return {layer_data.layer_name: layer_data for layer_data in layer_datas}

    def save_layer_datas(
        self, group: str, name: str, layer_datas: dict[str, "WindowLayerData"]
    ) -> None:
        """Set the window layer datas for the specified window."""
        window_path = self.get_window_root(group, name)
        json_data = [layer_data.serialize() for layer_data in layer_datas.values()]
        items_fname = window_path / "items.json"
        logger.info(f"Saving window items to {items_fname}")
        with open_atomic(items_fname, "w") as f:
            json.dump(json_data, f)

    def list_completed_layers(self, group: str, name: str) -> list[tuple[str, int]]:
        """List the layers available for this window that are completed.

        Args:
            group: the window group.
            name: the window name.

        Returns:
            a list of (layer_name, group_idx) completed layers.
        """
        window_path = self.get_window_root(group, name)
        layers_directory = window_path / LAYERS_DIRECTORY_NAME
        if not layers_directory.exists():
            return []

        completed_layers = []
        for layer_dir in layers_directory.iterdir():
            layer_name, group_idx = get_layer_and_group_from_dir_name(layer_dir.name)
            if not self.is_layer_completed(group, name, layer_name, group_idx):
                continue
            completed_layers.append((layer_name, group_idx))

        return completed_layers

    def is_layer_completed(
        self, group: str, name: str, layer_name: str, group_idx: int = 0
    ) -> bool:
        """Check whether the specified layer is completed in the given window.

        Completed means there is data in the layer and the data has been written
        (materialized).

        Args:
            group: the window group.
            name: the window name.
            layer_name: the layer name.
            group_idx: the index of the group within the layer.

        Returns:
            whether the layer is completed.
        """
        window_path = self.get_window_root(group, name)
        layer_dir = get_window_layer_dir(
            window_path,
            layer_name,
            group_idx,
        )
        return (layer_dir / "completed").exists()

    def mark_layer_completed(
        self, group: str, name: str, layer_name: str, group_idx: int = 0
    ) -> None:
        """Mark the specified layer completed for the given window.

        This must be done after the contents of the layer have been written. If a layer
        has multiple groups, the caller should wait until the contents of all groups
        have been written before marking them completed; this is because, when
        materializing a window, we skip materialization if the first group
        (group_idx=0) is marked completed.

        Args:
            group: the window group.
            name: the window name.
            layer_name: the layer name.
            group_idx: the index of the group within the layer.
        """
        window_path = self.get_window_root(group, name)
        layer_dir = get_window_layer_dir(window_path, layer_name, group_idx)
        # We assume the directory exists because the layer should be materialized before
        # being marked completed.
        (layer_dir / "completed").touch()


class FileDatasetStorageFactory(DatasetStorageFactory):
    """Factory class for FileDatasetStorage."""

    def get_storage(self, ds_path: UPath) -> FileDatasetStorage:
        """Get a FileDatasetStorage for the given dataset path."""
        return FileDatasetStorage(ds_path)
