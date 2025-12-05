"""The default file-based dataset storage backend."""

import json
import multiprocessing

import tqdm
from typing_extensions import override
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

from .storage import WindowStorage, WindowStorageFactory

logger = get_logger(__name__)


def load_window(storage: "FileWindowStorage", window_dir: UPath) -> Window:
    """Load the window from its directory by reading metadata.json.

    Args:
        storage: the underlying FileWindowStorage.
        window_dir: the path where the window is stored.

    Returns:
        the window object.
    """
    metadata_fname = window_dir / "metadata.json"
    with metadata_fname.open() as f:
        metadata = json.load(f)
    return Window.from_metadata(storage, metadata)


class FileWindowStorage(WindowStorage):
    """The default file-backed dataset storage."""

    def __init__(self, path: UPath):
        """Create a new FileWindowStorage.

        Args:
            path: the path to the dataset.
        """
        self.path = path

    @override
    def get_window_root(self, group: str, name: str) -> UPath:
        return Window.get_window_root(self.path, group, name)

    @override
    def get_windows(
        self,
        groups: list[str] | None = None,
        names: list[str] | None = None,
        show_progress: bool = False,
        workers: int = 0,
    ) -> list["Window"]:
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

    @override
    def create_or_update_window(self, window: Window) -> None:
        window_path = self.get_window_root(window.group, window.name)
        window_path.mkdir(parents=True, exist_ok=True)
        metadata_path = window_path / "metadata.json"
        logger.debug(f"Saving window metadata to {metadata_path}")
        with open_atomic(metadata_path, "w") as f:
            json.dump(window.get_metadata(), f)

    @override
    def get_layer_datas(self, group: str, name: str) -> dict[str, "WindowLayerData"]:
        window_path = self.get_window_root(group, name)
        items_fname = window_path / "items.json"
        if not items_fname.exists():
            return {}

        with items_fname.open() as f:
            layer_datas = [
                WindowLayerData.deserialize(layer_data) for layer_data in json.load(f)
            ]

        return {layer_data.layer_name: layer_data for layer_data in layer_datas}

    @override
    def save_layer_datas(
        self, group: str, name: str, layer_datas: dict[str, "WindowLayerData"]
    ) -> None:
        window_path = self.get_window_root(group, name)
        json_data = [layer_data.serialize() for layer_data in layer_datas.values()]
        items_fname = window_path / "items.json"
        logger.info(f"Saving window items to {items_fname}")
        with open_atomic(items_fname, "w") as f:
            json.dump(json_data, f)

    @override
    def list_completed_layers(self, group: str, name: str) -> list[tuple[str, int]]:
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

    @override
    def is_layer_completed(
        self, group: str, name: str, layer_name: str, group_idx: int = 0
    ) -> bool:
        window_path = self.get_window_root(group, name)
        layer_dir = get_window_layer_dir(
            window_path,
            layer_name,
            group_idx,
        )
        return (layer_dir / "completed").exists()

    @override
    def mark_layer_completed(
        self, group: str, name: str, layer_name: str, group_idx: int = 0
    ) -> None:
        window_path = self.get_window_root(group, name)
        layer_dir = get_window_layer_dir(window_path, layer_name, group_idx)
        # We assume the directory exists because the layer should be materialized before
        # being marked completed.
        (layer_dir / "completed").touch()


class FileWindowStorageFactory(WindowStorageFactory):
    """Factory class for FileWindowStorage."""

    def get_storage(self, ds_path: UPath) -> FileWindowStorage:
        """Get a FileWindowStorage for the given dataset path."""
        return FileWindowStorage(ds_path)
