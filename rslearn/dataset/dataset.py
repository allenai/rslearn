"""rslearn dataset class."""

import json
import multiprocessing

import tqdm
from upath import UPath

from rslearn.config import TileStoreConfig, load_layer_config
from rslearn.tile_stores import TileStore, load_tile_store

from .window import Window


class Dataset:
    """A rslearn dataset.

    Datasets are stored in a directory with the following structure:

    .. code-block:: none

        dataset/
            config.json
            windows/
                group1/
                    epsg:3857_10_623565_1528020/
                        metadata.json
                        layers/
                            sentinel2/
                                0_0_tci.tif
                            label/
                                0_0_tci.json
                    ...
                ...

    The dataset loads its configuration and supports actions like prepare, ingest, and
    materialize.
    """

    def __init__(self, path: UPath) -> None:
        """Initializes a new Dataset.

        Args:
            path: the root directory of the dataset
        """
        self.path = path

        # Load dataset configuration.
        with (self.path / "config.json").open("r") as f:
            config = json.load(f)
            self.layers = {
                layer_name: load_layer_config(d)
                for layer_name, d in config["layers"].items()
            }
            self.tile_store_config = TileStoreConfig.from_config(config["tile_store"])
            self.materializer_name = config.get("materialize")

    def load_windows(
        self,
        groups: list[str] | None = None,
        names: list[str] | None = None,
        show_progress: bool = False,
        workers: int = 0,
    ) -> list[Window]:
        """Load the windows in the dataset.

        Args:
            groups: an optional list of groups to filter loading
            names: an optional list of window names to filter loading
            show_progress: whether to show tqdm progress bar
            workers: number of parallel workers, default 0 (use main thread only to load windows)
        """
        window_dirs = []
        if not groups:
            groups = []
            for p in (self.path / "windows").iterdir():
                groups.append(p.name)
        for group in groups:
            group_dir = self.path / "windows" / group
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
            windows = [Window.load(window_dir) for window_dir in window_dirs]
        else:
            p = multiprocessing.Pool(workers)
            outputs = p.imap_unordered(Window.load, window_dirs)
            if show_progress:
                outputs = tqdm.tqdm(
                    outputs, total=len(window_dirs), desc="Loading windows"
                )
            windows = []
            for window in outputs:
                windows.append(window)
            p.close()

        return windows

    def get_tile_store(self) -> TileStore:
        """Get the tile store associated with this dataset.

        Returns:
            the TileStore
        """
        return load_tile_store(self.tile_store_config, self.path)
