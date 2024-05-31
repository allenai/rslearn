"""rslearn dataset class."""

import json
import os
from typing import Optional

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

    def __init__(self, ds_root: str) -> None:
        """Initializes a new Dataset.

        Args:
            ds_root: the root directory of the dataset
        """
        self.ds_root = ds_root

        # Load dataset configuration.
        with open(os.path.join(ds_root, "config.json")) as f:
            config = json.load(f)
            self.layers = {
                layer_name: load_layer_config(d)
                for layer_name, d in config["layers"].items()
            }
            self.tile_store_config = TileStoreConfig.from_config(config["tile_store"])
            self.materializer_name = config.get("materialize")

    def load_windows(
        self, groups: Optional[list[str]] = None, names: Optional[list[str]] = None
    ) -> list[Window]:
        """Load the windows in the dataset.

        Args:
            groups: an optional list of groups to filter loading
            names: an optional list of window names to filter loading
        """
        windows = []
        if not groups:
            groups = os.listdir(os.path.join(self.ds_root, "windows"))
        for group in groups:
            group_dir = os.path.join(self.ds_root, "windows", group)
            if names:
                cur_names = names
            else:
                cur_names = os.listdir(group_dir)

            for window_name in cur_names:
                window_dir = os.path.join(group_dir, window_name)
                window = Window.load(window_dir)
                windows.append(window)

        return windows

    def get_tile_store(self) -> TileStore:
        """Get the tile store associated with this dataset.

        Returns:
            the TileStore
        """
        return load_tile_store(self.tile_store_config, self.ds_root)
