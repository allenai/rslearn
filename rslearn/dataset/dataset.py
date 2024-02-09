import json
import os
from typing import Optional

import pytimeparse

import rslearn.data_sources
from rslearn.data_sources import QueryConfig, SpaceMode, TimeMode
from rslearn.tile_stores import FileTileStore, TileStore

from .window import Window, WindowLayerData


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
        with open(os.path.join(ds_root, "config.json"), "r") as f:
            self.config = json.load(f)

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

    def prepare(self) -> None:
        """Lookup data source items corresponding to windows in the dataset."""
        windows = self.load_windows()
        prepare_dataset_windows(self, windows)

    def get_tile_store(self) -> TileStore:
        # For now, only support FileTileStore.
        return FileTileStore(os.path.join(self.ds_root, "tiles"))

    def ingest(self) -> None:
        """Ingests items for retrieved layers."""
        windows = self.load_windows()
        ingest_dataset_windows(self, windows)


def prepare_dataset_windows(
    dataset: Dataset, windows: list[Window], force: bool = False
) -> None:
    """Prepare windows in a dataset.

    Preparing a window involves looking up items corresponding to the window in each of
    the retrieved layers specified in the dataset.

    Args:
        dataset: the dataset
        windows: the windows to prepare
        force: whether to prepare windows even if they were previously prepared
            (default false)
    """
    # Iterate over retrieved layers, and prepare each one.
    for layer_name, layer_cfg in dataset.config.get("layers", {}).items():
        if "data_source" not in layer_cfg:
            continue

        data_source = rslearn.data_sources.load_data_source(
            layer_cfg["data_source"]["name"],
            layer_cfg["data_source"]["args"],
        )

        # Get windows that need to be prepared for this layer.
        needed_windows = []
        for window in windows:
            if layer_name in window.layer_datas and not force:
                continue
        needed_windows.append(window)
        print(
            "Preparing {} windows for layer {}".format(len(needed_windows), layer_name)
        )

        # Get STGeometry for each window.
        geometries = []
        for window in needed_windows:
            geometry = window.get_geometry()

            if "time_offset" in layer_cfg:
                time_offset = pytimeparse.parse(layer_cfg["time_offset"])
                if geometry.time_range:
                    geometry.time_range = (
                        geometry.time_range[0] + time_offset,
                        geometry.time_range[1] + time_offset,
                    )

            geometries.append(geometry)

        # Create QueryConfig.
        query_config_dict = layer_cfg["data_source"].get("query_config", {})
        query_config = QueryConfig(
            space_mode=SpaceMode[query_config_dict.get("space_mode", "MOSAIC")],
            time_mode=TimeMode[query_config_dict.get("time_mode", "WITHIN")],
            max_matches=query_config_dict.get("max_matches", 1),
        )

        results = data_source.get_items(geometries, query_config)
        for window, result in zip(needed_windows, results):
            window.layer_datas[layer_name] = WindowLayerData(
                layer_name=layer_name,
                serialized_item_groups=[
                    [item.serialize() for item in group] for group in result
                ],
            )
            window.save(ds_root=dataset.ds_root)


def ingest_dataset_windows(dataset: Dataset, windows: list[Window]) -> None:
    """Ingest items for retrieved layers in a dataset.

    The items associated with the specified windows are downloaded and divided into
    tiles which are then added to the dataset's tile store.

    Args:
        dataset: the dataset
        windows: the windows to ingest
    """
    tile_store = dataset.get_tile_store()
    for layer_name, layer_cfg in dataset.config.get("layers", {}).items():
        if "data_source" not in layer_cfg:
            continue

        data_source = rslearn.data_sources.load_data_source(
            layer_cfg["data_source"]["name"],
            layer_cfg["data_source"]["args"],
        )

        geometries_by_item = {}
        for window in windows:
            if layer_name not in window.layer_datas:
                continue
            geometry = window.get_geometry()
            layer_data = window.layer_datas[layer_name]
            for group in layer_data.serialized_item_groups:
                for serialized_item in group:
                    item = data_source.deserialize_item(serialized_item)
                    if item not in geometries_by_item:
                        geometries_by_item[item] = []
                    geometries_by_item[item].append(geometry)

        print(
            "Ingesting {} items in layer {}".format(len(geometries_by_item), layer_name)
        )
        geometries_and_items = list(geometries_by_item.items())
        data_source.ingest(
            tile_store=tile_store,
            items=[item for item, _ in geometries_and_items],
            geometries=[geometries for _, geometries in geometries_and_items],
        )
