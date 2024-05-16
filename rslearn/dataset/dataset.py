"""rslearn dataset class."""

import json
import os
from typing import Optional

import rslearn.data_sources
from rslearn.config import TileStoreConfig, load_layer_config
from rslearn.data_sources import Item
from rslearn.tile_stores import PrefixedTileStore, TileStore, load_tile_store

from .materialize import Materializers
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

    def prepare(self) -> None:
        """Lookup data source items corresponding to windows in the dataset."""
        windows = self.load_windows()
        prepare_dataset_windows(self, windows)

    def get_tile_store(self) -> TileStore:
        """Get the tile store associated with this dataset.

        Returns:
            the TileStore
        """
        return load_tile_store(self.tile_store_config, self.ds_root)

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
    for layer_name, layer_cfg in dataset.layers.items():
        if not layer_cfg.data_source:
            continue

        data_source = rslearn.data_sources.data_source_from_config(
            layer_cfg, dataset.ds_root
        )

        # Get windows that need to be prepared for this layer.
        needed_windows = []
        for window in windows:
            layer_datas = window.load_layer_datas()
            if layer_name in layer_datas and not force:
                continue
            needed_windows.append(window)
        print(
            f"Preparing {len(needed_windows)} windows for layer {layer_name}"
        )

        # Get STGeometry for each window.
        geometries = []
        for window in needed_windows:
            geometry = window.get_geometry()

            # Apply temporal modifiers.
            time_offset = layer_cfg.data_source.time_offset
            if geometry.time_range and time_offset:
                geometry.time_range = (
                    geometry.time_range[0] + time_offset,
                    geometry.time_range[1] + time_offset,
                )
            duration = layer_cfg.data_source.duration
            if geometry.time_range and duration:
                geometry.time_range = (
                    geometry.time_range[0],
                    geometry.time_range[0] + duration,
                )

            geometries.append(geometry)

        results = data_source.get_items(geometries, layer_cfg.data_source.query_config)
        for window, result in zip(needed_windows, results):
            layer_datas = window.load_layer_datas()
            layer_datas[layer_name] = WindowLayerData(
                layer_name=layer_name,
                serialized_item_groups=[
                    [item.serialize() for item in group] for group in result
                ],
            )
            window.save_layer_datas(layer_datas)


def ingest_dataset_windows(dataset: Dataset, windows: list[Window]) -> None:
    """Ingest items for retrieved layers in a dataset.

    The items associated with the specified windows are downloaded and divided into
    tiles which are then added to the dataset's tile store.

    Args:
        dataset: the dataset
        windows: the windows to ingest
    """
    tile_store = dataset.get_tile_store()
    for layer_name, layer_cfg in dataset.layers.items():
        if not layer_cfg.data_source:
            continue

        data_source = rslearn.data_sources.data_source_from_config(
            layer_cfg, dataset.ds_root
        )

        geometries_by_item = {}
        for window in windows:
            layer_datas = window.load_layer_datas()
            if layer_name not in layer_datas:
                continue
            geometry = window.get_geometry()
            layer_data = layer_datas[layer_name]
            for group in layer_data.serialized_item_groups:
                for serialized_item in group:
                    item = data_source.deserialize_item(serialized_item)
                    if item not in geometries_by_item:
                        geometries_by_item[item] = []
                    geometries_by_item[item].append(geometry)

        print(
            f"Ingesting {len(geometries_by_item)} items in layer {layer_name}"
        )
        cur_tile_store = PrefixedTileStore(tile_store, (layer_name,))
        geometries_and_items = list(geometries_by_item.items())
        data_source.ingest(
            tile_store=cur_tile_store,
            items=[item for item, _ in geometries_and_items],
            geometries=[geometries for _, geometries in geometries_and_items],
        )


def is_window_ingested(dataset: Dataset, window: Window) -> bool:
    """Check if a window is ingested.

    Args:
        dataset: the dataset
        window: the window

    Returns:
        true if the window is ingested, false otherwise
    """
    tile_store = dataset.get_tile_store()
    layer_datas = window.load_layer_datas()
    for layer_name, layer_cfg in dataset.layers.items():
        if layer_name not in layer_datas:
            return False
        layer_data = layer_datas[layer_name]
        for group in layer_data.serialized_item_groups:
            for serialized_item in group:
                item = Item.deserialize(serialized_item)
                for band_set in layer_cfg.band_sets:
                    projection, _ = band_set.get_final_projection_and_bounds(
                        window.projection, window.bounds
                    )
                    layer_prefix = (
                        layer_name,
                        item.name,
                    )
                    # Make sure that layers exist containing each configured band.
                    # And that those layers are marked completed.
                    suffixes = tile_store.list_layers(layer_prefix)
                    needed_suffixes = []
                    needed_bands = {band for band in band_set.bands}
                    for suffix in suffixes:
                        cur_bands = suffix.split("_")
                        is_needed = False
                        for band in cur_bands:
                            if band in needed_bands:
                                is_needed = True
                                needed_bands.remove(band)
                        if not is_needed:
                            continue
                        needed_suffixes.append(suffix)
                    if len(needed_bands) > 0:
                        return False

                    for suffix in needed_suffixes:
                        layer_id = (
                            layer_name,
                            item.name,
                            suffix,
                            str(projection),
                        )
                        ts_layer = tile_store.get_layer(layer_id)
                        if not ts_layer:
                            return False
                        if not ts_layer.get_metadata().properties.get("completed"):
                            return False
    return True


def materialize_dataset_windows(dataset: Dataset, windows: list[Window]) -> None:
    """Materialize items for retrieved layers in a dataset.

    The portions of items corresponding to dataset windows are extracted from the tile
    store and written to the window directory.

    Args:
        dataset: the dataset
        windows: the windows to materialize
    """
    tile_store = dataset.get_tile_store()
    for layer_name, layer_cfg in dataset.layers.items():
        if not layer_cfg.data_source:
            continue

        for window in windows:
            if not is_window_ingested(dataset, window):
                print("not ingested")
                continue
            layer_datas = window.load_layer_datas()
            if layer_name not in layer_datas:
                print("not data")
                continue
            layer_data = layer_datas[layer_name]
            item_groups = []
            for serialized_group in layer_data.serialized_item_groups:
                item_group = []
                for serialized_item in serialized_group:
                    item = Item.deserialize(serialized_item)
                    item_group.append(item)
                item_groups.append(item_group)

            print(
                f"Materializing {len(item_groups)} item groups in layer {layer_name}"
            )

            if dataset.materializer_name:
                materializer = Materializers[dataset.materializer_name]
            else:
                materializer = Materializers[layer_cfg.layer_type.value]
            materializer.materialize(
                tile_store, window, layer_name, layer_cfg, item_groups
            )
