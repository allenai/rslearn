"""Functions to manage datasets."""

import os
from typing import Optional

import rslearn.data_sources
from rslearn.config import LayerConfig
from rslearn.data_sources import DataSource, Item
from rslearn.tile_stores import PrefixedTileStore, TileStore

from .dataset import Dataset
from .materialize import Materializers
from .window import Window, WindowLayerData


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
        if not layer_cfg.data_source.ingest:
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


def is_window_ingested(dataset: Dataset, window: Window, check_layer_name: Optional[str] = None) -> bool:
    """Check if a window is ingested.

    Args:
        dataset: the dataset
        window: the window
        check_layer_name: optional layer name to only check that layer is ingested

    Returns:
        true if the window is ingested, false otherwise
    """
    tile_store = dataset.get_tile_store()
    layer_datas = window.load_layer_datas()
    for layer_name, layer_cfg in dataset.layers.items():
        if check_layer_name and check_layer_name != layer_name:
            continue
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
                        print(needed_bands, suffixes, layer_prefix)
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


def materialize_window(window: Window, dataset: Dataset, data_source: DataSource, tile_store: TileStore, layer_name: str, layer_cfg: LayerConfig) -> None:
    """Materialize a window.

    Args:
        window: the window
        dataset: the dataset
        data_source: the DataSource
        tile_store: tile store of the dataset to materialize from
        layer_name: the layer name
        layer_cfg: the layer config
    """
    # Check if layer is materialized already.
    layer_dir = os.path.join(window.window_root, "layers", layer_name)
    if os.path.exists(layer_dir):
        return

    layer_datas = window.load_layer_datas()
    if layer_name not in layer_datas:
        print("not data")
        return
    layer_data = layer_datas[layer_name]
    item_groups = []
    for serialized_group in layer_data.serialized_item_groups:
        item_group = []
        for serialized_item in serialized_group:
            item = Item.deserialize(serialized_item)
            item_group.append(item)
        item_groups.append(item_group)

    if layer_cfg.data_source.ingest:
        if not is_window_ingested(dataset, window, check_layer_name=layer_name):
            print("not ingested")
            return

        print(
            f"Materializing {len(item_groups)} item groups in layer {layer_name} from tile store"
        )

        if dataset.materializer_name:
            materializer = Materializers[dataset.materializer_name]
        else:
            materializer = Materializers[layer_cfg.layer_type.value]
        materializer.materialize(
            tile_store, window, layer_name, layer_cfg, item_groups
        )

    else:
        # This window is meant to be materialized directly from the data source.
        print(
            f"Materializing {len(item_groups)} item groups in layer {layer_name} via data source"
        )
        data_source.materialize(window, item_groups, layer_name, layer_cfg)


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

        data_source = rslearn.data_sources.data_source_from_config(
            layer_cfg, dataset.ds_root
        )

        for window in windows:
            materialize_window(window, dataset, data_source, tile_store, layer_name, layer_cfg)
