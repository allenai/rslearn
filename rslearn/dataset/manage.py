"""Functions to manage datasets."""

import random
import time
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import rslearn.data_sources
from rslearn.config import (
    LayerConfig,
    LayerType,
    RasterLayerConfig,
)
from rslearn.data_sources import DataSource, Item
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStore, get_tile_store_with_layer

from .dataset import Dataset
from .materialize import Materializers
from .window import Window, WindowLayerData

logger = get_logger(__name__)


def retry(fn: Callable, retry_max_attempts: int, retry_backoff: timedelta) -> Any:
    """Retry the function multiple times in case of error.

    The function is retried until either the attempts are exhausted, or the function
    runs successfully without raising an Exception.

    Args:
        fn: the function to call.
        retry_max_attempts: retry this many times (plus the original attempt) before
            giving up (and raising Exception).
        retry_backoff: the base backoff time used to compute how long to wait between
            retries. The actual time is (retry_backoff * attempts) * r, where r is a
            random number between 1 and 2, and attempts is the number of attempts tried
            so far.
    """
    for attempt_idx in range(retry_max_attempts):
        try:
            return fn()
        except Exception as e:
            logger.warning(f"Retrying after catching error in retry loop: {e}")
            sleep_base_seconds = retry_backoff.total_seconds() * (attempt_idx + 1)
            time.sleep(sleep_base_seconds * (1 + random.random()))

    # Last attempt. This time we don't catch the exception.
    return fn()


def prepare_dataset_windows(
    dataset: Dataset,
    windows: list[Window],
    force: bool = False,
    retry_max_attempts: int = 0,
    retry_backoff: timedelta = timedelta(minutes=1),
) -> None:
    """Prepare windows in a dataset.

    Preparing a window involves looking up items corresponding to the window in each of
    the retrieved layers specified in the dataset.

    Args:
        dataset: the dataset
        windows: the windows to prepare
        force: whether to prepare windows even if they were previously prepared
            (default false)
        retry_max_attempts: set greater than zero to retry for this many attempts in
            case of error.
        retry_backoff: how long to wait before retrying (see retry).
    """
    # Iterate over retrieved layers, and prepare each one.
    for layer_name, layer_cfg in dataset.layers.items():
        if not layer_cfg.data_source:
            continue
        data_source_cfg = layer_cfg.data_source

        # Get windows that need to be prepared for this layer.
        needed_windows = []
        for window in windows:
            layer_datas = window.load_layer_datas()
            if layer_name in layer_datas and not force:
                continue
            needed_windows.append(window)
        logger.info(f"Preparing {len(needed_windows)} windows for layer {layer_name}")
        if len(needed_windows) == 0:
            continue

        # Create data source after checking for at least one window so it can be fast
        # if there are no windows to prepare.
        data_source = rslearn.data_sources.data_source_from_config(
            layer_cfg, dataset.path
        )

        # Get STGeometry for each window.
        geometries = []
        for window in needed_windows:
            geometry = window.get_geometry()

            # Apply temporal modifiers.
            time_offset = data_source_cfg.time_offset
            if geometry.time_range and time_offset:
                geometry.time_range = (
                    geometry.time_range[0] + time_offset,
                    geometry.time_range[1] + time_offset,
                )
            duration = data_source_cfg.duration
            if geometry.time_range and duration:
                geometry.time_range = (
                    geometry.time_range[0],
                    geometry.time_range[0] + duration,
                )

            geometries.append(geometry)

        results = retry(
            fn=lambda: data_source.get_items(geometries, data_source_cfg.query_config),
            retry_max_attempts=retry_max_attempts,
            retry_backoff=retry_backoff,
        )

        for window, result in zip(needed_windows, results):
            layer_datas = window.load_layer_datas()
            layer_datas[layer_name] = WindowLayerData(
                layer_name=layer_name,
                serialized_item_groups=[
                    [item.serialize() for item in group] for group in result
                ],
            )
            window.save_layer_datas(layer_datas)


def ingest_dataset_windows(
    dataset: Dataset,
    windows: list[Window],
    retry_max_attempts: int = 0,
    retry_backoff: timedelta = timedelta(minutes=1),
) -> None:
    """Ingest items for retrieved layers in a dataset.

    The items associated with the specified windows are downloaded and divided into
    tiles which are then added to the dataset's tile store.

    Args:
        dataset: the dataset
        windows: the windows to ingest
        retry_max_attempts: set greater than zero to retry for this many attempts in
            case of error.
        retry_backoff: how long to wait before retrying (see retry).
    """
    tile_store = dataset.get_tile_store()
    for layer_name, layer_cfg in dataset.layers.items():
        if not layer_cfg.data_source:
            continue
        if not layer_cfg.data_source.ingest:
            continue

        data_source = rslearn.data_sources.data_source_from_config(
            layer_cfg, dataset.path
        )

        geometries_by_item: dict = {}
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

        print(f"Ingesting {len(geometries_by_item)} items in layer {layer_name}")
        geometries_and_items = list(geometries_by_item.items())

        # Use retry loop for the actual data source ingest call.
        def ingest() -> None:
            data_source.ingest(
                tile_store=get_tile_store_with_layer(tile_store, layer_name, layer_cfg),
                items=[item for item, _ in geometries_and_items],
                geometries=[geometries for _, geometries in geometries_and_items],
            )

        retry(
            fn=ingest,
            retry_max_attempts=retry_max_attempts,
            retry_backoff=retry_backoff,
        )


def is_window_ingested(
    dataset: Dataset, window: Window, check_layer_name: str | None = None
) -> bool:
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

        layer_tile_store = get_tile_store_with_layer(tile_store, layer_name, layer_cfg)

        layer_data = layer_datas[layer_name]
        for group in layer_data.serialized_item_groups:
            for serialized_item in group:
                item = Item.deserialize(serialized_item)

                if layer_cfg.layer_type == LayerType.RASTER:
                    assert isinstance(layer_cfg, RasterLayerConfig)
                    for band_set in layer_cfg.band_sets:
                        # Make sure that layers exist containing each configured band.
                        # And that those layers are marked completed.
                        available_bands = layer_tile_store.get_raster_bands(item.name)
                        wanted_bands = {band for band in band_set.bands}
                        for cur_bands in available_bands:
                            is_needed = False
                            for band in cur_bands:
                                if band in wanted_bands:
                                    is_needed = True
                                    wanted_bands.remove(band)
                            if not is_needed:
                                continue
                        if len(wanted_bands) > 0:
                            return False

    return True


def materialize_window(
    window: Window,
    dataset: Dataset,
    data_source: DataSource,
    tile_store: TileStore,
    layer_name: str,
    layer_cfg: LayerConfig,
    retry_max_attempts: int = 0,
    retry_backoff: timedelta = timedelta(minutes=1),
) -> None:
    """Materialize a window.

    Args:
        window: the window
        dataset: the dataset
        data_source: the DataSource
        tile_store: tile store of the dataset to materialize from
        layer_name: the layer name
        layer_cfg: the layer config
        retry_max_attempts: set greater than zero to retry for this many attempts in
            case of error.
        retry_backoff: how long to wait before retrying (see retry).
    """
    # Check if layer is materialized already.
    if window.is_layer_completed(layer_name):
        return

    layer_datas = window.load_layer_datas()
    if layer_name not in layer_datas:
        logger.info(
            "Not materializing layer %s in window %s because it is not prepared",
            layer_name,
            window.name,
        )
        return
    layer_data = layer_datas[layer_name]
    item_groups = []
    for serialized_group in layer_data.serialized_item_groups:
        item_group = []
        for serialized_item in serialized_group:
            item = data_source.deserialize_item(serialized_item)
            item_group.append(item)
        item_groups.append(item_group)

    if layer_cfg.data_source is None:
        raise ValueError("data_source is required")
    if layer_cfg.data_source.ingest:
        if not is_window_ingested(dataset, window, check_layer_name=layer_name):
            logger.info(
                "Not materializing layer %s in window %s because it is not ingested",
                layer_name,
                window.name,
            )
            return

        print(
            f"Materializing {len(item_groups)} item groups in layer {layer_name} from tile store"
        )

        if dataset.materializer_name:
            materializer = Materializers[dataset.materializer_name]
        else:
            materializer = Materializers[layer_cfg.layer_type.value]

        retry(
            fn=lambda: materializer.materialize(
                get_tile_store_with_layer(tile_store, layer_name, layer_cfg),
                window,
                layer_name,
                layer_cfg,
                item_groups,
            ),
            retry_max_attempts=retry_max_attempts,
            retry_backoff=retry_backoff,
        )

    else:
        # This window is meant to be materialized directly from the data source.
        print(
            f"Materializing {len(item_groups)} item groups in layer {layer_name} via data source"
        )
        retry(
            fn=lambda: data_source.materialize(
                window, item_groups, layer_name, layer_cfg
            ),
            retry_max_attempts=retry_max_attempts,
            retry_backoff=retry_backoff,
        )


def materialize_dataset_windows(
    dataset: Dataset,
    windows: list[Window],
    retry_max_attempts: int = 0,
    retry_backoff: timedelta = timedelta(minutes=1),
) -> None:
    """Materialize items for retrieved layers in a dataset.

    The portions of items corresponding to dataset windows are extracted from the tile
    store and written to the window directory.

    Args:
        dataset: the dataset
        windows: the windows to materialize
        retry_max_attempts: set greater than zero to retry for this many attempts in
            case of error.
        retry_backoff: how long to wait before retrying (see retry).
    """
    tile_store = dataset.get_tile_store()
    for layer_name, layer_cfg in dataset.layers.items():
        if not layer_cfg.data_source:
            continue

        data_source = rslearn.data_sources.data_source_from_config(
            layer_cfg, dataset.path
        )

        for window in windows:
            materialize_window(
                window=window,
                dataset=dataset,
                data_source=data_source,
                tile_store=tile_store,
                layer_name=layer_name,
                layer_cfg=layer_cfg,
                retry_max_attempts=retry_max_attempts,
                retry_backoff=retry_backoff,
            )
