"""Functions to manage datasets."""

import random
import time
from collections.abc import Callable
from datetime import timedelta
from typing import Any

from rslearn.config import (
    LayerConfig,
    LayerType,
)
from rslearn.data_sources import DataSource, Item
from rslearn.dataset.handler_summaries import (
    LayerPrepareSummary,
    MaterializeDatasetWindowsSummary,
    MaterializeWindowLayersSummary,
    MaterializeWindowLayerSummary,
    PrepareDatasetWindowsSummary,
)
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStore, get_tile_store_with_layer

from .dataset import Dataset
from .materialize import Materializer, RasterMaterializer, VectorMaterializer
from .window import Window, WindowLayerData

logger = get_logger(__name__)


class AttemptsCounter:
    """A simple counter for tracking attempts (including initial attempt and retries)."""

    def __init__(self) -> None:
        """Initialize counter with value 0."""
        self.value = 0

    def increment(self) -> None:
        """Increment the counter by 1."""
        self.value += 1


def retry(
    fn: Callable,
    retry_max_attempts: int,
    retry_backoff: timedelta,
    attempts_counter: AttemptsCounter | None = None,
) -> Any:
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
        attempts_counter: an optional counter to increment for each attempt
    """
    for attempt_idx in range(retry_max_attempts):
        if attempts_counter:
            attempts_counter.increment()
        try:
            return fn()
        except Exception as e:
            logger.warning(f"Retrying after catching error in retry loop: {e}")
            sleep_base_seconds = retry_backoff.total_seconds() * (attempt_idx + 1)
            time.sleep(sleep_base_seconds * (1 + random.random()))

    # Last attempt. This time we don't catch the exception.
    if attempts_counter:
        attempts_counter.increment()
    return fn()


def prepare_dataset_windows(
    dataset: Dataset,
    windows: list[Window],
    force: bool = False,
    ignore_errors: bool = False,
    retry_max_attempts: int = 0,
    retry_backoff: timedelta = timedelta(minutes=1),
) -> PrepareDatasetWindowsSummary:
    """Prepare windows in a dataset.

    Preparing a window involves looking up items corresponding to the window in each of
    the retrieved layers specified in the dataset.

    Args:
        dataset: the dataset
        windows: the windows to prepare
        force: whether to prepare windows even if they were previously prepared
            (default false)
        ignore_errors: if True, catch errors and continue to next layer instead of
            raising. Errors are tracked in the returned summary.
        retry_max_attempts: set greater than zero to retry for this many attempts in
            case of error.
        retry_backoff: how long to wait before retrying (see retry).

    Returns:
        a summary of the prepare operation, fit for telemetry purposes
    """
    start_time = time.monotonic()
    layer_summaries: dict[str, LayerPrepareSummary] = {}

    # Iterate over retrieved layers, and prepare each one.
    for layer_name, layer_cfg in dataset.layers.items():
        layer_start_time = time.monotonic()

        if not layer_cfg.data_source:
            continue
        data_source_cfg = layer_cfg.data_source
        min_matches = data_source_cfg.query_config.min_matches

        # Get windows that need to be prepared for this layer.
        # Also track which windows are skipped vs previously rejected.
        needed_windows = []
        windows_skipped = 0
        windows_rejected = 0
        for window in windows:
            layer_datas = window.load_layer_datas()
            if layer_name in layer_datas and not force:
                # Window already has layer data - check if it was previously rejected
                layer_data = layer_datas[layer_name]
                if len(layer_data.serialized_item_groups) == 0 and min_matches > 0:
                    # Previously rejected due to min_matches
                    windows_rejected += 1
                else:
                    # Successfully prepared previously
                    windows_skipped += 1
                continue
            needed_windows.append(window)
        logger.info(f"Preparing {len(needed_windows)} windows for layer {layer_name}")

        if len(needed_windows) == 0:
            layer_summaries[layer_name] = LayerPrepareSummary(
                layer_name=layer_name,
                data_source_name=data_source_cfg.class_path,
                duration_seconds=time.monotonic() - layer_start_time,
                windows_prepared=0,
                windows_skipped=windows_skipped,
                windows_rejected=windows_rejected,
                get_items_attempts=0,
            )
            continue

        # Create data source after checking for at least one window so it can be fast
        # if there are no windows to prepare.
        data_source = layer_cfg.instantiate_data_source(dataset.path)

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

        attempts_counter = AttemptsCounter()
        try:
            results = retry(
                fn=lambda: data_source.get_items(
                    geometries, data_source_cfg.query_config
                ),
                retry_max_attempts=retry_max_attempts,
                retry_backoff=retry_backoff,
                attempts_counter=attempts_counter,
            )

            windows_prepared = 0
            for window, result in zip(needed_windows, results):
                layer_datas = window.load_layer_datas()
                layer_datas[layer_name] = WindowLayerData(
                    layer_name=layer_name,
                    serialized_item_groups=[
                        [item.serialize() for item in group] for group in result
                    ],
                )
                window.save_layer_datas(layer_datas)

                # If result is empty and min_matches > 0, window was rejected due to min_matches
                if len(result) == 0 and min_matches > 0:
                    windows_rejected += 1
                else:
                    windows_prepared += 1

            layer_summaries[layer_name] = LayerPrepareSummary(
                layer_name=layer_name,
                data_source_name=data_source_cfg.class_path,
                duration_seconds=time.monotonic() - layer_start_time,
                windows_prepared=windows_prepared,
                windows_skipped=windows_skipped,
                windows_rejected=windows_rejected,
                get_items_attempts=attempts_counter.value,
            )
        except Exception as e:
            if not ignore_errors:
                raise
            logger.exception(
                f"Error preparing layer {layer_name} for {len(needed_windows)} windows"
            )
            layer_summaries[layer_name] = LayerPrepareSummary(
                layer_name=layer_name,
                data_source_name=data_source_cfg.class_path,
                duration_seconds=time.monotonic() - layer_start_time,
                windows_prepared=0,
                windows_skipped=windows_skipped,
                windows_rejected=windows_rejected,
                get_items_attempts=attempts_counter.value,
                windows_failed=len(needed_windows),
                error_messages=[str(e)],
            )

    summary = PrepareDatasetWindowsSummary(
        duration_seconds=time.monotonic() - start_time,
        total_windows_requested=len(windows),
        layer_summaries=layer_summaries,
    )

    return summary


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

        data_source = layer_cfg.instantiate_data_source(dataset.path)

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

                if layer_cfg.type == LayerType.RASTER:
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
) -> MaterializeWindowLayerSummary:
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

    Returns:
        a summary of the materialize operation, fit for telemetry purposes
    """
    # Check if layer is materialized already.
    if window.is_layer_completed(layer_name):
        return MaterializeWindowLayerSummary(
            skipped=True,
            materialize_attempts=0,
        )

    layer_datas = window.load_layer_datas()
    if layer_name not in layer_datas:
        logger.info(
            "Not materializing layer %s in window %s because it is not prepared",
            layer_name,
            window.name,
        )
        return MaterializeWindowLayerSummary(
            skipped=True,
            materialize_attempts=0,
        )

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

    attempts_counter = AttemptsCounter()
    if layer_cfg.data_source.ingest:
        if not is_window_ingested(dataset, window, check_layer_name=layer_name):
            logger.info(
                "Not materializing layer %s in window %s because it is not ingested",
                layer_name,
                window.name,
            )
            return MaterializeWindowLayerSummary(
                skipped=True,
                materialize_attempts=0,
            )

        logger.info(
            f"Materializing {len(item_groups)} item groups in layer {layer_name} from tile store"
        )

        materializer: Materializer
        if layer_cfg.type == LayerType.RASTER:
            materializer = RasterMaterializer()
        elif layer_cfg.type == LayerType.VECTOR:
            materializer = VectorMaterializer()
        else:
            raise ValueError(f"unknown layer type {layer_cfg.type}")

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
            attempts_counter=attempts_counter,
        )

    else:
        # This window is meant to be materialized directly from the data source.
        logger.info(
            f"Materializing {len(item_groups)} item groups in layer {layer_name} via data source"
        )
        retry(
            fn=lambda: data_source.materialize(
                window, item_groups, layer_name, layer_cfg
            ),
            retry_max_attempts=retry_max_attempts,
            retry_backoff=retry_backoff,
            attempts_counter=attempts_counter,
        )

    return MaterializeWindowLayerSummary(
        skipped=False,
        materialize_attempts=attempts_counter.value,
    )


def materialize_dataset_windows(
    dataset: Dataset,
    windows: list[Window],
    ignore_errors: bool = False,
    retry_max_attempts: int = 0,
    retry_backoff: timedelta = timedelta(minutes=1),
) -> MaterializeDatasetWindowsSummary:
    """Materialize items for retrieved layers in a dataset.

    The portions of items corresponding to dataset windows are extracted from the tile
    store and written to the window directory.

    Args:
        dataset: the dataset
        windows: the windows to materialize
        ignore_errors: if True, catch errors per-window and continue. Errors are
            tracked in the returned summary.
        retry_max_attempts: set greater than zero to retry for this many attempts in
            case of error.
        retry_backoff: how long to wait before retrying (see retry).

    Returns:
        a summary of the materialize operation, fit for telemetry purposes
    """
    start_time = time.monotonic()

    layer_summaries: dict[str, MaterializeWindowLayersSummary] = {}

    tile_store = dataset.get_tile_store()
    for layer_name, layer_cfg in dataset.layers.items():
        layer_start_time = time.monotonic()

        total_materialize_attempts = 0
        total_skipped = 0
        windows_failed = 0
        error_messages: list[str] = []
        data_source_name = "N/A"

        if not layer_cfg.data_source:
            total_skipped = len(windows)
        else:
            data_source_name = layer_cfg.data_source.class_path
            data_source = layer_cfg.instantiate_data_source(dataset.path)

            for window in windows:
                try:
                    window_summary = materialize_window(
                        window=window,
                        dataset=dataset,
                        data_source=data_source,
                        tile_store=tile_store,
                        layer_name=layer_name,
                        layer_cfg=layer_cfg,
                        retry_max_attempts=retry_max_attempts,
                        retry_backoff=retry_backoff,
                    )
                    total_materialize_attempts += window_summary.materialize_attempts
                    if window_summary.skipped:
                        total_skipped += 1
                except Exception as e:
                    if not ignore_errors:
                        raise
                    logger.exception(
                        f"Error materializing window {window.name} layer {layer_name}"
                    )
                    windows_failed += 1
                    error_messages.append(str(e))

        layer_summaries[layer_name] = MaterializeWindowLayersSummary(
            layer_name=layer_name,
            data_source_name=data_source_name,
            duration_seconds=time.monotonic() - layer_start_time,
            total_windows_requested=len(windows),
            num_windows_materialized=len(windows) - total_skipped - windows_failed,
            materialize_attempts=total_materialize_attempts,
            windows_failed=windows_failed,
            error_messages=error_messages,
        )

    return MaterializeDatasetWindowsSummary(
        duration_seconds=time.monotonic() - start_time,
        total_windows_requested=len(windows),
        layer_summaries=layer_summaries,
    )
