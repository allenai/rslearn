from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import shapely


def test_era5land_dailyutc_v1_chunk_prefetch(tmp_path: Path) -> None:
    """Requesting a single day should prefetch and write all days in the chunk."""
    import xarray as xr
    import zarr  # noqa: F401
    from upath import UPath

    from rslearn.config import QueryConfig
    from rslearn.const import WGS84_PROJECTION
    from rslearn.data_sources.earthdatahub import ERA5LandDailyUTCv1
    from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
    from rslearn.utils.geometry import STGeometry

    valid_time = np.array(
        ["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[ns]"
    )
    latitude = np.array([1.0, 0.9], dtype=np.float64)
    longitude = np.array([0.0, 0.1], dtype=np.float64)

    t2m = np.arange(3 * 2 * 2, dtype=np.float32).reshape(3, 2, 2) + 280
    tp = np.arange(3 * 2 * 2, dtype=np.float32).reshape(3, 2, 2) * 0.001

    ds = xr.Dataset(
        data_vars=dict(
            t2m=(("valid_time", "latitude", "longitude"), t2m),
            tp=(("valid_time", "latitude", "longitude"), tp),
        ),
        coords=dict(valid_time=valid_time, latitude=latitude, longitude=longitude),
    )
    zarr_path = tmp_path / "era5_prefetch.zarr"
    ds.to_zarr(zarr_path, mode="w")

    data_source = ERA5LandDailyUTCv1(
        band_names=["t2m", "tp"],
        zarr_url=str(zarr_path),
        trust_env=False,
    )

    # Request only day 1.
    window_geom = STGeometry(
        WGS84_PROJECTION,
        shapely.box(0.0, 0.85, 0.2, 1.05),
        (datetime(2020, 1, 1, 0, tzinfo=UTC), datetime(2020, 1, 2, 0, tzinfo=UTC)),
    )
    groups = data_source.get_items([window_geom], query_config=QueryConfig())
    items = [group[0] for group in groups[0]]
    assert len(items) == 1  # Only day 1 was requested.

    ds_path = UPath(tmp_path / "ds")
    tile_store = DefaultTileStore(convert_rasters_to_cogs=False)
    tile_store.set_dataset_path(ds_path)
    layer_tile_store = TileStoreWithLayer(tile_store, "era5")

    data_source.ingest(
        tile_store=layer_tile_store,
        items=items,
        geometries=[[window_geom] for _ in items],
    )

    # All 3 days in the chunk should have been written, not just day 1.
    bands = ["t2m", "tp"]
    for day_idx, day_str in enumerate(["20200101", "20200102", "20200103"]):
        item_name = f"era5land_dailyutc_v1_{day_str}"
        assert layer_tile_store.is_raster_ready(item_name, bands), (
            f"Expected {item_name} to be written by chunk prefetch"
        )


def test_era5land_dailyutc_v1_lock_files_cleaned_up(tmp_path: Path) -> None:
    """Lock files should be removed after ingestion completes."""
    import xarray as xr
    import zarr  # noqa: F401
    from upath import UPath

    from rslearn.config import QueryConfig
    from rslearn.const import WGS84_PROJECTION
    from rslearn.data_sources.data_source import DataSourceContext
    from rslearn.data_sources.earthdatahub import ERA5LandDailyUTCv1
    from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
    from rslearn.utils.geometry import STGeometry

    valid_time = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    latitude = np.array([1.0, 0.9], dtype=np.float64)
    longitude = np.array([0.0, 0.1], dtype=np.float64)

    t2m = np.array(
        [[[280.0, 281.0], [282.0, 283.0]], [[284.0, 285.0], [286.0, 287.0]]],
        dtype=np.float32,
    )

    ds = xr.Dataset(
        data_vars=dict(t2m=(("valid_time", "latitude", "longitude"), t2m)),
        coords=dict(valid_time=valid_time, latitude=latitude, longitude=longitude),
    )
    zarr_path = tmp_path / "era5_locks.zarr"
    ds.to_zarr(zarr_path, mode="w")

    ds_path = UPath(tmp_path / "ds")
    context = DataSourceContext(ds_path=ds_path)

    data_source = ERA5LandDailyUTCv1(
        band_names=["t2m"],
        zarr_url=str(zarr_path),
        trust_env=False,
        context=context,
    )

    window_geom = STGeometry(
        WGS84_PROJECTION,
        shapely.box(0.0, 0.85, 0.2, 1.05),
        (datetime(2020, 1, 1, 0, tzinfo=UTC), datetime(2020, 1, 3, 0, tzinfo=UTC)),
    )
    groups = data_source.get_items([window_geom], query_config=QueryConfig())
    items = [group[0] for group in groups[0]]

    tile_store = DefaultTileStore(convert_rasters_to_cogs=False)
    tile_store.set_dataset_path(ds_path)
    layer_tile_store = TileStoreWithLayer(tile_store, "era5")

    data_source.ingest(
        tile_store=layer_tile_store,
        items=items,
        geometries=[[window_geom] for _ in items],
    )

    # Lock directory should exist but contain no lock files.
    lock_dir = ds_path / ".era5_chunk_locks"
    if lock_dir.exists():
        lock_files = [f for f in lock_dir.iterdir() if f.name.endswith(".lock")]
        assert len(lock_files) == 0, f"Stale lock files found: {lock_files}"
