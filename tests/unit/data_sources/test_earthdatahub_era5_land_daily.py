from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import shapely
import xarray as xr
import zarr  # noqa: F401
from upath import UPath

from rslearn.config import QueryConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.earthdatahub import ERA5LandDailyUTCv1
from rslearn.data_sources.utils import MatchedItemGroup
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils.geometry import STGeometry


def test_era5land_dailyutc_v1_chunk_items_and_ingest(tmp_path: Path) -> None:
    """get_items returns chunk-level items and ingest writes multi-timestep rasters."""
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
    zarr_path = tmp_path / "era5_chunk.zarr"
    ds.to_zarr(
        zarr_path,
        mode="w",
        encoding={
            "t2m": {"chunks": (3, 2, 2)},
            "tp": {"chunks": (3, 2, 2)},
        },
    )

    data_source = ERA5LandDailyUTCv1(
        band_names=["t2m", "tp"],
        zarr_url=str(zarr_path),
        trust_env=False,
    )

    # Request only day 1 — but all 3 days sit in a single chunk.
    window_geom = STGeometry(
        WGS84_PROJECTION,
        shapely.box(0.0, 0.85, 0.2, 1.05),
        (datetime(2020, 1, 1, 0, tzinfo=UTC), datetime(2020, 1, 2, 0, tzinfo=UTC)),
    )
    query_config = QueryConfig(space_mode=SpaceMode.SINGLE_COMPOSITE)
    groups = data_source.get_items([window_geom], query_config=query_config)

    # 1 geometry -> 1 outer entry; SINGLE_COMPOSITE -> 1 group; 1 chunk -> 1 item.
    assert len(groups) == 1
    assert len(groups[0]) == 1
    assert isinstance(groups[0][0], MatchedItemGroup)
    items = groups[0][0]
    assert len(items) == 1
    assert items[0].name == "era5land_v1_t0_y0_x0"

    # Ingest the chunk.
    ds_path = UPath(tmp_path / "ds")
    tile_store = DefaultTileStore(convert_rasters_to_cogs=False)
    tile_store.set_dataset_path(ds_path)
    layer_tile_store = TileStoreWithLayer(tile_store, "era5")

    data_source.ingest(
        tile_store=layer_tile_store,
        items=items,
        geometries=[[window_geom] for _ in items],
    )

    bands = ["t2m", "tp"]
    assert layer_tile_store.is_raster_ready(items[0], bands)


def test_era5land_dailyutc_v1_requires_single_composite(tmp_path: Path) -> None:
    """get_items rejects non-SINGLE_COMPOSITE space modes."""
    valid_time = np.array(["2020-01-01"], dtype="datetime64[ns]")
    latitude = np.array([1.0, 0.9], dtype=np.float64)
    longitude = np.array([0.0, 0.1], dtype=np.float64)
    t2m = np.zeros((1, 2, 2), dtype=np.float32)

    ds = xr.Dataset(
        data_vars=dict(t2m=(("valid_time", "latitude", "longitude"), t2m)),
        coords=dict(valid_time=valid_time, latitude=latitude, longitude=longitude),
    )
    zarr_path = tmp_path / "era5_mode.zarr"
    ds.to_zarr(
        zarr_path,
        mode="w",
        encoding={"t2m": {"chunks": (1, 2, 2)}},
    )

    data_source = ERA5LandDailyUTCv1(
        band_names=["t2m"],
        zarr_url=str(zarr_path),
        trust_env=False,
    )

    window_geom = STGeometry(
        WGS84_PROJECTION,
        shapely.box(0.0, 0.85, 0.2, 1.05),
        (datetime(2020, 1, 1, 0, tzinfo=UTC), datetime(2020, 1, 2, 0, tzinfo=UTC)),
    )

    with pytest.raises(ValueError, match="SINGLE_COMPOSITE"):
        data_source.get_items(
            [window_geom], query_config=QueryConfig(space_mode=SpaceMode.MOSAIC)
        )


def test_era5land_dailyutc_v1_rejects_min_matches(tmp_path: Path) -> None:
    """get_items rejects min_matches for custom chunk matching."""
    valid_time = np.array(["2020-01-01"], dtype="datetime64[ns]")
    latitude = np.array([1.0, 0.9], dtype=np.float64)
    longitude = np.array([0.0, 0.1], dtype=np.float64)
    t2m = np.zeros((1, 2, 2), dtype=np.float32)

    ds = xr.Dataset(
        data_vars=dict(t2m=(("valid_time", "latitude", "longitude"), t2m)),
        coords=dict(valid_time=valid_time, latitude=latitude, longitude=longitude),
    )
    zarr_path = tmp_path / "era5_min_matches.zarr"
    ds.to_zarr(
        zarr_path,
        mode="w",
        encoding={"t2m": {"chunks": (1, 2, 2)}},
    )

    data_source = ERA5LandDailyUTCv1(
        band_names=["t2m"],
        zarr_url=str(zarr_path),
        trust_env=False,
    )

    window_geom = STGeometry(
        WGS84_PROJECTION,
        shapely.box(0.0, 0.85, 0.2, 1.05),
        (datetime(2020, 1, 1, 0, tzinfo=UTC), datetime(2020, 1, 2, 0, tzinfo=UTC)),
    )

    with pytest.raises(ValueError, match="min_matches"):
        data_source.get_items(
            [window_geom],
            query_config=QueryConfig(
                space_mode=SpaceMode.SINGLE_COMPOSITE, min_matches=1
            ),
        )


def test_era5land_dailyutc_v1_spatial_chunk_splitting(tmp_path: Path) -> None:
    """When the Zarr has spatial chunks smaller than the window, multiple items
    are returned — one per (time, lat, lon) chunk triple."""
    # 2 time steps, 4 lat, 4 lon — write with explicit chunks (2, 2, 2)
    valid_time = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    latitude = np.array([1.0, 0.9, 0.8, 0.7], dtype=np.float64)
    longitude = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64)

    t2m = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4) + 280

    ds = xr.Dataset(
        data_vars=dict(t2m=(("valid_time", "latitude", "longitude"), t2m)),
        coords=dict(valid_time=valid_time, latitude=latitude, longitude=longitude),
    )
    zarr_path = tmp_path / "era5_spatial.zarr"
    ds.to_zarr(
        zarr_path,
        mode="w",
        encoding={"t2m": {"chunks": (2, 2, 2)}},
    )

    data_source = ERA5LandDailyUTCv1(
        band_names=["t2m"],
        zarr_url=str(zarr_path),
        trust_env=False,
    )

    # Window covers all 4x4 grid cells, 2 days
    window_geom = STGeometry(
        WGS84_PROJECTION,
        shapely.box(-0.05, 0.65, 0.35, 1.05),
        (datetime(2020, 1, 1, 0, tzinfo=UTC), datetime(2020, 1, 3, 0, tzinfo=UTC)),
    )
    query_config = QueryConfig(space_mode=SpaceMode.SINGLE_COMPOSITE)
    groups = data_source.get_items([window_geom], query_config=query_config)

    assert len(groups) == 1
    assert len(groups[0]) == 1
    items = groups[0][0]

    # 1 time chunk x 2 lat chunks x 2 lon chunks = 4 items
    assert len(items) == 4
    item_names = sorted(i.name for i in items)
    assert item_names == [
        "era5land_v1_t0_y0_x0",
        "era5land_v1_t0_y0_x1",
        "era5land_v1_t0_y1_x0",
        "era5land_v1_t0_y1_x1",
    ]

    # Ingest all 4 items
    ds_path = UPath(tmp_path / "ds")
    tile_store = DefaultTileStore(convert_rasters_to_cogs=False)
    tile_store.set_dataset_path(ds_path)
    layer_tile_store = TileStoreWithLayer(tile_store, "era5")

    data_source.ingest(
        tile_store=layer_tile_store,
        items=items,
        geometries=[[window_geom] for _ in items],
    )

    for item in items:
        assert layer_tile_store.is_raster_ready(item, ["t2m"])


def test_era5land_dailyutc_v1_time_range_after_dataset(tmp_path: Path) -> None:
    """get_items returns no items when the time range is entirely after the dataset."""
    valid_time = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    latitude = np.array([1.0, 0.9], dtype=np.float64)
    longitude = np.array([0.0, 0.1], dtype=np.float64)
    t2m = np.zeros((2, 2, 2), dtype=np.float32)

    ds = xr.Dataset(
        data_vars=dict(t2m=(("valid_time", "latitude", "longitude"), t2m)),
        coords=dict(valid_time=valid_time, latitude=latitude, longitude=longitude),
    )
    zarr_path = tmp_path / "era5_time_after.zarr"
    ds.to_zarr(
        zarr_path,
        mode="w",
        encoding={"t2m": {"chunks": (2, 2, 2)}},
    )

    data_source = ERA5LandDailyUTCv1(
        band_names=["t2m"],
        zarr_url=str(zarr_path),
        trust_env=False,
    )

    # Time range entirely after the dataset (2099).
    window_geom = STGeometry(
        WGS84_PROJECTION,
        shapely.box(-0.05, 0.85, 0.15, 1.05),
        (datetime(2099, 1, 1, 0, tzinfo=UTC), datetime(2099, 2, 1, 0, tzinfo=UTC)),
    )
    query_config = QueryConfig(space_mode=SpaceMode.SINGLE_COMPOSITE)
    groups = data_source.get_items([window_geom], query_config=query_config)

    assert len(groups) == 1
    assert len(groups[0]) == 1
    assert len(groups[0][0]) == 0


def test_era5land_dailyutc_v1_time_range_before_dataset(tmp_path: Path) -> None:
    """get_items returns no items when the time range is entirely before the dataset."""
    valid_time = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    latitude = np.array([1.0, 0.9], dtype=np.float64)
    longitude = np.array([0.0, 0.1], dtype=np.float64)
    t2m = np.zeros((2, 2, 2), dtype=np.float32)

    ds = xr.Dataset(
        data_vars=dict(t2m=(("valid_time", "latitude", "longitude"), t2m)),
        coords=dict(valid_time=valid_time, latitude=latitude, longitude=longitude),
    )
    zarr_path = tmp_path / "era5_time_before.zarr"
    ds.to_zarr(
        zarr_path,
        mode="w",
        encoding={"t2m": {"chunks": (2, 2, 2)}},
    )

    data_source = ERA5LandDailyUTCv1(
        band_names=["t2m"],
        zarr_url=str(zarr_path),
        trust_env=False,
    )

    # Time range entirely before the dataset (1900).
    window_geom = STGeometry(
        WGS84_PROJECTION,
        shapely.box(-0.05, 0.85, 0.15, 1.05),
        (datetime(1900, 1, 1, 0, tzinfo=UTC), datetime(1900, 2, 1, 0, tzinfo=UTC)),
    )
    query_config = QueryConfig(space_mode=SpaceMode.SINGLE_COMPOSITE)
    groups = data_source.get_items([window_geom], query_config=query_config)

    assert len(groups) == 1
    assert len(groups[0]) == 1
    assert len(groups[0][0]) == 0
