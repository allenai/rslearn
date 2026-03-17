import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rasterio
import shapely
import xarray as xr
import zarr  # noqa: F401
from upath import UPath

from rslearn.config import QueryConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.earthdatahub import ERA5LandDailyUTCv1
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils.geometry import STGeometry
from rslearn.utils.raster_format import get_bandset_dirname


def test_era5land_dailyutc_v1_ingest_from_local_zarr(tmp_path: Path) -> None:
    valid_time = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    latitude = np.array([1.0, 0.9], dtype=np.float64)  # descending, 0.1 deg spacing
    longitude = np.array([0.0, 0.1], dtype=np.float64)  # 0..360 convention

    t2m = np.array(
        [
            [[280.0, 281.0], [282.0, 283.0]],
            [[284.0, 285.0], [286.0, 287.0]],
        ],
        dtype=np.float32,
    )
    tp = np.array(
        [
            [[0.001, 0.002], [0.003, 0.004]],
            [[0.005, 0.006], [0.007, 0.008]],
        ],
        dtype=np.float32,
    )

    ds = xr.Dataset(
        data_vars=dict(
            t2m=(("valid_time", "latitude", "longitude"), t2m),
            tp=(("valid_time", "latitude", "longitude"), tp),
        ),
        coords=dict(valid_time=valid_time, latitude=latitude, longitude=longitude),
    )
    zarr_path = tmp_path / "era5_land_daily.zarr"
    ds.to_zarr(
        zarr_path,
        mode="w",
        encoding={
            "t2m": {"chunks": (2, 2, 2)},
            "tp": {"chunks": (2, 2, 2)},
        },
    )

    data_source = ERA5LandDailyUTCv1(
        band_names=["t2m", "tp"],
        zarr_url=str(zarr_path),
        trust_env=False,
    )

    window_geom = STGeometry(
        WGS84_PROJECTION,
        shapely.box(0.0, 0.85, 0.2, 1.05),
        (datetime(2020, 1, 1, 12, tzinfo=UTC), datetime(2020, 1, 3, 0, tzinfo=UTC)),
    )

    query_config = QueryConfig(space_mode=SpaceMode.SINGLE_COMPOSITE)
    groups = data_source.get_items([window_geom], query_config=query_config)

    # Both days are in 1 chunk -> 1 group with 1 item.
    assert len(groups) == 1
    assert len(groups[0]) == 1
    items = groups[0][0]
    assert len(items) == 1

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

    # Read the GeoTIFF directly to verify shape and values.
    raster_dir = ds_path / "tiles" / "era5" / items[0].name / get_bandset_dirname(bands)
    tif_files = [p for p in raster_dir.iterdir() if p.suffix == ".tif"]
    assert len(tif_files) == 1

    with rasterio.open(tif_files[0]) as src:
        out = src.read()
    # Shape is (C*T, H, W) = (2*2, 2, 2) = (4, 2, 2).
    assert out.shape == (4, 2, 2)

    # Verify metadata.json sidecar.
    metadata = json.loads((raster_dir / "metadata.json").read_text())
    assert metadata["num_channels"] == 2
    assert metadata["num_timesteps"] == 2
    assert len(metadata["timestamps"]) == 2

    # Reshape to (C, T, H, W) and verify values.
    array = out.reshape(2, 2, 2, 2)
    np.testing.assert_allclose(array[0], t2m)
    np.testing.assert_allclose(array[1], tp)


def test_era5land_dailyutc_v1_ingest_negative_longitude_bounds(tmp_path: Path) -> None:
    valid_time = np.array(["2024-06-01", "2024-06-02"], dtype="datetime64[ns]")
    latitude = np.array([51.6, 51.5], dtype=np.float64)  # descending, 0.1 deg spacing
    longitude = np.array(
        [359.8, 359.9], dtype=np.float64
    )  # 0..360 convention near Greenwich

    t2m = np.array(
        [
            [[280.0, 281.0], [282.0, 283.0]],
            [[284.0, 285.0], [286.0, 287.0]],
        ],
        dtype=np.float32,
    )
    tp = np.array(
        [
            [[0.001, 0.002], [0.003, 0.004]],
            [[0.005, 0.006], [0.007, 0.008]],
        ],
        dtype=np.float32,
    )

    ds = xr.Dataset(
        data_vars=dict(
            t2m=(("valid_time", "latitude", "longitude"), t2m),
            tp=(("valid_time", "latitude", "longitude"), tp),
        ),
        coords=dict(valid_time=valid_time, latitude=latitude, longitude=longitude),
    )
    zarr_path = tmp_path / "era5_land_daily_359.zarr"
    ds.to_zarr(
        zarr_path,
        mode="w",
        encoding={
            "t2m": {"chunks": (2, 2, 2)},
            "tp": {"chunks": (2, 2, 2)},
        },
    )

    data_source = ERA5LandDailyUTCv1(
        band_names=["t2m", "tp"],
        zarr_url=str(zarr_path),
        trust_env=False,
    )

    # Bounds around London (negative longitude) that should map to 359.x degrees.
    window_geom = STGeometry(
        WGS84_PROJECTION,
        shapely.box(-0.15, 51.49, -0.10, 51.52),
        (
            datetime(2024, 6, 1, 0, 0, tzinfo=UTC),
            datetime(2024, 6, 3, 0, 0, tzinfo=UTC),
        ),
    )

    query_config = QueryConfig(space_mode=SpaceMode.SINGLE_COMPOSITE)
    groups = data_source.get_items([window_geom], query_config=query_config)
    assert len(groups) == 1
    assert len(groups[0]) == 1
    items = groups[0][0]
    assert len(items) == 1

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

    # Read the GeoTIFF directly to verify shape and values.
    raster_dir = ds_path / "tiles" / "era5" / items[0].name / get_bandset_dirname(bands)
    tif_files = [p for p in raster_dir.iterdir() if p.suffix == ".tif"]
    assert len(tif_files) == 1

    with rasterio.open(tif_files[0]) as src:
        out = src.read()
    assert out.shape == (4, 2, 2)

    # Verify metadata.json sidecar.
    metadata = json.loads((raster_dir / "metadata.json").read_text())
    assert metadata["num_channels"] == 2
    assert metadata["num_timesteps"] == 2

    # Reshape and verify values.
    array = out.reshape(2, 2, 2, 2)
    np.testing.assert_allclose(array[0], t2m)
    np.testing.assert_allclose(array[1], tp)


def test_era5land_dailyutc_v1_spatial_chunk_ingest(tmp_path: Path) -> None:
    """When the Zarr has explicit spatial chunks, each item maps to one chunk
    and ingest writes a correctly-sized multi-timestep GeoTIFF for each."""
    valid_time = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    latitude = np.array([1.0, 0.9, 0.8, 0.7], dtype=np.float64)
    longitude = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64)

    t2m = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4) + 280

    ds = xr.Dataset(
        data_vars=dict(t2m=(("valid_time", "latitude", "longitude"), t2m)),
        coords=dict(valid_time=valid_time, latitude=latitude, longitude=longitude),
    )
    zarr_path = tmp_path / "era5_spatial_chunks.zarr"
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

    window_geom = STGeometry(
        WGS84_PROJECTION,
        shapely.box(-0.05, 0.65, 0.35, 1.05),
        (datetime(2020, 1, 1, 0, tzinfo=UTC), datetime(2020, 1, 3, 0, tzinfo=UTC)),
    )
    query_config = QueryConfig(space_mode=SpaceMode.SINGLE_COMPOSITE)
    groups = data_source.get_items([window_geom], query_config=query_config)

    items = groups[0][0]
    assert len(items) == 4  # 1 time x 2 lat x 2 lon

    ds_path = UPath(tmp_path / "ds")
    tile_store = DefaultTileStore(convert_rasters_to_cogs=False)
    tile_store.set_dataset_path(ds_path)
    layer_tile_store = TileStoreWithLayer(tile_store, "era5")

    data_source.ingest(
        tile_store=layer_tile_store,
        items=items,
        geometries=[[window_geom] for _ in items],
    )

    # Each item should be stored as a (1*2, 2, 2) = (2, 2, 2) GeoTIFF
    for item in items:
        assert layer_tile_store.is_raster_ready(item, ["t2m"])
        raster_dir = (
            ds_path / "tiles" / "era5" / item.name / get_bandset_dirname(["t2m"])
        )
        tif_files = [p for p in raster_dir.iterdir() if p.suffix == ".tif"]
        assert len(tif_files) == 1

        with rasterio.open(tif_files[0]) as src:
            out = src.read()
        # (C*T, H, W) = (1*2, 2, 2)
        assert out.shape == (2, 2, 2)

        metadata = json.loads((raster_dir / "metadata.json").read_text())
        assert metadata["num_channels"] == 1
        assert metadata["num_timesteps"] == 2
