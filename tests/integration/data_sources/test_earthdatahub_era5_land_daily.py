import numpy as np
import pytest
import rasterio
import shapely
from datetime import UTC, datetime


def test_era5land_dailyutc_v1_ingest_from_local_zarr(tmp_path) -> None:
    import xarray as xr
    import zarr  # noqa: F401

    from rslearn.const import WGS84_PROJECTION
    from rslearn.data_sources.earthdatahub import ERA5LandDailyUTCv1
    from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
    from rslearn.utils.geometry import STGeometry
    from upath import UPath

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
    ds.to_zarr(zarr_path, mode="w")

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
    from rslearn.config import QueryConfig

    groups = data_source.get_items([window_geom], query_config=QueryConfig())
    items = [group[0] for group in groups[0]]

    ds_path = UPath(tmp_path / "ds")
    tile_store = DefaultTileStore(convert_rasters_to_cogs=False)
    tile_store.set_dataset_path(ds_path)
    layer_tile_store = TileStoreWithLayer(tile_store, "era5")

    data_source.ingest(
        tile_store=layer_tile_store,
        items=items,
        geometries=[[window_geom] for _ in items],
    )

    from rslearn.utils.raster_format import get_bandset_dirname

    bands = ["t2m", "tp"]
    for idx, item in enumerate(items):
        raster_dir = ds_path / "tiles" / "era5" / item.name / get_bandset_dirname(bands)
        raster_fname = None
        for p in raster_dir.iterdir():
            if p.name in {"completed", "bands.json"}:
                continue
            if ".tmp." in p.name:
                continue
            raster_fname = p
            break
        assert raster_fname is not None

        with rasterio.open(raster_fname) as src:
            out = src.read()
        assert out.shape == (2, 2, 2)
        assert np.allclose(out[0], t2m[idx])
        assert np.allclose(out[1], tp[idx])


def test_era5land_dailyutc_v1_ingest_negative_longitude_bounds(tmp_path) -> None:
    import xarray as xr
    import zarr  # noqa: F401

    from rslearn.data_sources.earthdatahub import ERA5LandDailyUTCv1
    from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
    from rslearn.utils.geometry import STGeometry
    from rslearn.config import QueryConfig
    from rslearn.const import WGS84_PROJECTION
    from upath import UPath

    valid_time = np.array(["2024-06-01", "2024-06-02"], dtype="datetime64[ns]")
    latitude = np.array([51.6, 51.5], dtype=np.float64)  # descending, 0.1 deg spacing
    longitude = np.array([359.8, 359.9], dtype=np.float64)  # 0..360 convention near Greenwich

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
    ds.to_zarr(zarr_path, mode="w")

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

    groups = data_source.get_items([window_geom], query_config=QueryConfig())
    items = [group[0] for group in groups[0]]

    ds_path = UPath(tmp_path / "ds")
    tile_store = DefaultTileStore(convert_rasters_to_cogs=False)
    tile_store.set_dataset_path(ds_path)
    layer_tile_store = TileStoreWithLayer(tile_store, "era5")

    data_source.ingest(
        tile_store=layer_tile_store,
        items=items,
        geometries=[[window_geom] for _ in items],
    )

    from rslearn.utils.raster_format import get_bandset_dirname

    bands = ["t2m", "tp"]
    for idx, item in enumerate(items):
        raster_dir = ds_path / "tiles" / "era5" / item.name / get_bandset_dirname(bands)
        raster_fname = None
        for p in raster_dir.iterdir():
            if p.name in {"completed", "bands.json"}:
                continue
            if ".tmp." in p.name:
                continue
            raster_fname = p
            break
        assert raster_fname is not None

        with rasterio.open(raster_fname) as src:
            out = src.read()
        assert out.shape == (2, 2, 2)
        assert np.allclose(out[0], t2m[idx])
        assert np.allclose(out[1], tp[idx])


def test_era5land_dailyutc_v1_bounds_cross_dateline_error_message(tmp_path) -> None:
    from rslearn.data_sources.earthdatahub import ERA5LandDailyUTCv1

    with pytest.raises(ValueError, match=r"does not yet support .* cross the dateline"):
        ERA5LandDailyUTCv1(
            band_names=["t2m"],
            zarr_url=str(tmp_path / "unused.zarr"),
            bounds=[170.0, 0.0, -170.0, 1.0],
            trust_env=False,
        )
