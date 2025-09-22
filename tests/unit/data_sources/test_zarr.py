"""Tests for the Zarr data source."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest
import shapely.geometry
from rasterio.crs import CRS
from upath import UPath

from rslearn.config.dataset import RasterLayerConfig
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils.geometry import Projection, STGeometry


@pytest.fixture()
def sample_zarr_store(tmp_path):
    """Create a small Zarr cube suitable for testing."""

    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")

    times = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]")
    bands = ["B02", "B03", "B04"]
    y = np.arange(4)
    x = np.arange(4)

    data = np.zeros((len(times), len(bands), len(y), len(x)), dtype=np.float32)
    for ti in range(len(times)):
        for bi in range(len(bands)):
            data[ti, bi, :, :] = ti * 10.0 + float(bi)

    array = xr.DataArray(
        data,
        coords={"time": times, "band": bands, "y": y, "x": x},
        dims=("time", "band", "y", "x"),
        name="reflectance",
    )
    dataset = array.to_dataset()
    store_path = tmp_path / "cube.zarr"
    dataset.to_zarr(store_path, mode="w")
    return store_path


def build_layer_config(store_path) -> RasterLayerConfig:
    """Construct a raster layer configuration for the test cube."""

    config_dict = {
        "type": "raster",
        "band_sets": [
            {
                "dtype": "float32",
                "bands": ["B02", "B03", "B04"],
                "nodata_vals": [0.0, 0.0, 0.0],
            }
        ],
        "data_source": {
            "name": "rslearn.data_sources.zarr.ZarrDataSource",
            "store_uri": str(store_path),
            "data_variable": "reflectance",
            "crs": "EPSG:32633",
            "pixel_size": 1,
            "origin": [0.0, 0.0],
            "axis_names": {
                "x": "x",
                "y": "y",
                "time": "time",
                "band": "band",
            },
            "bands": ["B02", "B03", "B04"],
            "dtype": "float32",
            "nodata": 0.0,
            "chunk_shape": {"y": 4, "x": 4},
        },
    }
    return RasterLayerConfig.from_config(config_dict)


def test_zarr_ingest_and_read(tmp_path, sample_zarr_store):
    """Zarr data source should ingest chunks and support direct reads."""

    layer_cfg = build_layer_config(sample_zarr_store)

    from rslearn.data_sources.zarr import ZarrDataSource

    data_source = ZarrDataSource.from_config(
        layer_cfg, UPath(tmp_path / "dataset")
    )

    projection = Projection(CRS.from_epsg(32633), 1, -1)
    window_time = (
        datetime(2024, 1, 1),
        datetime(2024, 1, 1) + timedelta(hours=1),
    )
    window_geom = STGeometry(
        projection,
        shapely.geometry.box(0, 0, 4, 4),
        window_time,
    )

    groups = data_source.get_items(
        [window_geom], layer_cfg.data_source.query_config
    )
    assert len(groups) == 1
    assert len(groups[0]) == 1
    assert len(groups[0][0]) == 1
    item = groups[0][0][0]

    # Ensure serialization round-trips.
    reconstructed = data_source.deserialize_item(item.serialize())
    assert reconstructed.pixel_bounds == item.pixel_bounds

    layer_name = "zarr_layer"
    tile_store = DefaultTileStore()
    tile_store.set_dataset_path(UPath(tmp_path / "tiles"))

    data_source.ingest(
        TileStoreWithLayer(tile_store, layer_name),
        [item],
        [[window_geom]],
    )

    bands = ["B02", "B03", "B04"]
    assert tile_store.is_raster_ready(layer_name, item.name, bands)
    ingested = tile_store.read_raster(
        layer_name,
        item.name,
        bands,
        data_source.projection,
        item.pixel_bounds,
    )
    assert ingested.shape == (3, 4, 4)
    assert np.allclose(ingested[0], 0.0)
    assert np.allclose(ingested[1], 1.0)
    assert np.allclose(ingested[2], 2.0)

    # Direct reads without ingestion should also succeed on partial bounds.
    partial_bounds = (
        item.pixel_bounds[0],
        item.pixel_bounds[1],
        item.pixel_bounds[0] + 2,
        item.pixel_bounds[1] + 2,
    )
    direct = data_source.read_raster(
        layer_name,
        item.name,
        bands,
        data_source.projection,
        partial_bounds,
    )
    assert direct.shape == (3, 2, 2)
    assert np.allclose(direct[0], 0.0)
    assert np.allclose(direct[1], 1.0)
    assert np.allclose(direct[2], 2.0)

    # Tile store metadata helpers should reflect the item structure.
    assert data_source.is_raster_ready(layer_name, item.name, bands)
    assert data_source.get_raster_bands(layer_name, item.name) == [bands]
    assert (
        data_source.get_raster_bounds(layer_name, item.name, bands, data_source.projection)
        == item.pixel_bounds
    )
