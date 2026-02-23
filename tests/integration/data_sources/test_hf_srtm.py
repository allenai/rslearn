"""Mocked integration tests for the HuggingFace SRTM data source."""

import pathlib

import numpy as np
import pytest
from pytest_httpserver import HTTPServer
from rasterio.crs import CRS
from upath import UPath

from rslearn.config import QueryConfig, SpaceMode
from rslearn.data_sources.hf_srtm import SRTM
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat


def _make_srtm_geotiff(path: pathlib.Path) -> pathlib.Path:
    """Create a small SRTM-like GeoTIFF covering the Seattle area tile (N47W123)."""
    # SRTM tiles are 1x1 degree. With 10 pixels per degree, pixel coords = CRS * 10.
    west, south, east, north = -123, 47, -122, 48
    width, height = 10, 10
    x_res = 0.1  # (east - west) / width
    y_res = -0.1  # -(north - south) / height
    projection = Projection(CRS.from_epsg(4326), x_res, y_res)
    # pixel coords = CRS coords / resolution
    bounds = (
        int(west / x_res),  # -1230
        int(north / y_res),  # -480
        int(east / x_res),  # -1220
        int(south / y_res),  # -470
    )
    data = np.ones((1, height, width), dtype=np.int16) * 100
    raster_dir = UPath(path / "srtm_raster")
    GeotiffRasterFormat().encode_raster(
        raster_dir,
        projection,
        bounds,
        RasterArray(chw_array=data),
        fname="SRTM1N47W123V2.tif",
    )
    return raster_dir / "SRTM1N47W123V2.tif"


def test_srtm_ingest(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    httpserver: HTTPServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test SRTM ingest with mocked HuggingFace HTTP server."""
    # Create the file list and test GeoTIFF.
    file_list = ["N47/SRTM1N47W123V2.tif"]
    httpserver.expect_request("/file_list.json", method="GET").respond_with_json(
        file_list
    )

    tif_path = _make_srtm_geotiff(tmp_path)
    with open(tif_path, "rb") as f:
        tif_data = f.read()
    httpserver.expect_request(
        "/N47/SRTM1N47W123V2.tif", method="GET"
    ).respond_with_data(tif_data, content_type="image/tiff")

    monkeypatch.setattr(SRTM, "BASE_URL", httpserver.url_for("/"))

    data_source = SRTM()
    query_config = QueryConfig(space_mode=SpaceMode.MOSAIC, max_matches=1)
    item_groups = data_source.get_items([seattle2020], query_config)[0]
    assert len(item_groups) == 1 and len(item_groups[0]) == 1
    item = item_groups[0][0]

    assert item.name == "N47/SRTM1N47W123V2.tif"

    tile_store_dir = UPath(tmp_path / "tiles")
    tile_store = DefaultTileStore(str(tile_store_dir))
    tile_store.set_dataset_path(tile_store_dir)
    layer_name = "layer"

    data_source.ingest(
        TileStoreWithLayer(tile_store, layer_name), item_groups[0], [[seattle2020]]
    )
    assert tile_store.is_raster_ready(layer_name, item.name, ["dem"])


def test_srtm_cache_dir(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    httpserver: HTTPServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that SRTM file list cache works -- second get_items should work offline."""
    file_list = ["N47/SRTM1N47W123V2.tif"]
    httpserver.expect_request("/file_list.json", method="GET").respond_with_json(
        file_list
    )

    monkeypatch.setattr(SRTM, "BASE_URL", httpserver.url_for("/"))

    cache_dir = tmp_path / "cache"
    data_source = SRTM(cache_dir=str(cache_dir))
    query_config = QueryConfig(space_mode=SpaceMode.MOSAIC, max_matches=1)

    # First call fetches from the server and caches.
    item_groups = data_source.get_items([seattle2020], query_config)[0]
    assert len(item_groups) == 1 and len(item_groups[0]) == 1
    assert (cache_dir / SRTM.FILE_LIST_FILENAME).exists()

    # Stop the server so any HTTP request would fail.
    httpserver.clear()

    # Second instantiation with same cache_dir should work without the server.
    data_source2 = SRTM(cache_dir=str(cache_dir))
    item_groups2 = data_source2.get_items([seattle2020], query_config)[0]
    assert len(item_groups2) == 1 and len(item_groups2[0]) == 1
    assert item_groups2[0][0].name == item_groups[0][0].name
