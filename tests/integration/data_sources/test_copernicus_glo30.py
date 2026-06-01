"""Mocked integration tests for the Copernicus GLO-30 data source."""

import pathlib

import numpy as np
import pytest
from pytest_httpserver import HTTPServer
from rasterio.crs import CRS
from upath import UPath

from rslearn.config import QueryConfig, SpaceMode
from rslearn.data_sources.copernicus_glo30 import CopernicusGLO30, _tile_name
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat


def _make_glo30_geotiff(
    path: pathlib.Path, lat: int = 47, lon: int = -123
) -> pathlib.Path:
    """Create a small GLO-30-like GeoTIFF covering one 1x1 degree cell.

    The DEM has a simple gradient (elevation increases northward) so that slope
    and aspect are nonzero and testable.
    """
    west, south, east, north = lon, lat, lon + 1, lat + 1
    width, height = 10, 10
    x_res = 0.1
    y_res = -0.1
    projection = Projection(CRS.from_epsg(4326), x_res, y_res)
    bounds = (
        int(west / x_res),
        int(north / y_res),
        int(east / x_res),
        int(south / y_res),
    )

    # Gradient: elevation increases from south to north (row 0 = north = highest).
    data = np.zeros((1, height, width), dtype=np.float32)
    for row in range(height):
        data[0, row, :] = (height - 1 - row) * 100.0
    data[0] += 500.0

    raster_dir = UPath(path / "glo30_raster")
    tile_name = _tile_name(lat, lon)
    GeotiffRasterFormat().encode_raster(
        raster_dir,
        projection,
        bounds,
        RasterArray(chw_array=data),
        fname=f"{tile_name}.tif",
    )
    return raster_dir / f"{tile_name}.tif"


def _setup_mock_server(
    httpserver: HTTPServer,
    tif_path: pathlib.Path,
    lat: int = 47,
    lon: int = -123,
) -> None:
    """Register a mock endpoint for a GLO-30 tile on the httpserver."""
    tile_name = _tile_name(lat, lon)
    url_path = f"/{tile_name}/{tile_name}.tif"

    with open(tif_path, "rb") as f:
        tif_data = f.read()
    httpserver.expect_request(url_path, method="GET").respond_with_data(
        tif_data, content_type="image/tiff"
    )


def test_ingest_elevation_only(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    httpserver: HTTPServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test ingesting elevation-only from GLO-30 with mocked HTTP server."""
    tif_path = _make_glo30_geotiff(tmp_path)
    _setup_mock_server(httpserver, tif_path)

    monkeypatch.setattr(CopernicusGLO30, "BASE_URL", httpserver.url_for("/"))

    ds = CopernicusGLO30()
    # Override band_names to elevation only for this test.
    ds.band_names = ["elevation"]
    ds._needs_slope = False
    ds._needs_aspect = False

    query_config = QueryConfig(space_mode=SpaceMode.MOSAIC, max_matches=1)
    item_groups = ds.get_items([seattle2020], query_config)[0]
    assert len(item_groups) == 1

    items = item_groups[0].items
    # Seattle (lon=-122.33, lat=47.61) falls in the N47/W123 tile.
    matching = [it for it in items if "N47" in it.name and "W123" in it.name]
    assert len(matching) >= 1
    item = matching[0]

    tile_store_dir = UPath(tmp_path / "tiles")
    tile_store = DefaultTileStore(str(tile_store_dir))
    tile_store.set_dataset_path(tile_store_dir)
    layer_name = "layer"

    ds.ingest(
        TileStoreWithLayer(tile_store, layer_name),
        [item],
        [[seattle2020]],
    )
    assert tile_store.is_raster_ready(layer_name, item, ["elevation"])


def test_ingest_with_slope_aspect(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    httpserver: HTTPServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test ingesting elevation + slope + aspect from GLO-30."""
    tif_path = _make_glo30_geotiff(tmp_path)
    _setup_mock_server(httpserver, tif_path)

    monkeypatch.setattr(CopernicusGLO30, "BASE_URL", httpserver.url_for("/"))

    ds = CopernicusGLO30()
    # Default band_names should include all three.
    assert ds.band_names == ["elevation", "slope", "aspect"]

    query_config = QueryConfig(space_mode=SpaceMode.MOSAIC, max_matches=1)
    item_groups = ds.get_items([seattle2020], query_config)[0]
    items = item_groups[0].items
    matching = [it for it in items if "N47" in it.name and "W123" in it.name]
    item = matching[0]

    tile_store_dir = UPath(tmp_path / "tiles")
    tile_store = DefaultTileStore(str(tile_store_dir))
    tile_store.set_dataset_path(tile_store_dir)
    layer_name = "layer"

    ds.ingest(
        TileStoreWithLayer(tile_store, layer_name),
        [item],
        [[seattle2020]],
    )
    assert tile_store.is_raster_ready(
        layer_name, item, ["elevation", "slope", "aspect"]
    )


def test_ingest_skips_404(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    httpserver: HTTPServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tiles that return 404 (e.g. ocean) should be skipped gracefully."""
    tile_name = _tile_name(47, -123)
    url_path = f"/{tile_name}/{tile_name}.tif"
    httpserver.expect_request(url_path, method="GET").respond_with_data(
        b"Not Found", status=404
    )

    monkeypatch.setattr(CopernicusGLO30, "BASE_URL", httpserver.url_for("/"))

    ds = CopernicusGLO30()
    ds.band_names = ["elevation"]
    ds._needs_slope = False
    ds._needs_aspect = False

    query_config = QueryConfig(space_mode=SpaceMode.MOSAIC, max_matches=1)
    item_groups = ds.get_items([seattle2020], query_config)[0]
    items = item_groups[0].items
    matching = [it for it in items if "N47" in it.name and "W123" in it.name]
    item = matching[0]

    tile_store_dir = UPath(tmp_path / "tiles")
    tile_store = DefaultTileStore(str(tile_store_dir))
    tile_store.set_dataset_path(tile_store_dir)
    layer_name = "layer"

    # Should not raise — just logs a warning and skips.
    ds.ingest(
        TileStoreWithLayer(tile_store, layer_name),
        [item],
        [[seattle2020]],
    )
    assert not tile_store.is_raster_ready(layer_name, item, ["elevation"])
