"""Mocked integration tests for the XyzTiles data source."""

import io
import pathlib
from datetime import UTC, datetime

import numpy as np
import pytest
import shapely
from PIL import Image
from pytest_httpserver import HTTPServer
from upath import UPath

from rslearn.config import (
    BandSetConfig,
    DType,
    LayerConfig,
    LayerType,
    QueryConfig,
    SpaceMode,
)
from rslearn.data_sources.xyz_tiles import XyzTiles
from rslearn.dataset import Window
from rslearn.dataset.storage.file import FileWindowStorage
from rslearn.utils.geometry import STGeometry
from rslearn.utils.raster_format import GeotiffRasterFormat

# We use zoom=1 which gives a 2x2 grid of 256x256 tiles (512x512 total pixels).
ZOOM = 1
TILE_SIZE = 256
TILE_COL = 0
TILE_ROW = 0
PIXEL_VALUE = 128

TIME_RANGE = (datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC))


@pytest.fixture
def tile_server(httpserver: HTTPServer) -> str:
    """Serve a single 256x256 RGB tile at zoom=1, col=0, row=0.

    Returns the URL template with {z}/{x}/{y} placeholders.
    """
    arr = np.ones((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8) * PIXEL_VALUE
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    tile_data = buf.getvalue()

    httpserver.expect_request(
        f"/{ZOOM}/{TILE_COL}/{TILE_ROW}.png", method="GET"
    ).respond_with_data(tile_data, content_type="image/png")

    return httpserver.url_for("/") + "{z}/{x}/{y}.png"


def test_materialize(
    tmp_path: pathlib.Path,
    tile_server: str,
) -> None:
    """Test materialize reading a 64x64 subset fully inside the served tile."""
    data_source = XyzTiles(
        url_templates=[tile_server],
        time_ranges=[TIME_RANGE],
        zoom=ZOOM,
    )

    # Read a 64x64 region inside tile (0, 0).
    # Tile (0, 0) covers offset pixels [0, 256). Data source coords are shifted
    # by -pixel_offset, so we add that back to place ourselves inside the tile.
    proj = data_source.projection
    x0 = -data_source.pixel_offset + 32
    y0 = -data_source.pixel_offset + 32
    bounds = (x0, y0, x0 + 64, y0 + 64)

    geometry = STGeometry(proj, shapely.box(*bounds), TIME_RANGE)
    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([geometry], query_config)[0]
    assert len(item_groups) > 0

    layer_config = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[BandSetConfig(dtype=DType.UINT8, bands=["R", "G", "B"])],
    )
    window = Window(
        storage=FileWindowStorage(UPath(tmp_path / "rslearn_dataset")),
        group="default",
        name="default",
        projection=proj,
        bounds=bounds,
        time_range=TIME_RANGE,
    )
    window.save()

    data_source.materialize(window, item_groups, "raster", layer_config)
    raster_dir = window.get_raster_dir("raster", ["R", "G", "B"])
    assert (raster_dir / "geotiff.tif").exists()

    array = GeotiffRasterFormat().decode_raster(raster_dir, proj, bounds)
    assert array.shape == (3, 64, 64)
    assert np.all(array == PIXEL_VALUE)
