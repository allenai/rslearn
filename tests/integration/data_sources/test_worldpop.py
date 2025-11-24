import pathlib

import numpy as np
import pytest
import shapely
from pytest_httpserver import HTTPServer
from upath import UPath

from rslearn.config import (
    QueryConfig,
    SpaceMode,
)
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.worldpop import WorldPop
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.raster_format import GeotiffRasterFormat

# WorldPop is based on LocalFiles so it names the bands in order from B1, B2, ...
# But WorldPop just has single band so it is always B1.
TEST_BAND = "B1"

# Degrees per pixel to use in the GeoTIFF.
# This roughly corresponds to 10 m/pixel.
DEGREES_PER_PIXEL = 0.0001

# Size of the GeoTIFF.
SIZE = 16

# See tests/integration/fixtures/geometries/seattle2020/seattle2020.py.
SEATTLE_POINT = shapely.Point(-122.33, 47.61)

INDEX_HTML = """
<html>
<body>
<a href="usa/">usa</a>
</body>
</html>
"""

COUNTRY_HTML = """
<html>
<body>
<a href="usa_ppp_2020_constrained.tif">download</a>
</body>
</html>
"""


def make_test_tif(tmp_path: pathlib.Path) -> pathlib.Path:
    """Make a sample GeoTIFF file similar to the WorldPop ones.

    It will just be a 16x16 GeoTIFF corresponding to the seattle2020 test geometry so
    that it can be used to test the data source.

    Args:
        tmp_path: temporary directory that will be used to store the GeoTIFF.

    Returns:
        the GeoTIFF filename.
    """
    # Make the GeoTIFF 16x16 centered at the same point as seattle2020.
    src_geom = STGeometry(WGS84_PROJECTION, SEATTLE_POINT, None)
    projection = Projection(WGS84_PROJECTION.crs, DEGREES_PER_PIXEL, -DEGREES_PER_PIXEL)
    dst_geom = src_geom.to_projection(projection)
    bounds = (
        int(dst_geom.shp.x) - SIZE // 2,
        int(dst_geom.shp.y) - SIZE // 2,
        int(dst_geom.shp.x) + SIZE // 2,
        int(dst_geom.shp.y) + SIZE // 2,
    )
    array = np.ones((1, SIZE, SIZE), dtype=np.uint8)
    raster_path = UPath(tmp_path)
    raster_format = GeotiffRasterFormat()
    raster_format.encode_raster(raster_path, projection, bounds, array)

    return tmp_path / raster_format.fname


def test_worldpop(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    httpserver: HTTPServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests ingesting the example data corresponding to seattle2020.

    Args:
        tmp_path: temporary path for making zip file and for tile store.
        seattle2020: the geometry to use for prepare.
        httpserver: server for serving the example data.
        monkeypatch: monkey patch instance.
    """

    # The WorldPop data is large so we use test data instead. We need to start a test
    # server that serves the data.
    tif_fname = make_test_tif(tmp_path)
    with tif_fname.open("rb") as f:
        tif_data = f.read()
    httpserver.expect_request("/", method="GET").respond_with_data(
        INDEX_HTML, content_type="text/html"
    )
    httpserver.expect_request("/usa/", method="GET").respond_with_data(
        COUNTRY_HTML, content_type="text/html"
    )
    httpserver.expect_request(
        "/usa/usa_ppp_2020_constrained.tif", method="GET"
    ).respond_with_data(tif_data, content_type="image/tiff")

    # Initialize the WorldPop instance.
    # But we need to customize the URL that it retrieves from.
    monkeypatch.setattr(WorldPop, "INDEX_URLS", [httpserver.url_for("/")])
    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    data_source = WorldPop(
        worldpop_dir=UPath(tmp_path) / "worldpop",
    )

    print("get items")
    item_groups = data_source.get_items([seattle2020], query_config)[0]
    item = item_groups[0][0]

    tile_store_dir = UPath(tmp_path) / "tile_store"
    tile_store = DefaultTileStore(str(tile_store_dir))
    tile_store.set_dataset_path(tile_store_dir)

    print("ingest")
    layer_name = "layer"
    data_source.ingest(
        TileStoreWithLayer(tile_store, layer_name), item_groups[0], [[seattle2020]]
    )
    assert tile_store.is_raster_ready(layer_name, item.name, [TEST_BAND])

    # Double check that the data intersected our example GeoTIFF and isn't just all 0.
    bounds = (
        int(seattle2020.shp.bounds[0]),
        int(seattle2020.shp.bounds[1]),
        int(seattle2020.shp.bounds[2]),
        int(seattle2020.shp.bounds[3]),
    )
    raster_data = tile_store.read_raster(
        layer_name, item.name, [TEST_BAND], seattle2020.projection, bounds
    )
    assert raster_data.max() == 1
