import pathlib
import zipfile
from typing import Any

import numpy as np
import shapely
from pytest_httpserver import HTTPServer
from upath import UPath

from rslearn.config import (
    BandSetConfig,
    DType,
    LayerType,
    QueryConfig,
    RasterLayerConfig,
    SpaceMode,
)
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.worldcover import WorldCover
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.raster_format import GeotiffRasterFormat

# WorldCover is based on LocalFiles so it names the bands in order from B1, B2, ...
# But WorldCover just has single band so it is always B1.
TEST_BAND = "B1"

# Degrees per pixel to use in the GeoTIFF.
# This roughly corresponds to 10 m/pixel.
DEGREES_PER_PIXEL = 0.0001

# Size of the GeoTIFF.
SIZE = 16

# See tests/integration/fixtures/geometries/seattle2020/seattle2020.py.
SEATTLE_POINT = shapely.Point(-122.33, 47.61)


def make_test_zip(tmp_path: pathlib.Path) -> pathlib.Path:
    """Make a sample zip file similar to the ESA WorldCover 2021 ones.

    Example URL:
    https://worldcover2021.esa.int/data/archive/ESA_WorldCover_10m_2021_v200_60deg_macrotile_N30E000.zip

    Our zip file will just contain a single 16x16 GeoTIFF. We make sure it corresponds
    the seattle2020 test geometry so that it can be used to test the data source.

    Args:
        tmp_path: temporary directory that will be used to store the GeoTIFF and zip
            file.

    Returns:
        the filename of the zip file
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
    array = np.zeros((1, SIZE, SIZE), dtype=np.uint8)
    raster_path = UPath(tmp_path)
    raster_format = GeotiffRasterFormat()
    raster_format.encode_raster(raster_path, projection, bounds, array)

    # Create a zip file containing it.
    zip_fname = tmp_path / "data.zip"
    zipf = zipfile.ZipFile(zip_fname, "w")
    zipf.write(tmp_path / raster_format.fname)
    zipf.close()

    return zip_fname


def test_ingest_seattle(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    httpserver: HTTPServer,
    monkeypatch: Any,
) -> None:
    """Test ingesting an item corresponding to seattle2020 to local filesystem."""

    # The WorldCover data is large so we use test data instead. We need to start a test
    # server that serves the data.
    zip_fname = make_test_zip(tmp_path)
    with zip_fname.open("rb") as f:
        zip_data = f.read()
    httpserver.expect_request("/data.zip", method="GET").respond_with_data(
        zip_data, content_type="application/zip"
    )

    # Initialize the WorldCover instance.
    # But we need to customize the URL that it retrieves from.
    monkeypatch.setattr(WorldCover, "BASE_URL", httpserver.url_for("/"))
    monkeypatch.setattr(WorldCover, "ZIP_FILENAMES", ["data.zip"])
    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    layer_config = RasterLayerConfig(
        LayerType.RASTER,
        [BandSetConfig(config_dict={}, dtype=DType.UINT8, bands=[TEST_BAND])],
    )
    worldcover_dir = UPath(tmp_path) / "worldcover"
    data_source = WorldCover(
        config=layer_config,
        worldcover_dir=worldcover_dir,
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
