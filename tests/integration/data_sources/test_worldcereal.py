import pathlib
import zipfile

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
from rslearn.data_sources.worldcereal import WorldCereal
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.raster_format import GeotiffRasterFormat

# Degrees per pixel to use in the GeoTIFF.
# This roughly corresponds to 10 m/pixel.
DEGREES_PER_PIXEL = 0.0001

# Size of the GeoTIFF.
SIZE = 16

# See tests/integration/fixtures/geometries/seattle2020/seattle2020.py.
SEATTLE_POINT = shapely.Point(-122.33, 47.61)


def make_test_zips(tmp_path: pathlib.Path) -> dict[str, pathlib.Path]:
    """Make a sample zip file similar to the ESA WorldCereal 2021 ones.

    This is a little bit circular since it uses the class to define where the
    tif files go (and how they are named).

    Our zip file will just contain a single 16x16 GeoTIFF. We make sure it corresponds
    the seattle2020 test geometry so that it can be used to test the data source.

    Args:
        tmp_path: temporary directory that will be used to store the GeoTIFF and zip
            files.

    Returns:
        the filename of the zip files
    """
    seattle_aez = 1
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

    return_dict = {}
    for zip_file in WorldCereal.ZIP_FILENAMES:
        filepath = WorldCereal.zip_filepath_from_filename(zip_file)
        raster_path = UPath(tmp_path / "zips" / filepath)
        raster_path.mkdir(parents=True)
        raster_format = GeotiffRasterFormat()
        raster_format.encode_raster(
            raster_path,
            projection,
            bounds,
            array,
            fname=f"{seattle_aez}_{raster_path.stem}.tif",
        )

        # Create a zip file containing it.
        zip_fname = tmp_path / "zips" / zip_file
        zipf = zipfile.ZipFile(zip_fname, "w")
        zipf.write(
            raster_path / f"{seattle_aez}_{raster_path.stem}.tif",
            arcname=UPath(filepath) / f"{seattle_aez}_{raster_path.stem}.tif",
        )
        zipf.close()

        return_dict[zip_file] = zip_fname
    return return_dict


def test_with_worldcereal_dir(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    httpserver: HTTPServer,
) -> None:
    """Tests ingesting the example data corresponding to seattle2020.

    Args:
        worldcereal_dir: the directory to store WorldCereal GeoTIFFs.
        tmp_path: temporary path for making zip file and for tile store.
        seattle2020: the geometry to use for prepare.
        httpserver: server for serving the example data.
    """
    worldcereal_dir = UPath(tmp_path) / "worldcereal"
    # The WorldCover data is large so we use test data instead. We need to start a test
    # server that serves the data.
    zip_name_paths = make_test_zips(worldcereal_dir)
    for zip_file, zip_fname in zip_name_paths.items():
        with zip_fname.open("rb") as f:
            zip_data = f.read()
        httpserver.expect_request(f"/{zip_file}", method="GET").respond_with_data(
            zip_data, content_type="application/zip"
        )

    bands = [WorldCereal.band_from_zipfilename(f) for f in WorldCereal.ZIP_FILENAMES]
    for band in bands:
        print(f"Testing {band}")
        query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
        layer_config = RasterLayerConfig(
            LayerType.RASTER,
            [BandSetConfig(config_dict={}, dtype=DType.UINT8, bands=[band])],
        )
        data_source = WorldCereal(
            band=band,
            config=layer_config,
            worldcereal_dir=worldcereal_dir,
        )

        print("get items")
        item_groups = data_source.get_items([seattle2020], query_config)
        item = item_groups[0][0][0]
        tile_store_dir = UPath(worldcereal_dir) / "tile_store"
        tile_store = DefaultTileStore(str(tile_store_dir))
        tile_store.set_dataset_path(tile_store_dir)

        print("ingest")
        layer_name = "layer"
        data_source.ingest(
            TileStoreWithLayer(tile_store, layer_name),
            item_groups[0][0],
            [[seattle2020]],
        )
        print(list(tile_store_dir.glob("layer/1/*")))
        assert tile_store.is_raster_ready(layer_name, item.name, [band])
        # Double check that the data intersected our example GeoTIFF and isn't just all 0.
        bounds = (
            int(seattle2020.shp.bounds[0]),
            int(seattle2020.shp.bounds[1]),
            int(seattle2020.shp.bounds[2]),
            int(seattle2020.shp.bounds[3]),
        )
        raster_data = tile_store.read_raster(
            layer_name, item.name, [band], seattle2020.projection, bounds
        )
        assert raster_data.max() == 1
        print(f"Succeeded for {band}")
