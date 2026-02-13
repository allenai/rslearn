import pathlib
import zipfile

import numpy as np
import shapely
from pytest_httpserver import HTTPServer
from upath import UPath

from rslearn.config import (
    QueryConfig,
    SpaceMode,
)
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.data_source import DataSourceContext
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


def _make_test_zips(tmp_path: pathlib.Path) -> dict[str, pathlib.Path]:
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


def _setup_worldcereal_httpserver(
    worldcereal_dir: UPath, httpserver: HTTPServer
) -> None:
    """Create test zips and configure httpserver to serve them.

    Args:
        worldcereal_dir: directory to create test zips in.
        httpserver: the pytest httpserver to configure.
    """
    zip_name_paths = _make_test_zips(worldcereal_dir)
    for zip_file, zip_fname in zip_name_paths.items():
        with zip_fname.open("rb") as f:
            zip_data = f.read()
        httpserver.expect_request(f"/{zip_file}", method="GET").respond_with_data(
            zip_data, content_type="application/zip"
        )


def test_with_worldcereal_dir(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    httpserver: HTTPServer,
) -> None:
    """Tests ingesting the example data corresponding to seattle2020."""
    worldcereal_dir = UPath(tmp_path) / "worldcereal"
    _setup_worldcereal_httpserver(worldcereal_dir, httpserver)

    bands = [WorldCereal.band_from_zipfilename(f) for f in WorldCereal.ZIP_FILENAMES]
    for band in bands:
        print(f"Testing {band}")
        query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
        data_source = WorldCereal(
            band=band,
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
        assert raster_data.get_chw_array().max() == 1
        print(f"Succeeded for {band}")


def test_with_context_ds_path(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    httpserver: HTTPServer,
) -> None:
    """Tests WorldCereal when context.ds_path is set.

    Previously there was a bug where WorldCereal would pass a UPath to
    LocalFiles.__init__, which would have error when calling join_upath since it
    expects src_dir to be a string. This test prevents regression by making sure the
    code path works when including a DataSourceContext when initializing worldCereal.
    """
    ds_path = UPath(tmp_path) / "dataset"
    ds_path.mkdir(parents=True, exist_ok=True)

    # Use a relative worldcereal_dir path (relative to ds_path)
    worldcereal_dir = "worldcereal_data"
    worldcereal_abs_path = ds_path / worldcereal_dir

    _setup_worldcereal_httpserver(worldcereal_abs_path, httpserver)

    # Test with just the first band to keep the test fast
    band = WorldCereal.band_from_zipfilename(WorldCereal.ZIP_FILENAMES[0])

    # Create context with ds_path set - this triggers join_upath code path
    context = DataSourceContext(ds_path=ds_path)
    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    data_source = WorldCereal(
        band=band,
        worldcereal_dir=worldcereal_dir,  # relative path
        context=context,
    )

    # Verify we can get items (this exercises list_items -> join_upath)
    item_groups = data_source.get_items([seattle2020], query_config)
    assert len(item_groups) == 1
    assert len(item_groups[0]) > 0
    item = item_groups[0][0][0]

    # Verify ingest works
    tile_store_dir = ds_path / "tile_store"
    tile_store = DefaultTileStore(str(tile_store_dir))
    tile_store.set_dataset_path(tile_store_dir)

    layer_name = "layer"
    data_source.ingest(
        TileStoreWithLayer(tile_store, layer_name),
        item_groups[0][0],
        [[seattle2020]],
    )
    assert tile_store.is_raster_ready(layer_name, item.name, [band])
