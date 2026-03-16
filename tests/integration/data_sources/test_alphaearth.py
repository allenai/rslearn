"""Mocked integration tests for the AlphaEarth data source."""

import pathlib
import shutil
import tempfile
from typing import Any

import numpy as np
import pytest
import rasterio
import shapely
from upath import UPath

from rslearn.config import QueryConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.alphaearth import BANDS, AlphaEarth
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat

duckdb = pytest.importorskip("duckdb")

SEATTLE_WGS84_BOUNDS = (-122.34, 47.60, -122.32, 47.62)
FAR_WGS84_BOUNDS = (-122.70, 47.10, -122.60, 47.20)
DEGREES_PER_PIXEL = 0.001
RAW_PIXEL_VALUE = 50
DEQUANTIZED_VALUE = ((RAW_PIXEL_VALUE / 127.5) ** 2) * np.sign(RAW_PIXEL_VALUE)


def _load_spatial_extension(con: Any) -> None:
    try:
        con.execute("LOAD spatial;")
    except duckdb.Error:
        con.execute("INSTALL spatial;")
        con.execute("LOAD spatial;")


@pytest.fixture
def test_geotiff(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a small 64-band int8 GeoTIFF mimicking AlphaEarth data."""
    projection = Projection(WGS84_PROJECTION.crs, DEGREES_PER_PIXEL, -DEGREES_PER_PIXEL)
    west, south, east, north = SEATTLE_WGS84_BOUNDS
    bounds = (
        round(west / DEGREES_PER_PIXEL),
        round(north / -DEGREES_PER_PIXEL),
        round(east / DEGREES_PER_PIXEL),
        round(south / -DEGREES_PER_PIXEL),
    )
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    data = np.full((64, height, width), RAW_PIXEL_VALUE, dtype=np.int8)
    raster_dir = UPath(tmp_path / "raster")
    fmt = GeotiffRasterFormat()
    fmt.encode_raster(raster_dir, projection, bounds, RasterArray(chw_array=data))
    return raster_dir / fmt.fname


@pytest.fixture
def alphaearth_index() -> pathlib.Path:
    """Create a tiny GeoParquet STAC index for AlphaEarth."""
    index_path = pathlib.Path(tempfile.mkdtemp()) / "alphaearth_index.parquet"
    escaped_index_path = str(index_path).replace("'", "''")
    con = duckdb.connect()
    _load_spatial_extension(con)

    seattle_wkt = shapely.box(*SEATTLE_WGS84_BOUNDS).wkt
    far_wkt = shapely.box(*FAR_WGS84_BOUNDS).wkt
    con.execute(
        f"""
        COPY (
            SELECT
                item_id AS id,
                item_datetime AS datetime,
                STRUCT_PACK(
                    data := STRUCT_PACK(
                        href := data_href,
                        "type" := 'image/tiff',
                        roles := ['data']
                    )
                ) AS assets,
                ST_GeomFromText(geometry_wkt) AS geometry
            FROM (
                VALUES
                    (
                        'test_tile_2019',
                        TIMESTAMPTZ '2019-01-01 00:00:00+00',
                        's3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2019/test_tile_2019.tiff',
                        ?
                    ),
                    (
                        'test_tile_2020',
                        TIMESTAMPTZ '2020-01-01 00:00:00+00',
                        's3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2020/test_tile_2020.tiff',
                        ?
                    ),
                    (
                        'far_tile_2020',
                        TIMESTAMPTZ '2020-01-01 00:00:00+00',
                        's3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2020/far_tile_2020.tiff',
                        ?
                    )
                ) AS t(item_id, item_datetime, data_href, geometry_wkt)
        ) TO '{escaped_index_path}' (FORMAT PARQUET)
        """,
        [seattle_wkt, seattle_wkt, far_wkt],
    )
    con.close()
    assert index_path.exists()
    return index_path


def test_get_items(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    alphaearth_index: pathlib.Path,
) -> None:
    """Test that the AlphaEarth index is filtered by geometry and time."""
    data_source = AlphaEarth(
        metadata_cache_dir=str(tmp_path / "cache"),
        index_url=str(alphaearth_index),
    )

    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([seattle2020], query_config)[0]

    assert [group[0].name for group in item_groups] == ["test_tile_2020"]
    assert (
        data_source.get_item_by_name("test_tile_2020").data_href
        == "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2020/test_tile_2020.tiff"
    )


@pytest.mark.parametrize(
    "apply_dequantization,expected_value",
    [
        (False, RAW_PIXEL_VALUE),
        (True, pytest.approx(DEQUANTIZED_VALUE, abs=1e-4)),
    ],
    ids=["raw", "dequantized"],
)
def test_ingest(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    alphaearth_index: pathlib.Path,
    test_geotiff: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    apply_dequantization: bool,
    expected_value: float,
) -> None:
    """Test ingest with mocked AlphaEarth downloads."""

    def mock_download(url: str, local_path: str) -> None:
        shutil.copy(str(test_geotiff), local_path)

    monkeypatch.setattr(
        "rslearn.data_sources.alphaearth._download_url",
        mock_download,
    )

    data_source = AlphaEarth(
        metadata_cache_dir=str(tmp_path / "cache"),
        index_url=str(alphaearth_index),
        apply_dequantization=apply_dequantization,
    )

    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([seattle2020], query_config)[0]
    item = item_groups[0][0]

    tile_store_dir = UPath(tmp_path / "tiles")
    tile_store = DefaultTileStore(str(tile_store_dir))
    tile_store.set_dataset_path(tile_store_dir)
    layer_name = "layer"

    data_source.ingest(
        TileStoreWithLayer(tile_store, layer_name),
        item_groups[0],
        [[seattle2020]],
    )
    assert tile_store.is_raster_ready(layer_name, item, BANDS)

    bounds = (
        int(seattle2020.shp.bounds[0]),
        int(seattle2020.shp.bounds[1]),
        int(seattle2020.shp.bounds[2]),
        int(seattle2020.shp.bounds[3]),
    )
    array = tile_store.read_raster(
        layer_name, item, BANDS, seattle2020.projection, bounds
    )
    assert array.get_chw_array().max() == expected_value


@pytest.mark.parametrize(
    "apply_dequantization,expect_float",
    [
        (False, False),
        (True, True),
    ],
    ids=["raw", "dequantized"],
)
def test_read_raster_subset_bands(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    alphaearth_index: pathlib.Path,
    test_geotiff: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    apply_dequantization: bool,
    expect_float: bool,
) -> None:
    """Test direct raster reads with a subset of AlphaEarth bands."""
    data_source = AlphaEarth(
        metadata_cache_dir=str(tmp_path / "cache"),
        index_url=str(alphaearth_index),
        apply_dequantization=apply_dequantization,
    )
    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([seattle2020], query_config)[0]
    item = item_groups[0][0]

    original_rasterio_open = rasterio.open

    def mock_rasterio_open(url: Any, *args: Any, **kwargs: Any) -> Any:
        if "data.source.coop" in str(url):
            return original_rasterio_open(str(test_geotiff), *args, **kwargs)
        return original_rasterio_open(url, *args, **kwargs)

    monkeypatch.setattr(rasterio, "open", mock_rasterio_open)

    bounds = (
        int(seattle2020.shp.bounds[0]),
        int(seattle2020.shp.bounds[1]),
        int(seattle2020.shp.bounds[2]),
        int(seattle2020.shp.bounds[3]),
    )
    subset_bands = ["A00", "A01", "A02"]

    array = data_source.read_raster(
        layer_name="alphaearth",
        item=item,
        bands=subset_bands,
        projection=seattle2020.projection,
        bounds=bounds,
    )

    arr = array.get_chw_array()
    expected_height = bounds[3] - bounds[1]
    expected_width = bounds[2] - bounds[0]
    assert arr.shape == (3, expected_height, expected_width)
    if expect_float:
        assert arr.dtype == np.float32
    else:
        assert np.issubdtype(arr.dtype, np.integer)
