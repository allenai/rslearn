"""Mocked integration tests for the AWS Sentinel-2 Element84 data source."""

import pathlib
from datetime import UTC, datetime

import numpy as np
import pytest
import shapely
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
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.aws_sentinel2_element84 import Sentinel2
from rslearn.dataset import Window
from rslearn.dataset.storage.file import FileWindowStorage
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat
from rslearn.utils.stac import StacAsset, StacItem

SEATTLE_WGS84_BOUNDS = (-122.34, 47.60, -122.32, 47.62)
DEGREES_PER_PIXEL = 0.001


def _make_test_geotiff(path: pathlib.Path) -> pathlib.Path:
    """Create a small test GeoTIFF corresponding to the seattle2020 fixture."""
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
    data = np.ones((1, height, width), dtype=np.uint16) * 1000
    raster_dir = UPath(path / "raster")
    fmt = GeotiffRasterFormat()
    fmt.encode_raster(raster_dir, projection, bounds, RasterArray(chw_array=data))
    return raster_dir / fmt.fname


def _make_stac_item(asset_url: str) -> StacItem:
    """Create a mock StacItem with a red asset."""
    return StacItem(
        id="S2B_56MPC_20200720_0_L2A",
        properties={
            "datetime": "2020-07-20T00:00:00Z",
            "earthsearch:boa_offset_applied": False,
        },
        collection="sentinel-2-l2a",
        bbox=SEATTLE_WGS84_BOUNDS,
        geometry=shapely.geometry.mapping(shapely.box(*SEATTLE_WGS84_BOUNDS)),
        assets={
            "red": StacAsset(
                href=asset_url, title="Red", type="image/tiff", roles=["data"]
            ),
        },
        time_range=(
            datetime(2020, 7, 20, tzinfo=UTC),
            datetime(2020, 7, 21, tzinfo=UTC),
        ),
    )


def test_ingest(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    httpserver: HTTPServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test ingest with mocked STAC API and HTTP asset download."""
    tif_path = _make_test_geotiff(tmp_path)
    with open(tif_path, "rb") as f:
        tif_data = f.read()
    httpserver.expect_request("/test.tif", method="GET").respond_with_data(
        tif_data, content_type="image/tiff"
    )
    asset_url = httpserver.url_for("/test.tif")
    stac_item = _make_stac_item(asset_url)

    data_source = Sentinel2(assets=["red"])
    monkeypatch.setattr(data_source.client, "search", lambda **kw: [stac_item])

    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([seattle2020], query_config)[0]
    assert len(item_groups) > 0 and len(item_groups[0]) > 0
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
    assert tile_store.is_raster_ready(layer_name, item.name, ["B04"])


def test_materialize(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test direct materialize with mocked STAC API and local GeoTIFF."""
    tif_path = _make_test_geotiff(tmp_path)
    asset_url = f"file://{tif_path}"
    stac_item = _make_stac_item(asset_url)

    data_source = Sentinel2(assets=["red"])
    monkeypatch.setattr(data_source.client, "search", lambda **kw: [stac_item])

    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([seattle2020], query_config)[0]
    assert len(item_groups) > 0

    layer_config = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[BandSetConfig(dtype=DType.UINT16, bands=["B04"])],
    )
    bounds = (
        int(seattle2020.shp.bounds[0]),
        int(seattle2020.shp.bounds[1]),
        int(seattle2020.shp.bounds[2]),
        int(seattle2020.shp.bounds[3]),
    )
    window = Window(
        storage=FileWindowStorage(UPath(tmp_path / "rslearn_dataset")),
        group="default",
        name="default",
        projection=seattle2020.projection,
        bounds=bounds,
        time_range=seattle2020.time_range,
    )
    window.save()

    data_source.materialize(window, item_groups, "layer", layer_config)
    raster_dir = window.get_raster_dir("layer", ["B04"])
    assert (raster_dir / "geotiff.tif").exists()
