"""Mocked integration tests for the NASA ECOSTRESS L2T LSTE data source."""

import pathlib
from datetime import UTC, datetime

import numpy as np
import pytest
import shapely
from pytest_httpserver import HTTPServer
from upath import UPath

from rslearn.config import QueryConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.nasa_ecostress import EcostressLSTE
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat
from rslearn.utils.stac import StacAsset, StacItem

SEATTLE_WGS84_BOUNDS = (-122.34, 47.60, -122.32, 47.62)
DEGREES_PER_PIXEL = 0.001
GRANULE = "ECOv002_L2T_LSTE_00376_004_13TDE_20180731T000421_0712_01"


@pytest.fixture
def test_geotiff(tmp_path: pathlib.Path) -> pathlib.Path:
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
    data = np.full((1, height, width), 300.0, dtype=np.float32)
    raster_dir = UPath(tmp_path / "raster")
    GeotiffRasterFormat().encode_raster(
        raster_dir,
        projection,
        bounds,
        RasterArray(chw_array=data),
    )
    return raster_dir / "geotiff.tif"


def _make_stac_item(lst_href: str) -> StacItem:
    http_key = f"002/{GRANULE}/{GRANULE}_LST"
    assets = {
        http_key: StacAsset(
            href=lst_href, title="LST", type="image/tiff", roles=["data"]
        ),
    }
    return StacItem(
        id=GRANULE,
        properties={"datetime": "2020-07-20T00:04:21Z"},
        collection=EcostressLSTE.COLLECTION_NAME,
        bbox=SEATTLE_WGS84_BOUNDS,
        geometry=shapely.geometry.mapping(shapely.box(*SEATTLE_WGS84_BOUNDS)),
        assets=assets,
        time_range=(
            datetime(2020, 7, 20, tzinfo=UTC),
            datetime(2020, 7, 20, 0, 5, tzinfo=UTC),
        ),
    )


def test_ingest(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    test_geotiff: pathlib.Path,
    httpserver: HTTPServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with open(test_geotiff, "rb") as f:
        tif_data = f.read()
    httpserver.expect_request("/lst.tif", method="GET").respond_with_data(
        tif_data, content_type="image/tiff"
    )

    data_source = EcostressLSTE(band_names=["LST"])
    monkeypatch.setattr(
        data_source.client,
        "search",
        lambda **kw: [_make_stac_item(httpserver.url_for("/lst.tif"))],
    )

    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([seattle2020], query_config)[0]
    item = item_groups[0].items[0]

    tile_store_dir = UPath(tmp_path / "tiles")
    tile_store = DefaultTileStore(str(tile_store_dir))
    tile_store.set_dataset_path(tile_store_dir)

    data_source.ingest(
        TileStoreWithLayer(tile_store, "layer"),
        item_groups[0].items,
        [[seattle2020]],
    )
    assert tile_store.is_raster_ready("layer", item, ["LST"])


def test_get_items_skips_item_without_lst(
    seattle2020: STGeometry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An item that lacks the requested LST asset should be filtered out."""
    qc_only = StacItem(
        id=GRANULE,
        properties={"datetime": "2020-07-20T00:04:21Z"},
        collection=EcostressLSTE.COLLECTION_NAME,
        bbox=SEATTLE_WGS84_BOUNDS,
        geometry=shapely.geometry.mapping(shapely.box(*SEATTLE_WGS84_BOUNDS)),
        assets={
            f"002/{GRANULE}/{GRANULE}_QC": StacAsset(
                href="https://example.com/QC.tif",
                title="QC",
                type="image/tiff",
                roles=["data"],
            ),
        },
        time_range=(
            datetime(2020, 7, 20, tzinfo=UTC),
            datetime(2020, 7, 20, 0, 5, tzinfo=UTC),
        ),
    )

    data_source = EcostressLSTE(band_names=["LST"])
    monkeypatch.setattr(data_source.client, "search", lambda **kw: [qc_only])

    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([seattle2020], query_config)[0]
    assert item_groups == []
