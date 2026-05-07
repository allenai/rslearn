"""Mocked integration tests for the NASA HLS data source."""

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
from rslearn.data_sources.nasa_hls import Hls2S30
from rslearn.dataset import Window
from rslearn.dataset.storage.file import FileWindowStorage
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat
from rslearn.utils.stac import StacAsset, StacItem

SEATTLE_WGS84_BOUNDS = (-122.34, 47.60, -122.32, 47.62)
DEGREES_PER_PIXEL = 0.001


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
    data = np.ones((1, height, width), dtype=np.uint16) * 1000
    raster_dir = UPath(tmp_path / "raster")
    GeotiffRasterFormat().encode_raster(
        raster_dir,
        projection,
        bounds,
        RasterArray(chw_array=data),
    )
    return raster_dir / "geotiff.tif"


def _make_stac_item(asset_url: str, s3_asset_url: str | None = None) -> StacItem:
    assets = {
        "B04": StacAsset(
            href=asset_url, title="Band 4", type="image/tiff", roles=["data"]
        ),
    }
    if s3_asset_url is not None:
        assets["s3_B04"] = StacAsset(
            href=s3_asset_url,
            title="Band 4 S3",
            type="image/tiff",
            roles=["data"],
        )

    return StacItem(
        id="HLS.S30.T10TET.2020183T185919.v2.0",
        properties={
            "datetime": "2020-07-20T19:11:29.062Z",
            "eo:cloud_cover": 5,
        },
        collection="HLSS30_2.0",
        bbox=SEATTLE_WGS84_BOUNDS,
        geometry=shapely.geometry.mapping(shapely.box(*SEATTLE_WGS84_BOUNDS)),
        assets=assets,
        time_range=(
            datetime(2020, 7, 20, tzinfo=UTC),
            datetime(2020, 7, 21, tzinfo=UTC),
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
    httpserver.expect_request("/test.tif", method="GET").respond_with_data(
        tif_data, content_type="image/tiff"
    )

    data_source = Hls2S30(band_names=["B04"])
    monkeypatch.setattr(
        data_source.client,
        "search",
        lambda **kw: [_make_stac_item(httpserver.url_for("/test.tif"))],
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
    assert tile_store.is_raster_ready("layer", item, ["B04"])


def test_ingest_falls_back_from_s3_to_http(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    test_geotiff: pathlib.Path,
    httpserver: HTTPServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with open(test_geotiff, "rb") as f:
        tif_data = f.read()
    httpserver.expect_request("/fallback.tif", method="GET").respond_with_data(
        tif_data, content_type="image/tiff"
    )

    data_source = Hls2S30(band_names=["B04"])
    monkeypatch.setattr(
        data_source.client,
        "search",
        lambda **kw: [
            _make_stac_item(
                httpserver.url_for("/fallback.tif"),
                "s3://lp-prod-protected/example/B04.tif",
            )
        ],
    )

    original_download_asset = data_source._download_asset

    def mock_download_asset(asset_url: str, local_fname: str) -> None:
        if asset_url.startswith("s3://"):
            raise RuntimeError("AccessDenied")
        original_download_asset(asset_url, local_fname)

    monkeypatch.setattr(data_source, "_download_asset", mock_download_asset)

    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([seattle2020], query_config)[0]
    item = item_groups[0].items[0]

    tile_store_dir = UPath(tmp_path / "tiles_fallback")
    tile_store = DefaultTileStore(str(tile_store_dir))
    tile_store.set_dataset_path(tile_store_dir)

    data_source.ingest(
        TileStoreWithLayer(tile_store, "layer"),
        item_groups[0].items,
        [[seattle2020]],
    )
    assert tile_store.is_raster_ready("layer", item, ["B04"])


def test_materialize(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    test_geotiff: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_source = Hls2S30(band_names=["B04"])
    monkeypatch.setattr(
        data_source.client,
        "search",
        lambda **kw: [_make_stac_item(f"file://{test_geotiff}")],
    )

    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([seattle2020], query_config)[0]

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

    data_source.materialize(
        window,
        [group.items for group in item_groups],
        "layer",
        layer_config,
    )
    assert window.is_layer_completed("layer")
