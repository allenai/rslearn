from datetime import datetime

import affine
import numpy as np
import pytest
import rasterio
import shapely
from pytest_httpserver import HTTPServer
from rasterio.io import MemoryFile
from upath import UPath


def _make_geotiff_bytes(
    array: np.ndarray,
    crs: rasterio.crs.CRS,
    transform: affine.Affine,
    nodata: int | float | None = None,
) -> bytes:
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=array.shape[1],
            width=array.shape[2],
            count=array.shape[0],
            dtype=array.dtype,
            crs=crs,
            transform=transform,
            nodata=nodata,
        ) as dst:
            dst.write(array)
        return memfile.read()


def test_sentinel2_ingest_does_not_apply_scl_cloud_mask(
    tmp_path,
    httpserver: HTTPServer,
) -> None:
    pytest.importorskip("earthdaily")

    from rslearn.const import WGS84_PROJECTION
    from rslearn.data_sources.earthdaily import EarthDailyItem, Sentinel2
    from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
    from rslearn.utils.geometry import Projection, STGeometry

    # Create a tiny georeferenced raster and an aligned SCL mask.
    projection = Projection(WGS84_PROJECTION.crs, 0.0001, -0.0001)
    bounds = (0, 0, 4, 4)
    transform = affine.Affine(
        projection.x_resolution,
        0,
        bounds[0] * projection.x_resolution,
        0,
        projection.y_resolution,
        bounds[1] * projection.y_resolution,
    )

    red = np.full((1, 4, 4), 100, dtype=np.uint16)
    scl = np.zeros((1, 4, 4), dtype=np.uint8)
    scl[0, 1, 1] = 8  # medium cloud (default excluded)

    red_bytes = _make_geotiff_bytes(red, projection.crs, transform, nodata=0)
    scl_bytes = _make_geotiff_bytes(scl, projection.crs, transform, nodata=0)

    httpserver.expect_request("/red.tif", method="GET").respond_with_data(
        red_bytes, content_type="image/tiff"
    )
    httpserver.expect_request("/scl.tif", method="GET").respond_with_data(
        scl_bytes, content_type="image/tiff"
    )

    item_geom = STGeometry(
        WGS84_PROJECTION,
        shapely.box(-1, -1, 1, 1),
        (datetime(2020, 1, 1), datetime(2020, 1, 1)),
    )
    item = EarthDailyItem(
        name="S2_TEST_ITEM",
        geometry=item_geom,
        asset_urls={
            "red": httpserver.url_for("/red.tif"),
            "scl": httpserver.url_for("/scl.tif"),
        },
    )

    data_source = Sentinel2(assets=["red", "scl"], apply_scale_offset=False)

    ds_path = UPath(tmp_path / "ds")
    tile_store = DefaultTileStore(convert_rasters_to_cogs=False)
    tile_store.set_dataset_path(ds_path)
    layer_tile_store = TileStoreWithLayer(tile_store, "sentinel2")

    data_source.ingest(
        tile_store=layer_tile_store,
        items=[item],
        geometries=[[item_geom]],
    )

    out = tile_store.read_raster("sentinel2", item.name, ["B04"], projection, bounds)
    assert out.shape == red.shape
    assert out[0, 0, 0] == 100
    assert out[0, 1, 1] == 100

    out_scl = tile_store.read_raster("sentinel2", item.name, ["scl"], projection, bounds)
    assert out_scl.shape == scl.shape
    assert out_scl[0, 1, 1] == 8


def test_sentinel2_read_raster_applies_scale_offset_when_apply_scale_offset_true(
    httpserver: HTTPServer,
) -> None:
    pytest.importorskip("earthdaily")

    from rslearn.const import WGS84_PROJECTION
    from rslearn.data_sources.earthdaily import EarthDailyItem, Sentinel2
    from rslearn.utils.geometry import Projection, STGeometry

    projection = Projection(WGS84_PROJECTION.crs, 0.0001, -0.0001)
    bounds = (0, 0, 4, 4)
    transform = affine.Affine(
        projection.x_resolution,
        0,
        bounds[0] * projection.x_resolution,
        0,
        projection.y_resolution,
        bounds[1] * projection.y_resolution,
    )

    raw = np.full((1, 4, 4), 100, dtype=np.uint16)
    red_bytes = _make_geotiff_bytes(raw, projection.crs, transform, nodata=0)
    httpserver.expect_request("/red.tif", method="GET").respond_with_data(
        red_bytes, content_type="image/tiff"
    )

    item_geom = STGeometry(
        WGS84_PROJECTION,
        shapely.box(-1, -1, 1, 1),
        (datetime(2020, 1, 1), datetime(2020, 1, 1)),
    )
    item = EarthDailyItem(
        name="S2_TEST_ITEM",
        geometry=item_geom,
        asset_urls={"red": httpserver.url_for("/red.tif")},
        asset_scale_offsets={"red": [{"scale": 0.1, "offset": 1.0}]},
    )

    data_source = Sentinel2(assets=["red"], apply_scale_offset=True)
    data_source.get_item_by_name = lambda _: item  # type: ignore[method-assign]

    out = data_source.read_raster("sentinel2", item.name, ["B04"], projection, bounds)
    assert out.dtype == np.float32
    assert out.shape == raw.shape
    assert out[0, 0, 0] == pytest.approx(11.0)


def test_sentinel2_ingest_applies_scale_offset_when_apply_scale_offset_true(
    tmp_path,
    httpserver: HTTPServer,
) -> None:
    pytest.importorskip("earthdaily")

    from rslearn.const import WGS84_PROJECTION
    from rslearn.data_sources.earthdaily import EarthDailyItem, Sentinel2
    from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
    from rslearn.utils.geometry import Projection, STGeometry

    projection = Projection(WGS84_PROJECTION.crs, 0.0001, -0.0001)
    bounds = (0, 0, 4, 4)
    transform = affine.Affine(
        projection.x_resolution,
        0,
        bounds[0] * projection.x_resolution,
        0,
        projection.y_resolution,
        bounds[1] * projection.y_resolution,
    )

    raw = np.full((1, 4, 4), 100, dtype=np.uint16)
    red_bytes = _make_geotiff_bytes(raw, projection.crs, transform, nodata=0)
    httpserver.expect_request("/red.tif", method="GET").respond_with_data(
        red_bytes, content_type="image/tiff"
    )

    item_geom = STGeometry(
        WGS84_PROJECTION,
        shapely.box(-1, -1, 1, 1),
        (datetime(2020, 1, 1), datetime(2020, 1, 1)),
    )
    item = EarthDailyItem(
        name="S2_TEST_ITEM",
        geometry=item_geom,
        asset_urls={"red": httpserver.url_for("/red.tif")},
        asset_scale_offsets={"red": [{"scale": 0.1, "offset": 1.0}]},
    )

    data_source = Sentinel2(assets=["red"], apply_scale_offset=True)

    ds_path = UPath(tmp_path / "ds")
    tile_store = DefaultTileStore(convert_rasters_to_cogs=False)
    tile_store.set_dataset_path(ds_path)
    layer_tile_store = TileStoreWithLayer(tile_store, "sentinel2")

    data_source.ingest(
        tile_store=layer_tile_store,
        items=[item],
        geometries=[[item_geom]],
    )

    out = tile_store.read_raster("sentinel2", item.name, ["B04"], projection, bounds)
    assert out.dtype == np.float32
    assert out.shape == raw.shape
    assert out[0, 0, 0] == pytest.approx(11.0)


def test_sentinel2_get_items_passes_query_to_helper() -> None:
    pytest.importorskip("earthdaily")

    from rslearn.config.dataset import QueryConfig
    from rslearn.const import WGS84_PROJECTION
    from rslearn.data_sources.earthdaily import Sentinel2
    from rslearn.utils.geometry import STGeometry

    captured: dict[str, object] = {}

    class StubSearchResult:
        def item_collection(self):
            return []

    class StubClient:
        def search(self, **kwargs):
            captured.update(kwargs)
            return StubSearchResult()

    data_source = Sentinel2(
        assets=["red"],
        cloud_cover_max=15.0,
        query={"s2:product_type": {"eq": "S2MSI2A"}},
    )
    data_source._load_client = (  # type: ignore[method-assign]
        lambda: (object(), StubClient(), object())
    )

    geometry = STGeometry(
        WGS84_PROJECTION,
        shapely.box(-1, -1, 1, 1),
        (datetime(2020, 1, 1), datetime(2020, 1, 2)),
    )

    data_source.get_items([geometry], QueryConfig())

    assert captured["query"] == {
        "s2:product_type": {"eq": "S2MSI2A"},
        "eo:cloud_cover": {"lt": 15.0},
    }
    assert captured["max_items"] == 500


def test_sentinel2_ingest_raises_on_download_failure(
    tmp_path,
    httpserver: HTTPServer,
) -> None:
    pytest.importorskip("earthdaily")

    import requests

    from rslearn.const import WGS84_PROJECTION
    from rslearn.data_sources.earthdaily import EarthDailyItem, Sentinel2
    from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
    from rslearn.utils.geometry import STGeometry

    httpserver.expect_request("/red.tif", method="GET").respond_with_data(
        b"nope", status=500
    )

    item_geom = STGeometry(
        WGS84_PROJECTION,
        shapely.box(-1, -1, 1, 1),
        (datetime(2020, 1, 1), datetime(2020, 1, 1)),
    )
    item = EarthDailyItem(
        name="S2_TEST_ITEM",
        geometry=item_geom,
        asset_urls={"red": httpserver.url_for("/red.tif")},
    )

    data_source = Sentinel2(assets=["red"])

    ds_path = UPath(tmp_path / "ds")
    tile_store = DefaultTileStore(convert_rasters_to_cogs=False)
    tile_store.set_dataset_path(ds_path)
    layer_tile_store = TileStoreWithLayer(tile_store, "sentinel2")

    with pytest.raises(requests.exceptions.HTTPError):
        data_source.ingest(
            tile_store=layer_tile_store,
            items=[item],
            geometries=[[item_geom]],
        )
