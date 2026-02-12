"""Mocked integration tests for the SoilDB data source."""

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
from rslearn.data_sources.data_source import DataSourceContext
from rslearn.data_sources.soildb import SoilDB
from rslearn.dataset import Window
from rslearn.dataset.storage.file import FileWindowStorage
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.raster_format import GeotiffRasterFormat

# seattle2020 fixture is a UTM box; we need WGS84 bounds for the mock STAC item.
SEATTLE_WGS84_BOUNDS = (-122.34, 47.60, -122.32, 47.62)
DEGREES_PER_PIXEL = 0.001


def _make_test_geotiff(path: pathlib.Path) -> pathlib.Path:
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
    data = np.ones((1, height, width), dtype=np.uint16) * 7
    raster_dir = UPath(path / "raster")
    fmt = GeotiffRasterFormat()
    fmt.encode_raster(raster_dir, projection, bounds, data)
    return raster_dir / fmt.fname


def test_materialize_auto_asset(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    httpserver: HTTPServer,
) -> None:
    tif_path = _make_test_geotiff(tmp_path)
    asset_url = f"file://{tif_path}"

    collection_id = "clay.tot_iso.11277.2020.wpct"
    item_id = f"{collection_id}_20200101_20221231"
    item = {
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": item_id,
        "bbox": list(SEATTLE_WGS84_BOUNDS),
        "geometry": shapely.geometry.mapping(shapely.box(*SEATTLE_WGS84_BOUNDS)),
        "properties": {
            "datetime": "2021-01-01T00:00:00Z",
            "start_datetime": "2020-01-01T00:00:00Z",
            "end_datetime": "2022-12-31T00:00:00Z",
        },
        "assets": {
            # The per-collection default should match the 30m 0–30cm mean asset.
            f"{collection_id}_m_30m_b0cm..30cm": {
                "href": asset_url,
                "type": "image/tiff; profile=cloud-optimized",
                "roles": ["data"],
            },
            f"{collection_id}_m_120m_b0cm..30cm": {
                "href": asset_url,
                "type": "image/tiff",
                "roles": ["data"],
            },
            f"{collection_id}_p16_120m_b0cm..30cm": {
                "href": asset_url,
                "type": "image/tiff",
                "roles": ["data"],
            },
            "metadata": {
                "href": "https://example.com/metadata.json",
                "type": "application/json",
                "roles": ["metadata"],
            },
        },
    }
    collection = {
        "type": "Collection",
        "stac_version": "1.0.0",
        "id": collection_id,
        "links": [{"rel": "item", "href": "./item.json", "type": "application/json"}],
    }

    httpserver.expect_request("/collection.json", method="GET").respond_with_json(
        collection
    )
    httpserver.expect_request("/item.json", method="GET").respond_with_json(item)

    band_name = "CLAY"
    layer_config = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[BandSetConfig(dtype=DType.UINT16, bands=[band_name])],
    )
    data_source = SoilDB(
        collection_id=collection_id,
        collection_url=httpserver.url_for("/collection.json"),
        context=DataSourceContext(layer_config=layer_config),
    )

    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([seattle2020], query_config)[0]
    assert len(item_groups) > 0 and len(item_groups[0]) > 0

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
        time_range=(
            datetime(2020, 7, 20, tzinfo=UTC),
            datetime(2020, 7, 21, tzinfo=UTC),
        ),
    )
    window.save()

    data_source.materialize(window, item_groups, "layer", layer_config)
    raster_dir = window.get_raster_dir("layer", [band_name])
    assert (raster_dir / "geotiff.tif").exists()


def test_requires_explicit_asset_key_when_disabled(
    seattle2020: STGeometry,
    httpserver: HTTPServer,
) -> None:
    collection_id = "soil.types_ensemble_probabilities"
    item_id = f"{collection_id}_20000101_20221231"
    item = {
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": item_id,
        "bbox": list(SEATTLE_WGS84_BOUNDS),
        "geometry": shapely.geometry.mapping(shapely.box(*SEATTLE_WGS84_BOUNDS)),
        "properties": {"datetime": "2021-01-01T00:00:00Z"},
        "assets": {
            "a": {"href": "file:///tmp/a.tif", "type": "image/tiff", "roles": ["data"]},
            "b": {"href": "file:///tmp/b.tif", "type": "image/tiff", "roles": ["data"]},
        },
    }
    collection = {
        "type": "Collection",
        "stac_version": "1.0.0",
        "id": collection_id,
        "links": [{"rel": "item", "href": "./item.json", "type": "application/json"}],
    }

    httpserver.expect_request("/collection.json", method="GET").respond_with_json(
        collection
    )
    httpserver.expect_request("/item.json", method="GET").respond_with_json(item)

    data_source = SoilDB(
        collection_id=collection_id,
        collection_url=httpserver.url_for("/collection.json"),
    )

    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    with pytest.raises(ValueError, match="requires asset_key"):
        data_source.get_items([seattle2020], query_config)
