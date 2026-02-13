"""Mocked integration tests for the PlanetaryComputer data source."""

import pathlib
import xml.etree.ElementTree as ET
from datetime import UTC, datetime

import numpy as np
import planetary_computer
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
from rslearn.data_sources.planetary_computer import Sentinel2
from rslearn.dataset import Window
from rslearn.dataset.storage.file import FileWindowStorage
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.raster_format import GeotiffRasterFormat
from rslearn.utils.stac import StacAsset, StacItem

# seattle2020 fixture is a UTM box, we need WGS84 bounds for the mock STAC item.
SEATTLE_WGS84_BOUNDS = (-122.34, 47.60, -122.32, 47.62)
DEGREES_PER_PIXEL = 0.001

# Raw pixel value we write into the test GeoTIFF.
RAW_PIXEL_VALUE = 2000


@pytest.fixture
def test_geotiff(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a small test GeoTIFF in WGS84 covering the Seattle area."""
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
    data = np.ones((1, height, width), dtype=np.uint16) * RAW_PIXEL_VALUE
    raster_dir = UPath(tmp_path / "raster")
    fmt = GeotiffRasterFormat()
    fmt.encode_raster(raster_dir, projection, bounds, data)
    return raster_dir / fmt.fname


def _make_product_xml(scene_needs_harmonization: bool) -> ET.Element:
    """Create a mock product metadata XML.

    Args:
        scene_needs_harmonization: if True, include BOA_ADD_OFFSET=-1000 (the scene
            has +1000 offset baked into pixel values that needs to be removed).
            If False, omit it (no offset was applied to pixel values).
    """
    root = ET.Element("Level-2A_User_Product")
    if scene_needs_harmonization:
        offsets = ET.SubElement(root, "BOA_ADD_OFFSET_VALUES_LIST")
        for band_id in range(13):
            el = ET.SubElement(offsets, "BOA_ADD_OFFSET")
            el.set("band_id", str(band_id))
            el.text = "-1000"
    return root


def _make_stac_item(asset_url: str, product_metadata_url: str) -> StacItem:
    """Create a mock StacItem pointing to the given asset URLs."""
    return StacItem(
        id="S2B_MSIL2A_20200720T000000_test",
        properties={
            "datetime": "2020-07-20T00:00:00Z",
        },
        collection="sentinel-2-l2a",
        bbox=SEATTLE_WGS84_BOUNDS,
        geometry=shapely.geometry.mapping(shapely.box(*SEATTLE_WGS84_BOUNDS)),
        assets={
            "B04": StacAsset(
                href=asset_url, title="Band 4", type="image/tiff", roles=["data"]
            ),
            "product-metadata": StacAsset(
                href=product_metadata_url,
                title="Product metadata",
                type="application/xml",
                roles=["metadata"],
            ),
        },
        time_range=(
            datetime(2020, 7, 20, tzinfo=UTC),
            datetime(2020, 7, 21, tzinfo=UTC),
        ),
    )


HARMONIZATION_PARAMS = pytest.mark.parametrize(
    "enable_harmonization,scene_needs_harmonization,expected_value",
    [
        # Harmonization disabled: raw value always preserved.
        (False, False, RAW_PIXEL_VALUE),
        # Harmonization enabled, scene doesn't need it: value unchanged.
        (True, False, RAW_PIXEL_VALUE),
        # Harmonization enabled, scene needs it: clip(2000,1000)-1000 = 1000.
        (True, True, RAW_PIXEL_VALUE - 1000),
    ],
    ids=[
        "disabled-no_offset",
        "enabled-no_offset",
        "enabled-has_offset",
    ],
)


@HARMONIZATION_PARAMS
def test_sentinel2_ingest(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    test_geotiff: pathlib.Path,
    httpserver: HTTPServer,
    monkeypatch: pytest.MonkeyPatch,
    enable_harmonization: bool,
    scene_needs_harmonization: bool,
    expected_value: int,
) -> None:
    """Test PlanetaryComputer Sentinel2 ingest with harmonization variants."""
    monkeypatch.setattr(planetary_computer, "sign", lambda url: url)

    # Serve the test GeoTIFF for download during ingest.
    with open(test_geotiff, "rb") as f:
        tif_data = f.read()
    httpserver.expect_request("/test.tif", method="GET").respond_with_data(
        tif_data, content_type="image/tiff"
    )

    # Serve the product metadata XML.
    xml_bytes = ET.tostring(_make_product_xml(scene_needs_harmonization))
    httpserver.expect_request("/metadata.xml", method="GET").respond_with_data(
        xml_bytes, content_type="application/xml"
    )

    # Create the data source with mocked search function.
    data_source = Sentinel2(assets=["B04"], harmonize=enable_harmonization)
    stac_item = _make_stac_item(
        asset_url=httpserver.url_for("/test.tif"),
        product_metadata_url=httpserver.url_for("/metadata.xml"),
    )
    monkeypatch.setattr(data_source.client, "search", lambda **kw: [stac_item])

    # Run ingestion.
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

    # Read back and verify pixel values based on harmonization.
    bounds = (
        int(seattle2020.shp.bounds[0]),
        int(seattle2020.shp.bounds[1]),
        int(seattle2020.shp.bounds[2]),
        int(seattle2020.shp.bounds[3]),
    )
    array = tile_store.read_raster(
        layer_name, item.name, ["B04"], seattle2020.projection, bounds
    )
    assert array.get_chw_array().max() == expected_value


@HARMONIZATION_PARAMS
def test_sentinel2_materialize(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    test_geotiff: pathlib.Path,
    httpserver: HTTPServer,
    monkeypatch: pytest.MonkeyPatch,
    enable_harmonization: bool,
    scene_needs_harmonization: bool,
    expected_value: int,
) -> None:
    """Test PlanetaryComputer Sentinel2 direct materialize with harmonization variants."""
    monkeypatch.setattr(planetary_computer, "sign", lambda url: url)

    # Serve the product metadata XML (needed by get_read_callback for harmonization).
    xml_bytes = ET.tostring(_make_product_xml(scene_needs_harmonization))
    httpserver.expect_request("/metadata.xml", method="GET").respond_with_data(
        xml_bytes, content_type="application/xml"
    )

    # Create the data source with mock STAC item.
    data_source = Sentinel2(assets=["B04"], harmonize=enable_harmonization)
    stac_item = _make_stac_item(
        asset_url=f"file://{test_geotiff}",
        product_metadata_url=httpserver.url_for("/metadata.xml"),
    )
    monkeypatch.setattr(data_source.client, "search", lambda **kw: [stac_item])

    # Get the item to use.
    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([seattle2020], query_config)[0]
    assert len(item_groups) > 0

    # Perform the materialization.
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

    # Verify it was materialized.
    raster_dir = window.get_raster_dir("layer", ["B04"])
    assert (raster_dir / "geotiff.tif").exists()

    # Read back and verify pixel values match expected harmonization behavior.
    raster_array = GeotiffRasterFormat().decode_raster(
        raster_dir, seattle2020.projection, bounds
    )
    array = raster_array.get_chw_array()
    assert array.shape == (1, bounds[3] - bounds[1], bounds[2] - bounds[0])
    assert array.max() == expected_value
