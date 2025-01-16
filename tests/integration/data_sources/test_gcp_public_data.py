import os
import pathlib
import random
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
import shapely
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
from rslearn.data_sources.copernicus import get_sentinel2_tile_index
from rslearn.data_sources.gcp_public_data import (
    CorruptItemException,
    MissingXMLException,
    Sentinel2,
)
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils import STGeometry

TEST_BAND = "B04"


class TestSentinel2:
    """Tests the Sentinel2 data source."""

    def run_simple_test(
        self, tile_store_dir: UPath, seattle2020: STGeometry, **kwargs: Any
    ) -> None:
        """Apply test where we ingest an item corresponding to seattle2020."""
        layer_config = RasterLayerConfig(
            LayerType.RASTER,
            [BandSetConfig(config_dict={}, dtype=DType.UINT8, bands=[TEST_BAND])],
        )
        query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)

        # In case rtree is enabled, use a small time range to minimize the time needed
        # to create the index.
        assert seattle2020.time_range is not None
        rtree_time_range = (
            seattle2020.time_range[0],
            seattle2020.time_range[0] + timedelta(days=3),
        )
        data_source = Sentinel2(
            config=layer_config, rtree_time_range=rtree_time_range, **kwargs
        )

        print("get items")
        item_groups = data_source.get_items([seattle2020], query_config)[0]
        item = item_groups[0][0]
        tile_store = DefaultTileStore(str(tile_store_dir))
        tile_store.set_dataset_path(tile_store_dir)
        layer_name = "layer"
        print("ingest")
        data_source.ingest(
            TileStoreWithLayer(tile_store, layer_name), item_groups[0], [[seattle2020]]
        )
        assert tile_store.is_raster_ready(layer_name, item.name, [TEST_BAND])

    @pytest.mark.parametrize("use_rtree_index", [False, True])
    def test_local(
        self, tmp_path: pathlib.Path, seattle2020: STGeometry, use_rtree_index: bool
    ) -> None:
        """Test ingesting to local filesystem."""
        tile_store_dir = UPath(tmp_path) / "tiles"
        tile_store_dir.mkdir(parents=True, exist_ok=True)
        index_cache_dir = UPath(tmp_path) / "cache"
        index_cache_dir.mkdir(parents=True, exist_ok=True)
        self.run_simple_test(
            tile_store_dir,
            seattle2020,
            index_cache_dir=index_cache_dir,
            use_rtree_index=use_rtree_index,
        )

    @pytest.mark.parametrize("use_rtree_index", [False, True])
    def test_gcs(self, seattle2020: STGeometry, use_rtree_index: bool) -> None:
        """Test ingesting to GCS.

        Main thing is to test index_cache_dir being on GCS.
        """
        test_id = random.randint(10000, 99999)
        bucket_name = os.environ["TEST_BUCKET"]
        prefix = os.environ["TEST_PREFIX"] + f"test_{test_id}/"
        test_path = UPath(f"gcs://{bucket_name}/{prefix}")
        tile_store_dir = test_path / "tiles"
        index_cache_dir = test_path / "cache"
        self.run_simple_test(
            tile_store_dir,
            seattle2020,
            index_cache_dir=index_cache_dir,
            use_rtree_index=use_rtree_index,
        )


@pytest.fixture
def sentinel2_without_rtree(tmp_path: pathlib.Path) -> Sentinel2:
    layer_config = RasterLayerConfig(
        LayerType.RASTER,
        [BandSetConfig(config_dict={}, dtype=DType.UINT8, bands=[TEST_BAND])],
    )
    sentinel2 = Sentinel2(
        config=layer_config,
        use_rtree_index=False,
        index_cache_dir=UPath(tmp_path),
    )
    return sentinel2


def test_prepare_antimeridian_no_matches(sentinel2_without_rtree: Sentinel2) -> None:
    # Make sure get_items works for scenes and geometries near +/- 180 longitude.
    # At (0, 40) there should be no Sentinel-2 coverage.
    time_range = (
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 2, 1, tzinfo=timezone.utc),
    )
    negative_geom = STGeometry(
        WGS84_PROJECTION, shapely.box(-179.99, 40.0, -179.9, 40.1), time_range
    )
    positive_geom = STGeometry(
        WGS84_PROJECTION, shapely.box(179.9, 40.0, 179.99, 40.1), time_range
    )
    query_config = QueryConfig(space_mode=SpaceMode.MOSAIC)
    groups = sentinel2_without_rtree.get_items(
        [negative_geom, positive_geom], query_config
    )
    for group in groups:
        assert len(group) == 0


def test_prepare_antimeridian_yes_matches(sentinel2_without_rtree: Sentinel2) -> None:
    # Make sure get_items works for scenes and geometries near 0 longitude.
    # At (0, 63) there should be some Sentinel-2 scenes.
    query_config = QueryConfig(space_mode=SpaceMode.MOSAIC)
    time_range = (
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 2, 1, tzinfo=timezone.utc),
    )
    negative_geom = STGeometry(
        WGS84_PROJECTION, shapely.box(-179.99, 63.0, -179.9, 63.1), time_range
    )
    positive_geom = STGeometry(
        WGS84_PROJECTION, shapely.box(179.9, 63.0, 179.99, 63.1), time_range
    )
    groups = sentinel2_without_rtree.get_items(
        [negative_geom, positive_geom], query_config
    )
    for group in groups:
        assert len(group) > 0 and len(group[0]) > 0


def test_product_with_missing_xml(sentinel2_without_rtree: Sentinel2) -> None:
    # Verify that the data source raises a MissingXMLException for products that have
    # a product folder in the GCS bucket but no metadata XML file.

    # This is an example product that has the issue.
    item_name = "S2A_MSIL1C_20150822T175026_N0204_R012_T11GLR_20170122T004950"

    with pytest.raises(MissingXMLException):
        sentinel2_without_rtree.get_item_by_name(item_name)


def test_search_intersecting_product_with_missing_xml(
    sentinel2_without_rtree: Sentinel2, tmp_path: pathlib.Path
) -> None:
    # Calling get_items on a cell/year that contains a product with missing XML issue
    # should NOT cause an error. Instead, the data source should skip those bad products.

    # This is an example product that has the issue.
    item_name = "S2A_MSIL1C_20150822T175026_N0204_R012_T11GLR_20170122T004950"

    # Get a geometry that intersects the product's cell.
    # To do so, we create it at the center of the bounds of the cell.
    parts = item_name.split("_")
    sense_time_str = parts[2]
    cell_id = parts[5][1:]
    sense_day = datetime(
        int(sense_time_str[0:4]),
        int(sense_time_str[4:6]),
        int(sense_time_str[6:8]),
        tzinfo=timezone.utc,
    )
    time_range = (sense_day, sense_day + timedelta(days=1))
    cell_bounds = get_sentinel2_tile_index()[cell_id]
    center = (
        (cell_bounds[0] + cell_bounds[2]) / 2,
        (cell_bounds[1] + cell_bounds[3]) / 2,
    )
    shp = shapely.Point(center[0], center[1]).buffer(0.001)
    geometry = STGeometry(WGS84_PROJECTION, shp, time_range)

    # Ensure it includes products with matching tile, but does not throw error.
    query_config = QueryConfig(space_mode=SpaceMode.MOSAIC, max_matches=1000)
    item_groups = sentinel2_without_rtree.get_items([geometry], query_config)[0]
    flat_items = [item for item_list in item_groups for item in item_list]
    matching_items = [
        item for item in flat_items if item.name.split("_")[5][1:] == cell_id
    ]
    assert len(matching_items) > 0


def test_product_with_missing_bands(sentinel2_without_rtree: Sentinel2) -> None:
    # Verify that the data source raises a CorruptItemException for products that are
    # missing some bands.

    # This is an example product that is missing B08.jp2.
    item_name = "S2A_MSIL1C_20170705T142751_N0205_R053_T20MPC_20170705T142752"

    with pytest.raises(CorruptItemException):
        sentinel2_without_rtree.get_item_by_name(item_name)
