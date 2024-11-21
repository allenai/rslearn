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
from rslearn.data_sources.gcp_public_data import Sentinel2
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


def test_prepare_antimeridian_no_matches(tmp_path: pathlib.Path) -> None:
    # Make sure get_items works for scenes and geometries near +/- 180 longitude.
    # At (0, 40) there should be no Sentinel-2 coverage.
    layer_config = RasterLayerConfig(
        LayerType.RASTER,
        [BandSetConfig(config_dict={}, dtype=DType.UINT8, bands=[TEST_BAND])],
    )
    query_config = QueryConfig(space_mode=SpaceMode.MOSAIC)
    data_source = Sentinel2(
        config=layer_config,
        use_rtree_index=False,
        index_cache_dir=UPath(tmp_path),
    )
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
    groups = data_source.get_items([negative_geom, positive_geom], query_config)
    for group in groups:
        assert len(group) == 0


def test_prepare_antimeridian_yes_matches(tmp_path: pathlib.Path) -> None:
    # Make sure get_items works for scenes and geometries near 0 longitude.
    # At (0, 63) there should be some Sentinel-2 scenes.
    layer_config = RasterLayerConfig(
        LayerType.RASTER,
        [BandSetConfig(config_dict={}, dtype=DType.UINT8, bands=[TEST_BAND])],
    )
    query_config = QueryConfig(space_mode=SpaceMode.MOSAIC)
    data_source = Sentinel2(
        config=layer_config,
        use_rtree_index=False,
        index_cache_dir=UPath(tmp_path),
    )
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
    groups = data_source.get_items([negative_geom, positive_geom], query_config)
    for group in groups:
        assert len(group) > 0 and len(group[0]) > 0
