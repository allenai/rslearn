import os
import pathlib
import random
from datetime import timedelta

import pytest
from upath import UPath

from rslearn.config import (
    BandSetConfig,
    DType,
    LayerType,
    QueryConfig,
    RasterLayerConfig,
    SpaceMode,
)
from rslearn.data_sources.aws_open_data import Naip, Sentinel2, Sentinel2Modality
from rslearn.tile_stores import FileTileStore
from rslearn.utils import STGeometry


class TestNaip:
    """Tests the Naip data source."""

    TEST_BANDS = ["R", "G", "B", "IR"]

    def run_simple_test(self, tile_store_dir: UPath, seattle2020: STGeometry, **kwargs):
        """Apply test where we ingest an item corresponding to seattle2020."""
        layer_config = RasterLayerConfig(
            LayerType.RASTER,
            [BandSetConfig(config_dict={}, dtype=DType.UINT8, bands=self.TEST_BANDS)],
        )
        query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
        data_source = Naip(config=layer_config, states=["wa"], years=[2019], **kwargs)

        # Expand time range since NAIP isn't available very frequently.
        seattle2020.time_range = (
            seattle2020.time_range[0] - timedelta(days=500),
            seattle2020.time_range[1] + timedelta(days=500),
        )

        print("get items")
        item_groups = data_source.get_items([seattle2020], query_config)[0]
        item = item_groups[0][0]
        tile_store = FileTileStore(tile_store_dir)
        print("ingest")
        data_source.ingest(tile_store, item_groups[0], [[seattle2020]])
        expected_path = (
            tile_store_dir
            / item.name
            / "_".join(self.TEST_BANDS)
            / str(seattle2020.projection)
            / "geotiff.tif"
        )
        assert expected_path.exists()

    @pytest.mark.parametrize("use_rtree_index", [False, True])
    def test_local(
        self, tmp_path: pathlib.Path, seattle2020: STGeometry, use_rtree_index: bool
    ):
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
    def test_gcs(self, seattle2020: STGeometry, use_rtree_index: bool):
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


class TestSentinel2:
    """Tests the Naip data source."""

    TEST_BAND = "B04"
    TEST_MODALITY = Sentinel2Modality.L1C

    def run_simple_test(
        self, tile_store_dir: UPath, metadata_cache_dir: UPath, seattle2020: STGeometry
    ):
        """Apply test where we ingest an item corresponding to seattle2020."""
        layer_config = RasterLayerConfig(
            LayerType.RASTER,
            [BandSetConfig(config_dict={}, dtype=DType.UINT8, bands=[self.TEST_BAND])],
        )
        query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
        data_source = Sentinel2(
            config=layer_config,
            metadata_cache_dir=metadata_cache_dir,
            modality=self.TEST_MODALITY,
        )
        print("get items")
        item_groups = data_source.get_items([seattle2020], query_config)[0]
        item = item_groups[0][0]
        tile_store = FileTileStore(tile_store_dir)
        print("ingest")
        data_source.ingest(tile_store, item_groups[0], [[seattle2020]])
        expected_path = (
            tile_store_dir
            / item.name
            / self.TEST_BAND
            / str(seattle2020.projection)
            / "geotiff.tif"
        )
        assert expected_path.exists()

    def test_local(self, tmp_path: pathlib.Path, seattle2020: STGeometry):
        """Test ingesting to local filesystem."""
        tile_store_dir = UPath(tmp_path) / "tiles"
        tile_store_dir.mkdir(parents=True, exist_ok=True)
        metadata_cache_dir = UPath(tmp_path) / "cache"
        metadata_cache_dir.mkdir(parents=True, exist_ok=True)
        self.run_simple_test(tile_store_dir, metadata_cache_dir, seattle2020)

    def test_gcs(self, seattle2020: STGeometry):
        """Test ingesting to GCS.

        Main thing is to test metadata_cache_dir being on GCS.
        """
        test_id = random.randint(10000, 99999)
        bucket_name = os.environ["TEST_BUCKET"]
        prefix = os.environ["TEST_PREFIX"] + f"test_{test_id}/"
        test_path = UPath(f"gcs://{bucket_name}/{prefix}")
        tile_store_dir = test_path / "tiles"
        metadata_cache_dir = test_path / "cache"
        self.run_simple_test(tile_store_dir, metadata_cache_dir, seattle2020)