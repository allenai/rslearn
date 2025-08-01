import os
import pathlib
import random
from typing import Any

import pytest
from upath import UPath

from rslearn.config import (
    QueryConfig,
    SpaceMode,
)
from rslearn.data_sources.google_earth_engine import GEE
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils import STGeometry

RUNNING_IN_CI = os.environ.get("CI", "false").lower() == "true"


@pytest.mark.skipif(
    RUNNING_IN_CI,
    reason="Skipping in CI environment as the memory requirements are too big",
)
class TestGEE:
    """Tests the GEE data source."""

    TEST_BAND = "VV"

    def run_simple_test(
        self, tile_store_dir: UPath, seattle2020: STGeometry, **kwargs: Any
    ) -> None:
        """Apply test where we ingest an item corresponding to seattle2020.

        Here we use Sentinel-1.
        """
        query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
        data_source = GEE(
            collection_name="COPERNICUS/S1_GRD",
            gcs_bucket_name=os.environ["TEST_BUCKET"],
            bands=[self.TEST_BAND],
            service_account_name=os.environ["TEST_SERVICE_ACCOUNT_NAME"],
            service_account_credentials=os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
            filters=[
                ("transmitterReceiverPolarisation", ["VV", "VH"]),
                ("instrumentMode", "IW"),
                (
                    "system:index",
                    "S1B_IW_GRDH_1SDV_20200724T142047_20200724T142112_022614_02AEB8_28B0",
                ),
            ],
            **kwargs,
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
        assert tile_store.is_raster_ready(layer_name, item.name, [self.TEST_BAND])

    def test_local(self, tmp_path: pathlib.Path, seattle2020: STGeometry) -> None:
        """Test ingesting to local filesystem."""
        tile_store_dir = UPath(tmp_path) / "tiles"
        tile_store_dir.mkdir(parents=True, exist_ok=True)
        index_cache_dir = UPath(tmp_path) / "cache"
        index_cache_dir.mkdir(parents=True, exist_ok=True)
        self.run_simple_test(
            tile_store_dir,
            seattle2020,
            index_cache_dir=index_cache_dir,
        )

    def test_gcs(self, seattle2020: STGeometry) -> None:
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
        )
