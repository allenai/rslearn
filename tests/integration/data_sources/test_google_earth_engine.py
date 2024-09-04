import os
import pathlib
import random

from upath import UPath

from rslearn.config import (
    BandSetConfig,
    DType,
    LayerType,
    QueryConfig,
    RasterLayerConfig,
    SpaceMode,
)
from rslearn.data_sources.google_earth_engine import GEE
from rslearn.tile_stores import FileTileStore
from rslearn.utils import STGeometry


class TestGEE:
    """Tests the GEE data source."""

    TEST_BAND = "VV"

    def run_simple_test(self, tile_store_dir: UPath, seattle2020: STGeometry, **kwargs):
        """Apply test where we ingest an item corresponding to seattle2020.

        Here we use Sentinel-1.
        """
        layer_config = RasterLayerConfig(
            LayerType.RASTER,
            [BandSetConfig(config_dict={}, dtype=DType.UINT8, bands=[self.TEST_BAND])],
        )
        query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
        data_source = GEE(
            config=layer_config,
            collection_name="COPERNICUS/S1_GRD",
            gcs_bucket_name=os.environ["TEST_GEE_BUCKET"],
            service_account_name=os.environ["TEST_GEE_SERVICE_ACCOUNT_NAME"],
            service_account_credentials=os.environ[
                "TEST_GEE_SERVICE_ACCOUNT_CREDENTIALS"
            ],
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
        index_cache_dir = UPath(tmp_path) / "cache"
        index_cache_dir.mkdir(parents=True, exist_ok=True)
        self.run_simple_test(
            tile_store_dir,
            seattle2020,
            index_cache_dir=index_cache_dir,
        )

    def test_gcs(self, seattle2020: STGeometry):
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
