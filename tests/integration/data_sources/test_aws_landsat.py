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
from rslearn.data_sources.aws_landsat import LandsatOliTirs
from rslearn.tile_stores import FileTileStore
from rslearn.utils import STGeometry

TEST_BAND = "B8"


class TestLandsatOliTirs:
    """Tests the LandsatOliTirs data source."""

    def run_simple_test(
        self, tile_store_dir: UPath, metadata_cache_dir: UPath, seattle2020: STGeometry
    ) -> None:
        """Apply test where we ingest an item corresponding to seattle2020."""
        layer_config = RasterLayerConfig(
            LayerType.RASTER,
            [BandSetConfig(config_dict={}, dtype=DType.UINT8, bands=[TEST_BAND])],
        )
        query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
        data_source = LandsatOliTirs(
            config=layer_config, metadata_cache_dir=metadata_cache_dir
        )
        print("get items")
        item_groups = data_source.get_items([seattle2020], query_config)[0]  # type: ignore
        item = item_groups[0][0]
        tile_store = FileTileStore(tile_store_dir)
        print("ingest")
        data_source.ingest(tile_store, item_groups[0], [[seattle2020]])
        expected_path = (
            tile_store_dir
            / item.name
            / TEST_BAND
            / str(seattle2020.projection)
            / "geotiff.tif"
        )
        assert expected_path.exists()

    def test_local(self, tmp_path: pathlib.Path, seattle2020: STGeometry) -> None:
        """Test ingesting to local filesystem."""
        tile_store_dir = UPath(tmp_path) / "tiles"
        tile_store_dir.mkdir(parents=True, exist_ok=True)
        metadata_cache_dir = UPath(tmp_path) / "cache"
        self.run_simple_test(tile_store_dir, metadata_cache_dir, seattle2020)

    def test_gcs(self, seattle2020: STGeometry) -> None:
        """Test ingesting to GCS.

        Main thing is to test metadata_cache_dir being on GCS.
        """
        test_id = random.randint(10000, 99999)
        test_bucket = os.environ["TEST_BUCKET"]
        test_id_prefix = f"test_{test_id}/"
        test_path = UPath(f"gs://{test_bucket}/{test_id_prefix}")
        tile_store_dir = test_path / "tiles"
        metadata_cache_dir = test_path / "cache"
        self.run_simple_test(tile_store_dir, metadata_cache_dir, seattle2020)
