import os
import pathlib
import random
from typing import Any

from upath import UPath

from rslearn.config import (
    BandSetConfig,
    DType,
    LayerType,
    QueryConfig,
    RasterLayerConfig,
    SpaceMode,
)
from rslearn.data_sources.usgs_landsat import LandsatOliTirs
from rslearn.tile_stores import FileTileStore
from rslearn.utils import STGeometry


class TestLandsatOliTirs:
    """Tests the LandsatOliTirs data source."""

    TEST_BAND = "B8"

    def run_simple_test(
        self, tile_store_dir: UPath, seattle2020: STGeometry, **kwargs: Any
    ) -> None:
        """Apply test where we ingest an item corresponding to seattle2020."""
        layer_config = RasterLayerConfig(
            LayerType.RASTER,
            [BandSetConfig(config_dict={}, dtype=DType.UINT8, bands=[self.TEST_BAND])],
        )
        query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
        data_source = LandsatOliTirs(
            config=layer_config,
            username=os.environ["TEST_USGS_LANDSAT_USERNAME"],
            password=os.environ["TEST_USGS_LANDSAT_PASSWORD"],
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

    def test_local(self, tmp_path: pathlib.Path, seattle2020: STGeometry) -> None:
        """Test ingesting to local filesystem."""
        tile_store_dir = UPath(tmp_path) / "tiles"
        tile_store_dir.mkdir(parents=True, exist_ok=True)
        self.run_simple_test(
            tile_store_dir,
            seattle2020,
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
        self.run_simple_test(
            tile_store_dir,
            seattle2020,
        )
