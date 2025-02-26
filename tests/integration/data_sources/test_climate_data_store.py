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
    TimeMode,
)
from rslearn.data_sources.climate_data_store import ERA5LandMonthlyMeans
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils import STGeometry


class TestERA5LandMonthlyMeans:
    """Tests the ERA5LandMonthlyMeans data source from the Climate Data Store."""

    TEST_BANDS = ["2m_temperature", "total_precipitation"]

    def run_simple_test(self, tile_store_dir: UPath, seattle2020: STGeometry) -> None:
        """Apply test where we ingest an item corresponding to seattle2020."""
        # rslearn band names, remove "_"
        band_names = [band.replace("_", " ") for band in self.TEST_BANDS]
        layer_config = RasterLayerConfig(
            LayerType.RASTER,
            [BandSetConfig(config_dict={}, dtype=DType.FLOAT32, bands=self.TEST_BANDS)],
        )
        query_config = QueryConfig(
            space_mode=SpaceMode.INTERSECTS,
            time_mode=TimeMode.WITHIN,
            max_matches=2,  # We expect two items to match
        )
        data_source = ERA5LandMonthlyMeans(config=layer_config)
        print("get items")
        item_groups = data_source.get_items([seattle2020], query_config)[0]  # type: ignore
        item_0 = item_groups[0][0]
        item_1 = item_groups[1][0]
        tile_store = DefaultTileStore(str(tile_store_dir))
        tile_store.set_dataset_path(tile_store_dir)
        layer_name = "layer"
        print("ingest")
        data_source.ingest(
            TileStoreWithLayer(tile_store, layer_name), item_groups[0], [[seattle2020]]
        )
        data_source.ingest(
            TileStoreWithLayer(tile_store, layer_name), item_groups[1], [[seattle2020]]
        )
        assert tile_store.is_raster_ready(layer_name, item_0.name, band_names)
        assert tile_store.is_raster_ready(layer_name, item_1.name, band_names)

    def test_local(self, tmp_path: pathlib.Path, seattle2020: STGeometry) -> None:
        """Test ingesting to local filesystem."""
        tile_store_dir = UPath(tmp_path) / "tiles"
        tile_store_dir.mkdir(parents=True, exist_ok=True)
        self.run_simple_test(tile_store_dir, seattle2020)

    def test_gcs(self, seattle2020: STGeometry) -> None:
        """Test ingesting to GCS."""
        test_id = random.randint(10000, 99999)
        bucket_name = os.environ["TEST_BUCKET"]
        prefix = os.environ["TEST_PREFIX"]
        test_path = UPath(f"gcs://{bucket_name}/{prefix}/test_{test_id}")
        tile_store_dir = test_path / "tiles"
        self.run_simple_test(tile_store_dir, seattle2020)
