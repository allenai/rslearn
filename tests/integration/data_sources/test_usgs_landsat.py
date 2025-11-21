import os
import pathlib

from upath import UPath

from rslearn.config import (
    QueryConfig,
    SpaceMode,
)
from rslearn.data_sources.usgs_landsat import LandsatOliTirs
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils import STGeometry


class TestLandsatOliTirs:
    """Tests the LandsatOliTirs data source."""

    TEST_BAND = "B8"

    def test_local(self, tmp_path: pathlib.Path, seattle2020: STGeometry) -> None:
        """Apply test where we ingest an item corresponding to seattle2020."""
        tile_store_dir = UPath(tmp_path) / "tiles"
        tile_store_dir.mkdir(parents=True, exist_ok=True)

        query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
        data_source = LandsatOliTirs(
            username=os.environ["TEST_USGS_LANDSAT_USERNAME"],
            token=os.environ["TEST_USGS_LANDSAT_TOKEN"],
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
