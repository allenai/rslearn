import pathlib

from upath import UPath

from rslearn.config import (
    BandSetConfig,
    DType,
    LayerType,
    QueryConfig,
    RasterLayerConfig,
    SpaceMode,
)
from rslearn.data_sources.azure_sentinel1 import Sentinel1
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils import STGeometry


class TestSentinel1:
    """Tests the Sentinel1 data source."""

    def test_local(
        self, tmp_path: pathlib.Path, seattle2020: STGeometry, use_rtree_index: bool
    ) -> None:
        """Test ingesting an item corresponding to seattle2020 to local filesystem."""
        layer_config = RasterLayerConfig(
            LayerType.RASTER,
            [BandSetConfig(config_dict={}, dtype=DType.UINT8, bands=["vv"])],
        )
        query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
        s1_query_dict = {"sar:polarizations": ["VV", "VH"]}
        data_source = Sentinel1(
            config=layer_config,
            query=s1_query_dict,
        )

        print("get items")
        item_groups = data_source.get_items([seattle2020], query_config)[0]
        item = item_groups[0][0]

        tile_store_dir = UPath(tmp_path)
        tile_store = DefaultTileStore(str(tile_store_dir))
        tile_store.set_dataset_path(tile_store_dir)

        print("ingest")
        layer_name = "layer"
        data_source.ingest(
            TileStoreWithLayer(tile_store, layer_name), item_groups[0], [[seattle2020]]
        )
        assert tile_store.is_raster_ready(layer_name, item.name, ["vv"])
