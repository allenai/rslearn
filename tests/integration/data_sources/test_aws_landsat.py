import pathlib
from datetime import timedelta

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
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
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
        tile_store = DefaultTileStore(str(tile_store_dir))
        tile_store.set_dataset_path(tile_store_dir)
        layer_name = "layer"
        print("ingest")
        data_source.ingest(
            TileStoreWithLayer(tile_store, layer_name), item_groups[0], [[seattle2020]]
        )
        assert tile_store.is_raster_ready(layer_name, item.name, [TEST_BAND])

    def test_local(self, tmp_path: pathlib.Path, seattle2020: STGeometry) -> None:
        """Test ingesting to local filesystem."""
        tile_store_dir = UPath(tmp_path) / "tiles"
        tile_store_dir.mkdir(parents=True, exist_ok=True)
        metadata_cache_dir = UPath(tmp_path) / "cache"
        self.run_simple_test(tile_store_dir, metadata_cache_dir, seattle2020)

    def test_cloud_cover_sorting(
        self, tmp_path: pathlib.Path, seattle2020: STGeometry
    ) -> None:
        """Verify that data source sorts by cloud cover correctly when requested."""
        # Adjust the time range to have enough time for 10 images.
        # Need assert for type checker to be happy.
        assert seattle2020.time_range is not None
        seattle2020.time_range = (
            seattle2020.time_range[0],
            seattle2020.time_range[0] + timedelta(days=120),
        )

        # Initialize the data source and perform the get_items query.
        layer_config = RasterLayerConfig(
            LayerType.RASTER,
            [BandSetConfig(config_dict={}, dtype=DType.UINT8, bands=[TEST_BAND])],
        )
        query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS, max_matches=10)
        data_source = LandsatOliTirs(
            config=layer_config,
            metadata_cache_dir=UPath(tmp_path),
            sort_by="cloud_cover",
        )
        item_groups = data_source.get_items([seattle2020], query_config)[0]

        # Verify the result.
        # There should be enough images matching this geometry that we exhaust max_matches.
        assert len(item_groups) == 10
        # And they should be ordered by cloud cover.
        cloud_covers = [group[0].cloud_cover for group in item_groups]
        sorted_cloud_covers = list(sorted(cloud_covers))
        assert cloud_covers == sorted_cloud_covers
