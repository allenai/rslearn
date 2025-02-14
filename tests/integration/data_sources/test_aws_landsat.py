import pathlib

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
from rslearn.data_sources.aws_landsat import LandsatOliTirs
from rslearn.dataset import Window
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils import STGeometry

TEST_BAND = "B8"


class TestLandsatOliTirs:
    """Tests the LandsatOliTirs data source."""

    @pytest.fixture
    def landsat_layer_config(self) -> RasterLayerConfig:
        return RasterLayerConfig(
            LayerType.RASTER,
            [BandSetConfig(config_dict={}, dtype=DType.UINT8, bands=[TEST_BAND])],
        )

    @pytest.fixture
    def landsat_data_source(
        self, tmp_path: pathlib.Path, landsat_layer_config: RasterLayerConfig
    ) -> LandsatOliTirs:
        return LandsatOliTirs(
            config=landsat_layer_config, metadata_cache_dir=UPath(tmp_path)
        )

    def test_ingest(
        self,
        tmp_path: pathlib.Path,
        seattle2020: STGeometry,
        landsat_data_source: LandsatOliTirs,
    ) -> None:
        """Test ingesting to local filesystem."""
        tile_store_dir = UPath(tmp_path)

        print("get items")
        query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
        item_groups = landsat_data_source.get_items([seattle2020], query_config)[0]
        item = item_groups[0][0]
        tile_store = DefaultTileStore(str(tile_store_dir))
        tile_store.set_dataset_path(tile_store_dir)
        layer_name = "layer"
        print("ingest")
        landsat_data_source.ingest(
            TileStoreWithLayer(tile_store, layer_name), item_groups[0], [[seattle2020]]
        )
        assert tile_store.is_raster_ready(layer_name, item.name, [TEST_BAND])

    def test_materialize(
        self,
        tmp_path: pathlib.Path,
        seattle2020: STGeometry,
        landsat_data_source: LandsatOliTirs,
        landsat_layer_config: RasterLayerConfig,
    ) -> None:
        """Test directly materializing from the data source."""
        ds_path = UPath(tmp_path)
        group = "default"
        window_name = "default"
        bounds = (
            int(seattle2020.shp.bounds[0]),
            int(seattle2020.shp.bounds[1]),
            int(seattle2020.shp.bounds[2]),
            int(seattle2020.shp.bounds[3]),
        )
        window = Window(
            path=Window.get_window_root(ds_path, group, window_name),
            group=group,
            name=window_name,
            projection=seattle2020.projection,
            bounds=bounds,
            time_range=seattle2020.time_range,
        )
        window.save()
        query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
        item_groups = landsat_data_source.get_items([seattle2020], query_config)[0]
        landsat_data_source.materialize(
            window, item_groups, "landsat", landsat_layer_config
        )
        assert window.is_layer_completed("landsat")
