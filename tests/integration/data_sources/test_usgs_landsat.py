import os
import pathlib
from typing import Any

import pytest
from upath import UPath

from rslearn.config import (
    BandSetConfig,
    DataSourceConfig,
    DType,
    LayerType,
    QueryConfig,
    RasterLayerConfig,
    SpaceMode,
)
from rslearn.data_sources.usgs_landsat import LandsatOliTirs
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
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
            token=os.environ["TEST_USGS_LANDSAT_TOKEN"],
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
        self.run_simple_test(
            tile_store_dir,
            seattle2020,
        )


class TestLoadFromConfig:
    """Tests LandsatOliTirs.load_from_config."""

    TEST_BAND = "B8"

    def test_config_missing_password_or_token(self, tmp_path: pathlib.Path) -> None:
        data_source_config = DataSourceConfig(
            "landsat",
            QueryConfig(),
            dict(
                username=os.environ["TEST_USGS_LANDSAT_USERNAME"],
            ),
        )
        config = RasterLayerConfig(
            LayerType.RASTER,
            [BandSetConfig(config_dict={}, dtype=DType.UINT8, bands=[self.TEST_BAND])],
            data_source=data_source_config,
        )
        with pytest.raises(ValueError):
            LandsatOliTirs.from_config(config, UPath(tmp_path))

    def test_okay_config(self, tmp_path: pathlib.Path) -> None:
        data_source_config = DataSourceConfig(
            "landsat",
            QueryConfig(),
            dict(
                username=os.environ["TEST_USGS_LANDSAT_USERNAME"],
                token=os.environ["TEST_USGS_LANDSAT_TOKEN"],
            ),
        )
        config = RasterLayerConfig(
            LayerType.RASTER,
            [BandSetConfig(config_dict={}, dtype=DType.UINT8, bands=[self.TEST_BAND])],
            data_source=data_source_config,
        )
        LandsatOliTirs.from_config(config, UPath(tmp_path))
