"""Test rslearn.data_sources.copernicus."""

import pathlib

import numpy as np
import pytest
from rasterio.enums import Resampling
from upath import UPath

from rslearn.config import (
    QueryConfig,
    SpaceMode,
)
from rslearn.data_sources.copernicus import Copernicus, Sentinel2, Sentinel2ProductType
from rslearn.log_utils import get_logger
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils import STGeometry

logger = get_logger(__name__)


class TestCopernicus:
    """Tests for Copernicus data source."""

    def test_ingest_sentinel2(
        self, tmp_path: pathlib.Path, seattle2020: STGeometry
    ) -> None:
        """Test ingesting a Sentinel-2 scene corresponding to seattle2020."""
        band_names = ["R", "G", "B"]
        data_source = Copernicus(
            glob_to_bands={
                "*/GRANULE/*/IMG_DATA/*_TCI.jp2": band_names,
            },
            query_filter="Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'S2MSI1C')",
        )

        logger.info("get items")
        query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
        item_groups = data_source.get_items([seattle2020], query_config)[0]
        item = item_groups[0][0]

        tile_store_dir = UPath(tmp_path)
        tile_store = DefaultTileStore(str(tile_store_dir))
        tile_store.set_dataset_path(tile_store_dir)

        logger.info(f"ingest item {item.name}")
        layer_name = "layer"
        data_source.ingest(
            TileStoreWithLayer(tile_store, layer_name), item_groups[0], [[seattle2020]]
        )
        assert tile_store.is_raster_ready(layer_name, item.name, band_names)


class TestSentinel2:
    """Tests for Sentinel2 data source."""

    @pytest.mark.parametrize("harmonize", [False, True])
    def test_harmonize(
        self, harmonize: bool, tmp_path: pathlib.Path, seattle2020: STGeometry
    ) -> None:
        """Verify that we get correct pixel values with harmonization."""
        data_source = Sentinel2(
            assets=["B04", "TCI"],
            product_type=Sentinel2ProductType.L2A,
            harmonize=harmonize,
        )

        logger.info("get items")
        query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
        item_groups = data_source.get_items([seattle2020], query_config)[0]
        item = item_groups[0][0]

        tile_store_dir = UPath(tmp_path)
        tile_store = DefaultTileStore(str(tile_store_dir))
        tile_store.set_dataset_path(tile_store_dir)

        logger.info(f"ingest item {item.name}")
        layer_name = "layer"
        data_source.ingest(
            TileStoreWithLayer(tile_store, layer_name), item_groups[0], [[seattle2020]]
        )

        # So with harmonization, B04 should be about 10 times of TCI.
        # But only if it isn't too dark or too bright (since TCI loses some of the
        # dynamic range).
        bounds = (
            int(seattle2020.shp.bounds[0]),
            int(seattle2020.shp.bounds[1]),
            int(seattle2020.shp.bounds[2]),
            int(seattle2020.shp.bounds[3]),
        )
        b04 = tile_store.read_raster(
            layer_name,
            item.name,
            ["B04"],
            seattle2020.projection,
            bounds,
            resampling=Resampling.nearest,
        )[0, :, :]
        red = tile_store.read_raster(
            layer_name,
            item.name,
            ["R", "G", "B"],
            seattle2020.projection,
            bounds,
            resampling=Resampling.nearest,
        )[0, :, :].astype(np.uint16)
        check_array = (b04 > red * 8) & (b04 < red * 12)
        count = np.count_nonzero((~check_array) & (b04 > 500) & (b04 < 2500))

        if harmonize:
            assert count == 0
        else:
            assert count > 1000
