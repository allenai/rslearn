import os
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
from rslearn.data_sources.planet import Planet
from rslearn.tile_stores import FileTileStore
from rslearn.utils import STGeometry

RUNNING_IN_CI = os.environ.get("CI", "false").lower() == "true"


@pytest.mark.skipif(RUNNING_IN_CI, reason="Skipping in CI environment")
class TestPlanet:
    """Tests the Planet data source."""

    TEST_BANDS = ["b01", "b02", "b03", "b04", "b05", "b06", "b07", "b08"]

    def test_simple(self, tmp_path: pathlib.Path, seattle2020: STGeometry):
        """Apply test where we ingest an item corresponding to seattle2020."""
        tile_store_dir = UPath(tmp_path)
        layer_config = RasterLayerConfig(
            LayerType.RASTER,
            [BandSetConfig(config_dict={}, dtype=DType.UINT8, bands=self.TEST_BANDS)],
        )
        query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
        data_source = Planet(
            config=layer_config,
            item_type_id="PSScene",
            asset_type_id="ortho_analytic_8b_sr",
            bands=self.TEST_BANDS,
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
            / "_".join(self.TEST_BANDS)
            / str(seattle2020.projection)
            / "geotiff.tif"
        )
        assert expected_path.exists()
