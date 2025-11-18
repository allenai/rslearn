import os
import pathlib

from upath import UPath

from rslearn.config import (
    BandSetConfig,
    DType,
    LayerConfig,
    LayerType,
    QueryConfig,
    SpaceMode,
)
from rslearn.data_sources.xyz_tiles import XyzTiles
from rslearn.dataset import Window
from rslearn.utils import STGeometry


class TestXyzTiles:
    """Tests the XyzTiles data source."""

    TEST_BANDS = ["R", "G", "B"]

    def run_simple_test(self, dst_dir: UPath, seattle2020: STGeometry) -> None:
        """Apply test where we ingest an item corresponding to seattle2020."""
        layer_config = LayerConfig(
            layer_type=LayerType.RASTER,
            band_sets=[BandSetConfig(dtype=DType.UINT8, bands=self.TEST_BANDS)],
        )
        query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
        assert seattle2020.time_range is not None
        data_source = XyzTiles(
            url_templates=[os.environ["TEST_XYZ_TILES_TEMPLATE"]],
            time_ranges=[seattle2020.time_range],
            zoom=13,
        )
        item_groups = data_source.get_items([seattle2020], query_config)[0]
        print(item_groups)
        window = Window(
            path=dst_dir,
            group="default",
            name="default",
            projection=seattle2020.projection,
            bounds=tuple([int(x) for x in seattle2020.shp.bounds]),  # type: ignore
            time_range=seattle2020.time_range,
        )
        window.save()
        print("materialize")
        data_source.materialize(window, item_groups, "raster", layer_config)
        expected_path = window.get_raster_dir("raster", self.TEST_BANDS) / "geotiff.tif"
        assert expected_path.exists()

    def test_local(self, tmp_path: pathlib.Path, seattle2020: STGeometry) -> None:
        """Test ingesting to local filesystem."""
        dst_dir = UPath(tmp_path)
        self.run_simple_test(dst_dir, seattle2020)
