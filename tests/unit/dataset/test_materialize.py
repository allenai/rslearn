import pathlib

import numpy as np
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset.materialize import read_raster_window_from_tiles
from rslearn.tile_stores.default import DefaultTileStore
from rslearn.tile_stores.tile_store import TileStoreWithLayer


class TestReadRasterWindowFromTiles:
    """Unit tests for read_raster_window_from_tiles."""

    LAYER_NAME = "layer"
    ITEM_NAME = "item"
    BANDS = ["band"]
    BOUNDS = (0, 0, 4, 4)

    def test_basic_mosaic(self, tmp_path: pathlib.Path) -> None:
        """Make sure mosaics are processed correctly.

        We create dst covering top half and src covering entire image and make sure
        that the bottom half of src is copied.
        """
        tile_store = DefaultTileStore()
        tile_store.set_dataset_path(UPath(tmp_path))
        src = 2 * np.ones((1, 4, 4), dtype=np.uint8)
        tile_store.write_raster(
            self.LAYER_NAME,
            self.ITEM_NAME,
            self.BANDS,
            WGS84_PROJECTION,
            self.BOUNDS,
            src,
        )

        dst = np.zeros((1, 4, 4), dtype=np.uint8)
        dst[0, 0:2, 0:4] = 1
        read_raster_window_from_tiles(
            dst=dst,
            tile_store=TileStoreWithLayer(tile_store, self.LAYER_NAME),
            item_name=self.ITEM_NAME,
            bands=self.BANDS,
            projection=WGS84_PROJECTION,
            bounds=self.BOUNDS,
            src_indexes=[0],
            dst_indexes=[0],
            nodata_vals=[0],
        )
        assert np.all(dst[0, 0:2, 0:4] == 1)
        assert np.all(dst[0, 2:4, 0:4] == 2)

    def test_nodata(self, tmp_path: pathlib.Path) -> None:
        """Test nodata handling.

        Now we use two bands with different nodata values. We verify that the dst is
        only overwritten when both bands are the nodata value.
        """
        tile_store = DefaultTileStore()
        tile_store.set_dataset_path(UPath(tmp_path))
        src = 3 * np.ones((2, 4, 4), dtype=np.uint8)
        tile_store.write_raster(
            self.LAYER_NAME,
            self.ITEM_NAME,
            self.BANDS,
            WGS84_PROJECTION,
            self.BOUNDS,
            src,
        )

        nodata_vals = [1.0, 2.0]
        dst = np.zeros((2, 4, 4), dtype=np.uint8)
        # Set top 1 and left 2.
        # So then only topleft has both bands matching nodata.
        dst[:, 0:2, 0:4] = nodata_vals[0]
        dst[:, 0:4, 0:2] = nodata_vals[1]
        read_raster_window_from_tiles(
            dst=dst,
            tile_store=TileStoreWithLayer(tile_store, self.LAYER_NAME),
            item_name=self.ITEM_NAME,
            bands=self.BANDS,
            projection=WGS84_PROJECTION,
            bounds=self.BOUNDS,
            src_indexes=[0, 1],
            dst_indexes=[0, 1],
            nodata_vals=nodata_vals,
        )
        assert np.all(dst[:, 0:2, 2:4] == nodata_vals[0])
        assert np.all(dst[:, 2:4, 0:2] == nodata_vals[1])
        assert np.all(dst[:, 0:2, 0:2] == 3)
        assert np.all(dst[:, 2:4, 2:4] == 0)
