"""Tests for rslearn.dataset.tile_utils (read_raster_window_from_tiles)."""

import pathlib

import numpy as np
from shapely.geometry import Polygon
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.data_source import Item
from rslearn.dataset.materialize import resolve_nodata_values
from rslearn.dataset.tile_utils import read_raster_window_from_tiles
from rslearn.tile_stores.default import DefaultTileStore
from rslearn.tile_stores.tile_store import TileStoreWithLayer
from rslearn.utils.geometry import STGeometry
from rslearn.utils.raster_array import RasterArray, RasterMetadata


class TestReadRasterWindowFromTiles:
    """Unit tests for read_raster_window_from_tiles.

    It should merge data from the new item into a `dst: RasterArray`, only overwriting
    dst at pixels where all bands are nodata.
    """

    LAYER_NAME = "layer"
    BANDS = ["band1", "band2"]
    BOUNDS = (0, 0, 4, 4)
    PROJECTION = WGS84_PROJECTION

    def _make_item(self, name: str = "item") -> Item:
        bbox = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        return Item(name, STGeometry(self.PROJECTION, bbox, None))

    def test_basic_mosaic(self, tmp_path: pathlib.Path) -> None:
        """Make sure mosaics are processed correctly.

        We create dst covering top half and src covering entire image and make sure
        that only the bottom half of src is copied.
        """
        tile_store = DefaultTileStore()
        tile_store.set_dataset_path(UPath(tmp_path))
        item = self._make_item()
        bands = ["band1"]
        # Make src value 2 everywhere.
        src = 2 * np.ones((1, 4, 4), dtype=np.uint8)
        tile_store.write_raster(
            self.LAYER_NAME,
            item,
            bands,
            self.PROJECTION,
            self.BOUNDS,
            RasterArray(chw_array=src),
        )

        # Pre-fill top half with 1 so only bottom half is nodata (0).
        dst_arr = np.zeros((1, 1, 4, 4), dtype=np.uint8)
        dst_arr[0, 0, 0:2, 0:4] = 1
        dst = RasterArray(array=dst_arr)
        read_raster_window_from_tiles(
            tile_store=TileStoreWithLayer(tile_store, self.LAYER_NAME),
            item=item,
            bands=bands,
            projection=self.PROJECTION,
            bounds=self.BOUNDS,
            nodata_vals=(0,),
            band_dtype=np.uint8,
            dst=dst,
        )
        result = dst.get_chw_array()
        # Top half should be unchanged.
        assert np.all(result[0, 0:2, 0:4] == 1)
        # Bottom half should be updated to match src.
        assert np.all(result[0, 2:4, 0:4] == 2)

    def test_nodata(self, tmp_path: pathlib.Path) -> None:
        """Test nodata handling.

        Now we use two bands with different nodata values. We verify that the dst is
        only overwritten when both bands are the nodata value.
        """
        tile_store = DefaultTileStore()
        tile_store.set_dataset_path(UPath(tmp_path))
        item = self._make_item()
        src = 3 * np.ones((2, 4, 4), dtype=np.uint8)
        tile_store.write_raster(
            self.LAYER_NAME,
            item,
            self.BANDS,
            self.PROJECTION,
            self.BOUNDS,
            RasterArray(chw_array=src),
        )

        nodata_vals = (1.0, 2.0)
        dst_arr = np.zeros((2, 1, 4, 4), dtype=np.uint8)
        # Set first band 1 in top half, and second band 2 in left half.
        # So then only topleft has both bands matching nodata.
        dst_arr[0, 0, 0:2, 0:4] = nodata_vals[0]
        dst_arr[1, 0, 0:4, 0:2] = nodata_vals[1]
        dst = RasterArray(array=dst_arr)
        read_raster_window_from_tiles(
            tile_store=TileStoreWithLayer(tile_store, self.LAYER_NAME),
            item=item,
            bands=self.BANDS,
            projection=self.PROJECTION,
            bounds=self.BOUNDS,
            nodata_vals=nodata_vals,
            band_dtype=np.uint8,
            dst=dst,
        )
        result = dst.get_chw_array()
        # Top-right: band1 had nodata but band2 had 0 (not nodata) -> not overwritten.
        assert np.all(result[0, 0:2, 2:4] == nodata_vals[0])
        assert np.all(result[1, 0:2, 2:4] == 0)
        # Bottom-left: band1 had 0 (not nodata) but band2 had nodata -> not overwritten.
        assert np.all(result[0, 2:4, 0:2] == 0)
        assert np.all(result[1, 2:4, 0:2] == nodata_vals[1])
        # Top-left: both bands at nodata -> updated to 3.
        assert np.all(result[:, 0:2, 0:2] == 3)
        # Bottom-right: both bands at 0, not nodata -> unchanged.
        assert np.all(result[:, 2:4, 2:4] == 0)

    def test_nan_nodata_mosaic(self, tmp_path: pathlib.Path) -> None:
        """First-valid merge should detect NaN nodata and fill from source.

        NaN requires special handling since NaN != NaN.
        """
        tile_store = DefaultTileStore()
        tile_store.set_dataset_path(UPath(tmp_path))
        item = self._make_item()
        bands = ["band1"]
        src = np.full((1, 4, 4), 5.0, dtype=np.float32)
        tile_store.write_raster(
            self.LAYER_NAME,
            item,
            bands,
            self.PROJECTION,
            self.BOUNDS,
            RasterArray(chw_array=src),
        )

        dst_arr = np.full((1, 1, 4, 4), np.nan, dtype=np.float32)
        dst_arr[0, 0, 0:2, :] = 1.0
        dst = RasterArray(array=dst_arr)
        read_raster_window_from_tiles(
            tile_store=TileStoreWithLayer(tile_store, self.LAYER_NAME),
            item=item,
            bands=bands,
            projection=self.PROJECTION,
            bounds=self.BOUNDS,
            nodata_vals=(np.nan,),
            band_dtype=np.float32,
            dst=dst,
        )
        result = dst.get_chw_array()
        assert np.all(result[0, 0:2, :] == 1.0)
        assert np.all(result[0, 2:4, :] == 5.0)

    def test_none_nodata_unconditional_overwrite(self, tmp_path: pathlib.Path) -> None:
        """When nodata_vals is None, source pixels overwrite dst unconditionally."""
        tile_store = DefaultTileStore()
        tile_store.set_dataset_path(UPath(tmp_path))
        item = self._make_item()
        bands = ["band1"]
        src = 5 * np.ones((1, 4, 4), dtype=np.uint8)
        tile_store.write_raster(
            self.LAYER_NAME,
            item,
            bands,
            self.PROJECTION,
            self.BOUNDS,
            RasterArray(chw_array=src),
        )

        dst_arr = np.full((1, 1, 4, 4), 99, dtype=np.uint8)
        dst = RasterArray(array=dst_arr)
        read_raster_window_from_tiles(
            tile_store=TileStoreWithLayer(tile_store, self.LAYER_NAME),
            item=item,
            bands=bands,
            projection=self.PROJECTION,
            bounds=self.BOUNDS,
            nodata_vals=None,
            band_dtype=np.uint8,
            dst=dst,
        )
        result = dst.get_chw_array()
        assert np.all(result == 5)


class TestResolveNodataValues:
    """Tests for resolve_nodata_values."""

    LAYER_NAME = "layer"
    BANDS = ["band1", "band2"]
    BOUNDS = (0, 0, 4, 4)
    PROJECTION = WGS84_PROJECTION

    def _make_item(self, name: str = "item") -> Item:
        bbox = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        return Item(name, STGeometry(self.PROJECTION, bbox, None))

    def test_returns_none_when_source_has_no_nodata(
        self, tmp_path: pathlib.Path
    ) -> None:
        """resolve_nodata_values should return None when source has no nodata."""
        tile_store = DefaultTileStore()
        tile_store.set_dataset_path(UPath(tmp_path))
        item = self._make_item()
        src = np.ones((2, 4, 4), dtype=np.uint8)
        tile_store.write_raster(
            self.LAYER_NAME,
            item,
            self.BANDS,
            self.PROJECTION,
            self.BOUNDS,
            RasterArray(chw_array=src),
        )
        result = resolve_nodata_values(
            TileStoreWithLayer(tile_store, self.LAYER_NAME),
            [item],
            self.BANDS,
        )
        assert result is None

    def test_returns_tuple_when_source_has_nodata(self, tmp_path: pathlib.Path) -> None:
        """resolve_nodata_values returns a tuple when source metadata has nodata."""
        tile_store = DefaultTileStore()
        tile_store.set_dataset_path(UPath(tmp_path))
        item = self._make_item()
        nodata_meta = RasterMetadata(nodata_values=(255.0, 255.0))
        src = np.ones((2, 4, 4), dtype=np.uint8)
        tile_store.write_raster(
            self.LAYER_NAME,
            item,
            self.BANDS,
            self.PROJECTION,
            self.BOUNDS,
            RasterArray(chw_array=src, metadata=nodata_meta),
        )
        result = resolve_nodata_values(
            TileStoreWithLayer(tile_store, self.LAYER_NAME),
            [item],
            self.BANDS,
        )
        assert result == (255.0, 255.0)
