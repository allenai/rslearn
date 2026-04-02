"""Tests for rslearn.dataset.tile_utils (read_raster_window_from_tiles)."""

import pathlib

import numpy as np
from shapely.geometry import Polygon
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.data_source import Item
from rslearn.dataset.compositing import FirstValidCompositor
from rslearn.dataset.materialize import resolve_nodata_values
from rslearn.dataset.tile_utils import read_raster_window_from_tiles
from rslearn.tile_stores.default import DefaultTileStore
from rslearn.tile_stores.tile_store import TileStoreWithLayer
from rslearn.utils.geometry import STGeometry
from rslearn.utils.raster_array import RasterArray, RasterMetadata
from rslearn.utils.raster_format import GeotiffRasterFormat


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
            nodata_vals=[0],
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

        nodata_vals = [1.0, 2.0]
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


class TestNodataEndToEnd:
    """Verify nodata values propagate through ingestion -> compositing -> materialization."""

    LAYER_NAME = "layer"
    BANDS = ["band1"]
    BOUNDS = (0, 0, 4, 4)
    PROJECTION = WGS84_PROJECTION
    NODATA_VAL = -32768.0

    def _make_item(self, name: str = "item") -> Item:
        bbox = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        return Item(name, STGeometry(self.PROJECTION, bbox, None))

    def test_nodata_persists_through_ingestion_and_materialization(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Write a raster with nodata metadata, composite it, encode/decode the result.

        Checks that nodata_values survive:
        1. Tile store write/read (ingestion)
        2. Compositing (resolve_nodata_values + FirstValidCompositor)
        3. Raster format encode/decode (materialization output)
        """
        tile_store = DefaultTileStore()
        tile_store.set_dataset_path(UPath(tmp_path))
        item = self._make_item()

        src = np.ones((1, 4, 4), dtype=np.float32) * 42.0
        raster = RasterArray(
            chw_array=src,
            metadata=RasterMetadata(nodata_values=[self.NODATA_VAL]),
        )
        tile_store.write_raster(
            self.LAYER_NAME, item, self.BANDS, self.PROJECTION, self.BOUNDS, raster
        )

        # 1. Verify nodata survives ingestion into the tile store.
        ts_with_layer = TileStoreWithLayer(tile_store, self.LAYER_NAME)
        metadata = ts_with_layer.get_raster_metadata(item, self.BANDS)
        assert metadata.nodata_values == [self.NODATA_VAL]

        # 2. Verify resolve_nodata_values picks it up.
        resolved = resolve_nodata_values(ts_with_layer, [item], self.BANDS)
        assert resolved == [self.NODATA_VAL]

        # 3. Verify compositing propagates nodata metadata.
        from rasterio.enums import Resampling

        composite = FirstValidCompositor().build_composite(
            group=[item],
            nodata_vals=resolved,
            bands=self.BANDS,
            bounds=self.BOUNDS,
            band_dtype=np.float32,
            tile_store=ts_with_layer,
            projection=self.PROJECTION,
            resampling_method=Resampling.bilinear,
            remapper=None,
        )
        assert composite.metadata.nodata_values == [self.NODATA_VAL]
        np.testing.assert_array_equal(composite.get_chw_array(), src)

        # 4. Verify encode/decode roundtrip preserves nodata (materialization output).
        output_dir = UPath(tmp_path / "materialized")
        output_dir.mkdir()
        fmt = GeotiffRasterFormat()
        fmt.encode_raster(output_dir, self.PROJECTION, self.BOUNDS, composite)
        decoded = fmt.decode_raster(output_dir, self.PROJECTION, self.BOUNDS)
        assert decoded.metadata.nodata_values == [self.NODATA_VAL]
        np.testing.assert_array_equal(decoded.get_chw_array(), src)
