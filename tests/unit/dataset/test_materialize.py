import pathlib
from datetime import UTC, datetime
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from shapely.geometry import Polygon
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.data_source import Item
from rslearn.dataset.materialize import (
    build_mean_composite,
    build_median_composite,
    build_temporal_stack_composite,
    read_raster_window_from_tiles,
)
from rslearn.tile_stores.default import DefaultTileStore
from rslearn.tile_stores.tile_store import TileStoreWithLayer
from rslearn.utils.geometry import STGeometry
from rslearn.utils.raster_array import RasterArray


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
            item.name,
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
            item.name,
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


class TestBuildMeanComposite:
    """Unit tests for build_mean_composite"""

    LAYER_NAME = "layer"
    BANDS = ["band1", "band2"]
    BOUNDS = (0, 0, 4, 4)
    PROJECTION = WGS84_PROJECTION

    @pytest.fixture
    def tile_store(self, tmp_path: pathlib.Path) -> DefaultTileStore:
        store = DefaultTileStore()
        store.set_dataset_path(UPath(tmp_path))
        return store

    def make_item(self, name: str) -> Item:
        """Create a simple mock item with a name property."""
        return Item(
            name=name,
            geometry=STGeometry(
                projection=self.PROJECTION,
                shp=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
                time_range=None,
            ),
        )

    def test_mean_of_two_items(self, tile_store: DefaultTileStore) -> None:
        """Test mean composite of two 2-band rasters with valid values."""
        nodata_vals = [0, 0]

        array1 = np.array(
            [np.full((4, 4), 2, dtype=np.uint8), np.full((4, 4), 6, dtype=np.uint8)]
        )
        array2 = np.array(
            [np.full((4, 4), 4, dtype=np.uint8), np.full((4, 4), 10, dtype=np.uint8)]
        )

        item1 = self.make_item("item1")
        item2 = self.make_item("item2")

        for item, data in zip([item1, item2], [array1, array2]):
            tile_store.write_raster(
                self.LAYER_NAME,
                item.name,
                self.BANDS,
                self.PROJECTION,
                self.BOUNDS,
                RasterArray(chw_array=data),
            )

        composite = build_mean_composite(
            group=[item1, item2],
            nodata_vals=nodata_vals,
            bands=self.BANDS,
            bounds=self.BOUNDS,
            band_dtype=np.uint8,
            tile_store=TileStoreWithLayer(tile_store, self.LAYER_NAME),
            projection=self.PROJECTION,
            remapper=None,
        )

        expected = np.array(
            [
                np.full((4, 4), 3, dtype=np.uint8),  # Mean of 2 and 4
                np.full((4, 4), 8, dtype=np.uint8),  # Mean of 6 and 10
            ]
        )
        assert np.array_equal(composite.get_chw_array(), expected)

    def test_mean_three_items_partial_overlap(
        self, tile_store: DefaultTileStore
    ) -> None:
        """Test mean composite with 3 items having different spatial extents (float32)."""
        nodata_vals = [0.0, 0.0]

        def make_array(val1: Any, val2: Any) -> npt.NDArray[np.float32]:
            return np.array(
                [
                    np.full((4, 4), val1, dtype=np.float32),
                    np.full((4, 4), val2, dtype=np.float32),
                ],
                dtype=np.float32,
            )

        item1 = self.make_item("item1")
        item2 = self.make_item("item2")
        item3 = self.make_item("item3")

        # item1: full coverage
        array1 = make_array(3.0, 9.0)

        # item2: only covers left half (nodata in right half)
        array2 = make_array(6.0, 12.0)
        array2[:, :, 2:4] = 0.0

        # item3: only covers bottom half
        array3 = make_array(9.0, 15.0)
        array3[:, 0:2, :] = 0.0

        for item, array in zip([item1, item2, item3], [array1, array2, array3]):
            tile_store.write_raster(
                self.LAYER_NAME,
                item.name,
                self.BANDS,
                self.PROJECTION,
                self.BOUNDS,
                RasterArray(chw_array=array),
            )

        composite = build_mean_composite(
            group=[item1, item2, item3],
            nodata_vals=nodata_vals,
            bands=self.BANDS,
            bounds=self.BOUNDS,
            band_dtype=np.float32,
            tile_store=TileStoreWithLayer(tile_store, self.LAYER_NAME),
            projection=self.PROJECTION,
            remapper=None,
        )

        # Expected values are exact means (no rounding)
        expected = np.array(
            [
                [  # band1
                    [4.5, 4.5, 3.0, 3.0],
                    [4.5, 4.5, 3.0, 3.0],
                    [6.0, 6.0, 6.0, 6.0],
                    [6.0, 6.0, 6.0, 6.0],
                ],
                [  # band2
                    [10.5, 10.5, 9.0, 9.0],
                    [10.5, 10.5, 9.0, 9.0],
                    [12.0, 12.0, 12.0, 12.0],
                    [12.0, 12.0, 12.0, 12.0],
                ],
            ],
            dtype=np.float32,
        )

        assert np.array_equal(composite.get_chw_array(), expected)

    def test_mean_with_different_nodata_vals(
        self, tile_store: DefaultTileStore
    ) -> None:
        """Test mean composite where each band has a different nodata value (float32)."""

        # Different nodata values for each band
        nodata_vals = [0.0, 99.0]  # Band 1 has 0.0 as nodata, Band 2 has 99.0 as nodata

        array1 = np.array(
            [
                np.full((4, 4), 2.0, dtype=np.float32),
                np.full((4, 4), 6.0, dtype=np.float32),
            ]
        )
        array2 = np.array(
            [
                np.full((4, 4), 4.0, dtype=np.float32),
                np.full((4, 4), 10.0, dtype=np.float32),
            ]
        )

        # Manually set some pixels to nodata
        array1[0, 0, 0] = 0.0  # Set (0,0) to nodata in first band
        array2[1, 1, 1] = 99.0  # Set (1,1) to nodata in second band

        item1 = self.make_item("item1")
        item2 = self.make_item("item2")

        for item, data in zip([item1, item2], [array1, array2]):
            tile_store.write_raster(
                self.LAYER_NAME,
                item.name,
                self.BANDS,
                self.PROJECTION,
                self.BOUNDS,
                RasterArray(chw_array=data),
            )

        composite = build_mean_composite(
            group=[item1, item2],
            nodata_vals=nodata_vals,
            bands=self.BANDS,
            bounds=self.BOUNDS,
            band_dtype=np.float32,
            tile_store=TileStoreWithLayer(tile_store, self.LAYER_NAME),
            projection=self.PROJECTION,
            remapper=None,
        )

        # Expected float means
        expected = np.array(
            [
                np.full((4, 4), 3.0, dtype=np.float32),  # Mean of 2.0 and 4.0
                np.full((4, 4), 8.0, dtype=np.float32),  # Mean of 6.0 and 10.0
            ]
        )

        # Override the pixels where a nodata was injected:
        expected[0, 0, 0] = 4.0  # band 1, (0,0): only valid value is 4.0
        expected[1, 1, 1] = 6.0  # band 2, (1,1): only valid value is 6.0

        # Check that the composite is as expected (allow small float error)
        assert np.allclose(composite.get_chw_array(), expected, atol=1e-6)


class TestBuildMedianComposite:
    """Unit tests for build_median_composite"""

    LAYER_NAME = "layer"
    BANDS = ["band1", "band2"]
    BOUNDS = (0, 0, 4, 4)
    PROJECTION = WGS84_PROJECTION

    @pytest.fixture
    def tile_store(self, tmp_path: pathlib.Path) -> DefaultTileStore:
        store = DefaultTileStore()
        store.set_dataset_path(UPath(tmp_path))
        return store

    def make_item(self, name: str) -> Item:
        """Create a simple mock item with a name property."""
        return Item(
            name=name,
            geometry=STGeometry(
                projection=self.PROJECTION,
                shp=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
                time_range=None,
            ),
        )

    def test_median_of_two_items(self, tile_store: DefaultTileStore) -> None:
        """Median of two 2-band rasters with valid values everywhere."""
        nodata_vals = [0, 0]

        array1 = np.array(
            [
                np.full((4, 4), 2, dtype=np.uint8),
                np.full((4, 4), 6, dtype=np.uint8),
            ]
        )
        array2 = np.array(
            [
                np.full((4, 4), 4, dtype=np.uint8),
                np.full((4, 4), 10, dtype=np.uint8),
            ]
        )

        item1 = self.make_item("item1")
        item2 = self.make_item("item2")

        for item, data in zip([item1, item2], [array1, array2]):
            tile_store.write_raster(
                self.LAYER_NAME,
                item.name,
                self.BANDS,
                self.PROJECTION,
                self.BOUNDS,
                RasterArray(chw_array=data),
            )

        composite = build_median_composite(
            group=[item1, item2],
            nodata_vals=nodata_vals,
            bands=self.BANDS,
            bounds=self.BOUNDS,
            band_dtype=np.uint8,
            tile_store=TileStoreWithLayer(tile_store, self.LAYER_NAME),
            projection=self.PROJECTION,
            remapper=None,
        )

        # Median of (2,4) -> 3; (6,10) -> 8
        expected = np.array(
            [
                np.full((4, 4), 3, dtype=np.uint8),
                np.full((4, 4), 8, dtype=np.uint8),
            ]
        )
        assert np.array_equal(composite.get_chw_array(), expected)

    def test_median_three_items_partial_overlap(
        self, tile_store: DefaultTileStore
    ) -> None:
        """Median with 3 items having different spatial extents."""
        nodata_vals = [0, 0]

        def make_array(val1: Any, val2: Any) -> npt.NDArray[np.float32]:
            return np.array(
                [
                    np.full((4, 4), val1, dtype=np.uint8),
                    np.full((4, 4), val2, dtype=np.uint8),
                ]
            )

        item1 = self.make_item("item1")
        item2 = self.make_item("item2")
        item3 = self.make_item("item3")

        # item1: full coverage
        array1 = make_array(3, 9)

        # item2: covers left half only (right half nodata)
        array2 = make_array(6, 12)
        array2[:, :, 2:4] = 0

        # item3: covers bottom half only (top half nodata)
        array3 = make_array(9, 15)
        array3[:, 0:2, :] = 0

        for item, array in zip([item1, item2, item3], [array1, array2, array3]):
            tile_store.write_raster(
                self.LAYER_NAME,
                item.name,
                self.BANDS,
                self.PROJECTION,
                self.BOUNDS,
                RasterArray(chw_array=array),
            )

        composite = build_median_composite(
            group=[item1, item2, item3],
            nodata_vals=nodata_vals,
            bands=self.BANDS,
            bounds=self.BOUNDS,
            band_dtype=np.uint8,
            tile_store=TileStoreWithLayer(tile_store, self.LAYER_NAME),
            projection=self.PROJECTION,
            remapper=None,
        )

        # Band1 values by region (exclude nodata=0):
        # TL: {3,6} -> median (3+6)/2 = 4.5 -> 4 (uint8)
        # TR: {3}   -> 3
        # BL: {3,6,9} -> 6
        # BR: {3,9} -> (3+9)/2 = 6
        # Band2: TL {9,12}->10.5->10; TR {9}->9; BL {9,12,15}->12; BR {9,15}->12
        expected = np.array(
            [
                [  # band1
                    [4, 4, 3, 3],
                    [4, 4, 3, 3],
                    [6, 6, 6, 6],
                    [6, 6, 6, 6],
                ],
                [  # band2
                    [10, 10, 9, 9],
                    [10, 10, 9, 9],
                    [12, 12, 12, 12],
                    [12, 12, 12, 12],
                ],
            ],
            dtype=np.uint8,
        )

        assert np.array_equal(composite.get_chw_array(), expected)

    def test_median_with_different_nodata_vals(
        self, tile_store: DefaultTileStore
    ) -> None:
        """Median where each band has a different nodata value and per-pixel masks."""
        nodata_vals = [0, 99]

        array1 = np.array(
            [
                np.full((4, 4), 2, dtype=np.uint8),
                np.full((4, 4), 6, dtype=np.uint8),
            ]
        )
        array2 = np.array(
            [
                np.full((4, 4), 4, dtype=np.uint8),
                np.full((4, 4), 10, dtype=np.uint8),
            ]
        )

        # Set some pixels to band-specific nodata
        array1[0, 0, 0] = 0  # band1 nodata at (0,0)
        array2[1, 1, 1] = 99  # band2 nodata at (1,1)

        item1 = self.make_item("item1")
        item2 = self.make_item("item2")

        for item, data in zip([item1, item2], [array1, array2]):
            tile_store.write_raster(
                self.LAYER_NAME,
                item.name,
                self.BANDS,
                self.PROJECTION,
                self.BOUNDS,
                RasterArray(chw_array=data),
            )

        composite = build_median_composite(
            group=[item1, item2],
            nodata_vals=nodata_vals,
            bands=self.BANDS,
            bounds=self.BOUNDS,
            band_dtype=np.uint8,
            tile_store=TileStoreWithLayer(tile_store, self.LAYER_NAME),
            projection=self.PROJECTION,
            remapper=None,
        )

        # Base median everywhere: (2,4)->3; (6,10)->8.
        # band1 at (0,0): only 4 is valid -> 4
        # band2 at (1,1): only 6 is valid -> 6
        expected = np.array(
            [
                np.full((4, 4), 3, dtype=np.uint8),
                np.full((4, 4), 8, dtype=np.uint8),
            ]
        )
        expected[0, 0, 0] = 4
        expected[1, 1, 1] = 6

        assert np.array_equal(composite.get_chw_array(), expected)


class TestBuildTemporalStackComposite:
    """Tests for build_temporal_stack_composite."""

    LAYER_NAME = "layer"
    BANDS = ["B1"]
    BOUNDS = (0, 0, 4, 4)
    PROJECTION = WGS84_PROJECTION

    def test_two_timesteps(self, tmp_path: pathlib.Path) -> None:
        """Two items with different time ranges produce a (C, 2, H, W) stack.

        Here, both items fully cover the window's spatial extent.
        """
        tile_store = DefaultTileStore()
        tile_store.set_dataset_path(UPath(tmp_path))

        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        t1 = datetime(2024, 1, 2, tzinfo=UTC)
        t2 = datetime(2024, 1, 3, tzinfo=UTC)
        bbox = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])

        item_a = Item("a", STGeometry(self.PROJECTION, bbox, (t0, t1)))
        data_a = np.full((1, 4, 4), 10, dtype=np.uint8)
        tile_store.write_raster(
            self.LAYER_NAME,
            "a",
            self.BANDS,
            self.PROJECTION,
            self.BOUNDS,
            RasterArray(chw_array=data_a, time_range=(t0, t1)),
        )

        item_b = Item("b", STGeometry(self.PROJECTION, bbox, (t1, t2)))
        data_b = np.full((1, 4, 4), 20, dtype=np.uint8)
        tile_store.write_raster(
            self.LAYER_NAME,
            "b",
            self.BANDS,
            self.PROJECTION,
            self.BOUNDS,
            RasterArray(chw_array=data_b, time_range=(t1, t2)),
        )

        result = build_temporal_stack_composite(
            group=[item_a, item_b],
            nodata_vals=[0],
            bands=self.BANDS,
            bounds=self.BOUNDS,
            band_dtype=np.uint8,
            tile_store=TileStoreWithLayer(tile_store, self.LAYER_NAME),
            projection=self.PROJECTION,
            remapper=None,
        )

        assert result.array.shape == (1, 2, 4, 4)
        # First timestep should be from the first item.
        assert np.all(result.array[:, 0, :, :] == 10)
        # Second timestep should be from the second item.
        assert np.all(result.array[:, 1, :, :] == 20)
        # Timestamps should be set correctly too.
        assert result.timestamps == [(t0, t1), (t1, t2)]

    def test_spatial_mosaic_within_timestep(self, tmp_path: pathlib.Path) -> None:
        """Two items with the same time range are spatially composited into one timestep."""
        tile_store = DefaultTileStore()
        tile_store.set_dataset_path(UPath(tmp_path))

        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        t1 = datetime(2024, 1, 2, tzinfo=UTC)
        bbox_left = Polygon([(0, 0), (2, 0), (2, 4), (0, 4)])
        bbox_right = Polygon([(2, 0), (4, 0), (4, 4), (2, 4)])

        item_a = Item("a", STGeometry(self.PROJECTION, bbox_left, (t0, t1)))
        data_a = np.full((1, 4, 2), 5, dtype=np.uint8)
        tile_store.write_raster(
            self.LAYER_NAME,
            "a",
            self.BANDS,
            self.PROJECTION,
            (0, 0, 2, 4),
            RasterArray(chw_array=data_a, time_range=(t0, t1)),
        )

        item_b = Item("b", STGeometry(self.PROJECTION, bbox_right, (t0, t1)))
        data_b = np.full((1, 4, 2), 15, dtype=np.uint8)
        tile_store.write_raster(
            self.LAYER_NAME,
            "b",
            self.BANDS,
            self.PROJECTION,
            (2, 0, 4, 4),
            RasterArray(chw_array=data_b, time_range=(t0, t1)),
        )

        result = build_temporal_stack_composite(
            group=[item_a, item_b],
            nodata_vals=[0],
            bands=self.BANDS,
            bounds=self.BOUNDS,
            band_dtype=np.uint8,
            tile_store=TileStoreWithLayer(tile_store, self.LAYER_NAME),
            projection=self.PROJECTION,
            remapper=None,
        )

        assert result.array.shape == (1, 1, 4, 4)
        assert np.all(result.array[:, 0, :, 0:2] == 5)
        assert np.all(result.array[:, 0, :, 2:4] == 15)
        assert result.timestamps == [(t0, t1)]

    def test_multi_timestep_item(self, tmp_path: pathlib.Path) -> None:
        """An item with T>1 in the tile store is stacked correctly."""
        tile_store = DefaultTileStore()
        tile_store.set_dataset_path(UPath(tmp_path))

        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        t1 = datetime(2024, 1, 2, tzinfo=UTC)
        t2 = datetime(2024, 1, 3, tzinfo=UTC)
        bbox = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])

        # Write a CTHW raster with T=2 and timestamps via write_raster.
        data = np.zeros((1, 2, 4, 4), dtype=np.uint8)
        data[:, 0, :, :] = 30
        data[:, 1, :, :] = 40
        timestamps = [(t0, t1), (t1, t2)]
        raster = RasterArray(array=data, timestamps=timestamps)
        tile_store.write_raster(
            self.LAYER_NAME,
            "a",
            self.BANDS,
            self.PROJECTION,
            self.BOUNDS,
            raster,
        )

        item_a = Item("a", STGeometry(self.PROJECTION, bbox, (t0, t2)))

        result = build_temporal_stack_composite(
            group=[item_a],
            nodata_vals=[0],
            bands=self.BANDS,
            bounds=self.BOUNDS,
            band_dtype=np.uint8,
            tile_store=TileStoreWithLayer(tile_store, self.LAYER_NAME),
            projection=self.PROJECTION,
            remapper=None,
        )

        assert result.array.shape == (1, 2, 4, 4)
        assert np.all(result.array[:, 0, :, :] == 30)
        assert np.all(result.array[:, 1, :, :] == 40)
        assert result.timestamps == timestamps

    def test_temporal_stack_with_window_clipping(self, tmp_path: pathlib.Path) -> None:
        """Timesteps outside the window time range are clipped.

        Item 1: T=3, timestamps [Jan Feb Mar], covers top 3/4
        Item 2: T=3, timestamps [Mar Apr May], covers bottom 3/4
        Window time range: (Feb Mar Apr).

        So result should have Item 1 for Feb Mar and Item 2 for Mar Apr.
        """
        tile_store = DefaultTileStore()
        tile_store.set_dataset_path(UPath(tmp_path))

        jan = datetime(2024, 1, 1, tzinfo=UTC)
        feb = datetime(2024, 2, 1, tzinfo=UTC)
        mar = datetime(2024, 3, 1, tzinfo=UTC)
        apr = datetime(2024, 4, 1, tzinfo=UTC)
        may = datetime(2024, 5, 1, tzinfo=UTC)
        jun = datetime(2024, 6, 1, tzinfo=UTC)

        # Item 1 (value=10): covers top 3/4 (rows 0-2 of a 4-row image), 3 timesteps.
        bbox_top = Polygon([(0, 0), (4, 0), (4, 3), (0, 3)])
        item_a = Item("a", STGeometry(self.PROJECTION, bbox_top, (jan, apr)))
        data_a = np.full((1, 3, 3, 4), 10, dtype=np.uint8)  # (C=1, T=3, H=3, W=4)
        ts_a = [(jan, feb), (feb, mar), (mar, apr)]
        tile_store.write_raster(
            self.LAYER_NAME,
            "a",
            self.BANDS,
            self.PROJECTION,
            (0, 0, 4, 3),
            RasterArray(array=data_a, timestamps=ts_a),
        )

        # Item 2 (value=20): covers bottom 3/4 (rows 1-3), 3 timesteps.
        bbox_bot = Polygon([(0, 1), (4, 1), (4, 4), (0, 4)])
        item_b = Item("b", STGeometry(self.PROJECTION, bbox_bot, (mar, jun)))
        data_b = np.full((1, 3, 3, 4), 20, dtype=np.uint8)  # (C=1, T=3, H=3, W=4)
        ts_b = [(mar, apr), (apr, may), (may, jun)]
        tile_store.write_raster(
            self.LAYER_NAME,
            "b",
            self.BANDS,
            self.PROJECTION,
            (0, 1, 4, 4),
            RasterArray(array=data_b, timestamps=ts_b),
        )

        request_time_range = (feb, may)

        result = build_temporal_stack_composite(
            group=[item_a, item_b],
            nodata_vals=[0],
            bands=self.BANDS,
            bounds=self.BOUNDS,
            band_dtype=np.uint8,
            tile_store=TileStoreWithLayer(tile_store, self.LAYER_NAME),
            projection=self.PROJECTION,
            remapper=None,
            request_time_range=request_time_range,
        )

        # Should have 3 timesteps after clipping.
        assert result.array.shape == (1, 3, 4, 4)
        assert result.timestamps == [(feb, mar), (mar, apr), (apr, may)]

        # T=0 (Feb): only Item 1, top 3 rows.
        assert np.all(result.array[:, 0, 0:3, :] == 10)
        assert np.all(result.array[:, 0, 3, :] == 0)  # nodata

        # T=1 (Mar): Item 1 top 3 rows (first-valid), Item 2 bottom 3 rows.
        # Overlap in rows 1-2: first-valid -> Item 1 wins.
        assert np.all(result.array[:, 1, 0:3, :] == 10)
        assert np.all(result.array[:, 1, 3, :] == 20)

        # T=2 (Apr): only Item 2, bottom 3 rows.
        assert np.all(result.array[:, 2, 0, :] == 0)  # nodata
        assert np.all(result.array[:, 2, 1:4, :] == 20)

    def test_temporal_stack_nodata_two_bands(self, tmp_path: pathlib.Path) -> None:
        """Nodata handling with two bands and per-band nodata values.

        nodata_vals = [1, 2].
        Item 1: T=2 [Jan Feb], full coverage, both bands are nodata
                everywhere ([1, 2]) except topleft pixel (0,0) which is [10, 10].
        Item 2: T=1 [Feb], full coverage, [20, 20] everywhere.
        Window: [Jan Feb].

        Expected:
          T=0 (Jan): only Item 1. (0,0)=[10,10], rest=[1,2] (nodata).
          T=1 (Feb): Item 1 at topleft pixel, Item 2 elsewhere.
        """
        tile_store = DefaultTileStore()
        tile_store.set_dataset_path(UPath(tmp_path))

        jan = datetime(2024, 1, 1, tzinfo=UTC)
        feb = datetime(2024, 2, 1, tzinfo=UTC)
        mar = datetime(2024, 3, 1, tzinfo=UTC)

        bands = ["B1", "B2"]
        nodata_vals = [1, 2]
        bbox = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])

        # Item 1: nodata everywhere except topleft, two timesteps.
        item_a = Item("a", STGeometry(self.PROJECTION, bbox, (jan, mar)))
        data_a = np.empty((2, 2, 4, 4), dtype=np.uint8)
        data_a[0, :, :, :] = 1  # band0 = nodata
        data_a[1, :, :, :] = 2  # band1 = nodata
        data_a[:, :, 0, 0] = 10  # topleft pixel is valid in both timesteps
        ts_a = [(jan, feb), (feb, mar)]
        tile_store.write_raster(
            self.LAYER_NAME,
            "a",
            bands,
            self.PROJECTION,
            self.BOUNDS,
            RasterArray(array=data_a, timestamps=ts_a),
        )

        # Item 2: [20, 20] everywhere, single timestep covering feb.
        item_b = Item("b", STGeometry(self.PROJECTION, bbox, (feb, mar)))
        data_b = np.full((2, 4, 4), 20, dtype=np.uint8)
        tile_store.write_raster(
            self.LAYER_NAME,
            "b",
            bands,
            self.PROJECTION,
            self.BOUNDS,
            RasterArray(chw_array=data_b, time_range=(feb, mar)),
        )

        request_time_range = (jan, mar)

        result = build_temporal_stack_composite(
            group=[item_a, item_b],
            nodata_vals=nodata_vals,
            bands=bands,
            bounds=self.BOUNDS,
            band_dtype=np.uint8,
            tile_store=TileStoreWithLayer(tile_store, self.LAYER_NAME),
            projection=self.PROJECTION,
            remapper=None,
            request_time_range=request_time_range,
        )

        assert result.array.shape == (2, 2, 4, 4)
        assert result.timestamps == [(jan, feb), (feb, mar)]

        # T=0 (Jan): only Item 1.
        # Topleft valid.
        assert result.array[0, 0, 0, 0] == 10
        assert result.array[1, 0, 0, 0] == 10
        # Rest is nodata.
        assert np.all(result.array[0, 0, 0, 1:] == 1)
        assert np.all(result.array[0, 0, 1:, :] == 1)
        assert np.all(result.array[1, 0, 0, 1:] == 2)
        assert np.all(result.array[1, 0, 1:, :] == 2)

        # T=1 (Feb): Item 1 first-valid, then Item 2 fills nodata.
        # Topleft: Item 1 has [10,10] (not nodata) -> keeps.
        assert result.array[0, 1, 0, 0] == 10
        assert result.array[1, 1, 0, 0] == 10
        # Everywhere else: Item 1 had [1,2] (nodata) -> Item 2's [20,20].
        assert np.all(result.array[0, 1, 0, 1:] == 20)
        assert np.all(result.array[0, 1, 1:, :] == 20)
        assert np.all(result.array[1, 1, 0, 1:] == 20)
        assert np.all(result.array[1, 1, 1:, :] == 20)
