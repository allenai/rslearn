import pathlib
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
    read_raster_window_from_tiles,
)
from rslearn.tile_stores.default import DefaultTileStore
from rslearn.tile_stores.tile_store import TileStoreWithLayer
from rslearn.utils.geometry import STGeometry


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
        # Set first band 1 in top half, and second band 2 in left half.
        # So then only topleft has both bands matching nodata.
        dst[0, 0:2, 0:4] = nodata_vals[0]
        dst[1, 0:4, 0:2] = nodata_vals[1]
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
        # Top-right should be unchanged.
        assert np.all(dst[0, 0:2, 2:4] == nodata_vals[0])
        assert np.all(dst[1, 0:2, 2:4] == 0)
        # Bottom-left should be unchanged.
        assert np.all(dst[0, 2:4, 0:2] == 0)
        assert np.all(dst[1, 2:4, 0:2] == nodata_vals[1])
        # Top-left is updated to 3.
        assert np.all(dst[:, 0:2, 0:2] == 3)
        # Bottom-right should be unchanged (still 0).
        assert np.all(dst[:, 2:4, 2:4] == 0)


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
                data,
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
        assert np.array_equal(composite, expected)

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
                array,
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

        assert np.array_equal(composite, expected)

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
                data,
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
        assert np.allclose(composite, expected, atol=1e-6)


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
                data,
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
        assert np.array_equal(composite, expected)

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
                array,
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

        assert np.array_equal(composite, expected)

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
                data,
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

        assert np.array_equal(composite, expected)


class TestSingleFileMaterialization:
    """Unit tests for single_file_materialization option in RasterMaterializer."""

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

    def test_single_file_materialization(
        self, tile_store: DefaultTileStore, tmp_path: pathlib.Path
    ) -> None:
        """Test that single_file_materialization writes all item groups to one file."""
        from unittest.mock import MagicMock

        from rslearn.config import BandSetConfig, DType, LayerConfig, LayerType
        from rslearn.dataset.materialize import RasterMaterializer
        from rslearn.utils.raster_format import GeotiffRasterFormat

        # Create items and write raster data for each
        items_group1 = [self.make_item("item1")]
        items_group2 = [self.make_item("item2")]
        items_group3 = [self.make_item("item3")]

        # Write different values for each item
        for i, item in enumerate([items_group1[0], items_group2[0], items_group3[0]]):
            array = np.full((2, 4, 4), i + 1, dtype=np.uint8)
            tile_store.write_raster(
                self.LAYER_NAME,
                item.name,
                self.BANDS,
                self.PROJECTION,
                self.BOUNDS,
                array,
            )

        # Create a mock window
        window = MagicMock()
        window.projection = self.PROJECTION
        window.bounds = self.BOUNDS

        # Track the directories where rasters are written
        written_dirs: list[UPath] = []

        def get_raster_dir(layer_name: str, bands: list[str], group_idx: int) -> UPath:
            path = UPath(tmp_path) / "window" / f"{layer_name}_{group_idx}"
            written_dirs.append(path)
            return path

        window.get_raster_dir = get_raster_dir

        # Create layer config with single_file_materialization enabled
        layer_cfg = LayerConfig(
            type=LayerType.RASTER,
            band_sets=[
                BandSetConfig(
                    dtype=DType.UINT8,
                    bands=self.BANDS,
                )
            ],
            single_file_materialization=True,
        )

        # Create item groups (3 groups)
        item_groups = [items_group1, items_group2, items_group3]

        # Materialize
        materializer = RasterMaterializer()
        materializer.materialize(
            TileStoreWithLayer(tile_store, self.LAYER_NAME),
            window,
            self.LAYER_NAME,
            layer_cfg,
            item_groups,
        )

        # Verify: with single_file_materialization, only group_idx=0 should be used
        assert len(written_dirs) == 1
        assert "0" in str(written_dirs[0])

        # Verify that the stacked file was created
        raster_format = GeotiffRasterFormat()
        assert (written_dirs[0] / raster_format.stacked_fname).exists()
        assert (written_dirs[0] / raster_format.stacked_metadata_fname).exists()

        # Verify the stacked data can be read and has correct shape
        decoded = raster_format.decode_stacked_raster(
            written_dirs[0], self.PROJECTION, self.BOUNDS
        )
        assert decoded.shape == (3, 2, 4, 4)  # (num_groups, num_channels, H, W)

        # Verify the values for each group
        for i in range(3):
            expected_val = i + 1
            assert np.all(decoded[i, :, :, :] == expected_val)

        # Verify mark_layer_completed was called only for group 0
        window.mark_layer_completed.assert_called_once_with(self.LAYER_NAME, 0)
