"""Tests for rslearn.dataset.sentinel2_scl compositors."""

import pathlib

import numpy as np
import pytest
from rasterio.enums import Resampling
from shapely.geometry import Polygon
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.data_source import Item
from rslearn.dataset.sentinel2_scl import Sentinel2SCLBestClear, Sentinel2SCLFirstValid
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.tile_stores.default import DefaultTileStore
from rslearn.utils.geometry import PixelBounds, STGeometry
from rslearn.utils.raster_array import RasterArray

LAYER_NAME = "layer"
BOUNDS: PixelBounds = (0, 0, 4, 4)
PROJECTION = WGS84_PROJECTION
OUTPUT_BANDS = ["B04"]
STORE_BANDS = OUTPUT_BANDS + ["SCL"]


def _make_item(name: str) -> Item:
    bbox = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    return Item(name, STGeometry(PROJECTION, bbox, None))


def _write_items(
    store: DefaultTileStore,
    items: list[Item],
    arrays: list[np.ndarray],
) -> None:
    for item, arr in zip(items, arrays):
        store.write_raster(
            LAYER_NAME,
            item,
            STORE_BANDS,
            PROJECTION,
            BOUNDS,
            RasterArray(chw_array=arr),
        )


class TestSentinel2SCLFirstValid:
    """Tests for Sentinel2SCLFirstValid compositor."""

    def test_sorts_items_by_scl_cloud_score(self, tmp_path: pathlib.Path) -> None:
        """Items should be reordered so the least cloudy item is first."""
        store = DefaultTileStore()
        store.set_dataset_path(UPath(tmp_path))

        item_cloudy = _make_item("cloudy")
        item_clear = _make_item("clear")

        cloudy_data = np.stack(
            [
                np.full((4, 4), 100, dtype=np.uint16),  # B04
                np.full((4, 4), 9, dtype=np.uint16),  # SCL high cloud
            ],
            axis=0,
        )
        clear_data = np.stack(
            [
                np.full((4, 4), 200, dtype=np.uint16),  # B04
                np.full((4, 4), 4, dtype=np.uint16),  # SCL clear class
            ],
            axis=0,
        )
        _write_items(store, [item_cloudy, item_clear], [cloudy_data, clear_data])

        compositor = Sentinel2SCLFirstValid(scl_band="SCL")
        result = compositor.build_composite(
            group=[item_cloudy, item_clear],
            nodata_val=0,
            bands=OUTPUT_BANDS,
            bounds=BOUNDS,
            band_dtype=np.uint16,
            tile_store=TileStoreWithLayer(store, LAYER_NAME),
            projection=PROJECTION,
            resampling_method=Resampling.bilinear,
            remapper=None,
        )

        # Clear item (200) should come first in FIRST_VALID compositing.
        assert np.all(result.get_chw_array() == 200)

    def test_single_item_no_scoring(self, tmp_path: pathlib.Path) -> None:
        """With a single item, SCL scoring should not run."""
        store = DefaultTileStore()
        store.set_dataset_path(UPath(tmp_path))

        item = _make_item("single")
        data = np.stack(
            [
                np.full((4, 4), 42, dtype=np.uint16),
                np.full((4, 4), 4, dtype=np.uint16),
            ],
            axis=0,
        )
        _write_items(store, [item], [data])

        compositor = Sentinel2SCLFirstValid(scl_band="SCL")

        def fail_score(*args: object, **kwargs: object) -> float | None:
            raise AssertionError("should not be called")

        original_score_item = compositor._score_item
        compositor._score_item = fail_score  # type: ignore[method-assign]
        try:
            result = compositor.build_composite(
                group=[item],
                nodata_val=0,
                bands=OUTPUT_BANDS,
                bounds=BOUNDS,
                band_dtype=np.uint16,
                tile_store=TileStoreWithLayer(store, LAYER_NAME),
                projection=PROJECTION,
                resampling_method=Resampling.bilinear,
                remapper=None,
            )
        finally:
            compositor._score_item = original_score_item  # type: ignore[method-assign]

        assert np.all(result.get_chw_array() == 42)

    def test_missing_scl_band_raises(self, tmp_path: pathlib.Path) -> None:
        """Scoring should fail if the configured SCL band is unavailable."""
        store = DefaultTileStore()
        store.set_dataset_path(UPath(tmp_path))

        item_a = _make_item("a")
        item_b = _make_item("b")
        for item, value in [(item_a, 10), (item_b, 20)]:
            data = np.full((1, 4, 4), value, dtype=np.uint16)  # only B04, no SCL
            store.write_raster(
                LAYER_NAME,
                item,
                OUTPUT_BANDS,
                PROJECTION,
                BOUNDS,
                RasterArray(chw_array=data),
            )

        compositor = Sentinel2SCLFirstValid(scl_band="SCL")
        with pytest.raises(ValueError, match="missing scoring bands"):
            compositor.build_composite(
                group=[item_a, item_b],
                nodata_val=0,
                bands=OUTPUT_BANDS,
                bounds=BOUNDS,
                band_dtype=np.uint16,
                tile_store=TileStoreWithLayer(store, LAYER_NAME),
                projection=PROJECTION,
                resampling_method=Resampling.bilinear,
                remapper=None,
            )

    def test_all_items_dropped_returns_all_nodata(self, tmp_path: pathlib.Path) -> None:
        """If all scored items are dropped, FIRST_VALID should return nodata image."""
        store = DefaultTileStore()
        store.set_dataset_path(UPath(tmp_path))

        item_a = _make_item("a")
        item_b = _make_item("b")
        data_a = np.stack(
            [
                np.full((4, 4), 10, dtype=np.uint16),
                np.full((4, 4), 4, dtype=np.uint16),
            ],
            axis=0,
        )
        data_b = np.stack(
            [
                np.full((4, 4), 20, dtype=np.uint16),
                np.full((4, 4), 4, dtype=np.uint16),
            ],
            axis=0,
        )
        _write_items(store, [item_a, item_b], [data_a, data_b])

        compositor = Sentinel2SCLFirstValid(scl_band="SCL")

        def drop_item(*args: object, **kwargs: object) -> float | None:
            return None

        original_score_item = compositor._score_item
        compositor._score_item = drop_item  # type: ignore[method-assign]
        try:
            result = compositor.build_composite(
                group=[item_a, item_b],
                nodata_val=255,
                bands=OUTPUT_BANDS,
                bounds=BOUNDS,
                band_dtype=np.uint16,
                tile_store=TileStoreWithLayer(store, LAYER_NAME),
                projection=PROJECTION,
                resampling_method=Resampling.bilinear,
                remapper=None,
            )
        finally:
            compositor._score_item = original_score_item  # type: ignore[method-assign]

        assert np.all(result.get_chw_array() == 255)
        assert result.metadata.nodata_value == 255


class TestSentinel2SCLBestClear:
    """Tests for Sentinel2SCLBestClear compositor."""

    def test_selects_item_with_highest_clear_cover(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The selected item should maximize clear pixels over the window."""
        store = DefaultTileStore()
        store.set_dataset_path(UPath(tmp_path))

        item_less_clear = _make_item("less_clear")
        item_more_clear = _make_item("more_clear")

        less_clear_scl = np.array(
            [
                [4, 4, 4, 4],
                [4, 4, 4, 4],
                [9, 9, 9, 9],
                [9, 9, 9, 9],
            ],
            dtype=np.uint16,
        )
        more_clear_scl = np.array(
            [
                [4, 4, 4, 4],
                [4, 4, 4, 4],
                [4, 4, 4, 4],
                [9, 9, 9, 9],
            ],
            dtype=np.uint16,
        )
        less_clear_data = np.stack(
            [np.full((4, 4), 100, dtype=np.uint16), less_clear_scl], axis=0
        )
        more_clear_data = np.stack(
            [np.full((4, 4), 200, dtype=np.uint16), more_clear_scl], axis=0
        )
        _write_items(
            store,
            [item_less_clear, item_more_clear],
            [less_clear_data, more_clear_data],
        )

        compositor = Sentinel2SCLBestClear(scl_band="SCL")
        result = compositor.build_composite(
            group=[item_less_clear, item_more_clear],
            nodata_val=0,
            bands=OUTPUT_BANDS,
            bounds=BOUNDS,
            band_dtype=np.uint16,
            tile_store=TileStoreWithLayer(store, LAYER_NAME),
            projection=PROJECTION,
            resampling_method=Resampling.bilinear,
            remapper=None,
        )

        assert np.all(result.get_chw_array() == 200)

    def test_valid_cover_breaks_clear_cover_tie(self, tmp_path: pathlib.Path) -> None:
        """If clear cover ties, the item with better valid cover should win."""
        store = DefaultTileStore()
        store.set_dataset_path(UPath(tmp_path))

        item_low_valid = _make_item("low_valid")
        item_high_valid = _make_item("high_valid")

        low_valid_scl = np.array(
            [
                [4, 4, 4, 4],
                [4, 4, 4, 4],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint16,
        )
        high_valid_scl = np.array(
            [
                [4, 4, 4, 4],
                [4, 4, 4, 4],
                [9, 9, 9, 9],
                [9, 9, 9, 9],
            ],
            dtype=np.uint16,
        )
        low_valid_data = np.stack(
            [np.full((4, 4), 100, dtype=np.uint16), low_valid_scl], axis=0
        )
        high_valid_data = np.stack(
            [np.full((4, 4), 200, dtype=np.uint16), high_valid_scl], axis=0
        )
        _write_items(
            store,
            [item_low_valid, item_high_valid],
            [low_valid_data, high_valid_data],
        )

        compositor = Sentinel2SCLBestClear(scl_band="SCL")
        result = compositor.build_composite(
            group=[item_low_valid, item_high_valid],
            nodata_val=0,
            bands=OUTPUT_BANDS,
            bounds=BOUNDS,
            band_dtype=np.uint16,
            tile_store=TileStoreWithLayer(store, LAYER_NAME),
            projection=PROJECTION,
            resampling_method=Resampling.bilinear,
            remapper=None,
        )

        assert np.all(result.get_chw_array() == 200)

    def test_materializes_only_selected_item(self, tmp_path: pathlib.Path) -> None:
        """Lower-ranked items should not fill nodata pixels from the best scene."""
        store = DefaultTileStore()
        store.set_dataset_path(UPath(tmp_path))

        item_best = _make_item("best")
        item_fallback = _make_item("fallback")

        best_image = np.full((4, 4), 100, dtype=np.uint16)
        best_image[0, 0] = 0
        best_scl = np.full((4, 4), 4, dtype=np.uint16)
        fallback_image = np.full((4, 4), 200, dtype=np.uint16)
        fallback_scl = np.full((4, 4), 9, dtype=np.uint16)
        _write_items(
            store,
            [item_best, item_fallback],
            [
                np.stack([best_image, best_scl], axis=0),
                np.stack([fallback_image, fallback_scl], axis=0),
            ],
        )

        compositor = Sentinel2SCLBestClear(scl_band="SCL")
        result = compositor.build_composite(
            group=[item_best, item_fallback],
            nodata_val=0,
            bands=OUTPUT_BANDS,
            bounds=BOUNDS,
            band_dtype=np.uint16,
            tile_store=TileStoreWithLayer(store, LAYER_NAME),
            projection=PROJECTION,
            resampling_method=Resampling.bilinear,
            remapper=None,
        )

        expected = np.full((1, 4, 4), 100, dtype=np.uint16)
        expected[0, 0, 0] = 0
        assert np.all(result.get_chw_array() == expected)

    def test_custom_clear_values(self, tmp_path: pathlib.Path) -> None:
        """Configured clear values should control clear-cover scoring."""
        store = DefaultTileStore()
        store.set_dataset_path(UPath(tmp_path))

        item_default_clear = _make_item("default_clear")
        item_snow = _make_item("snow")
        default_clear_data = np.stack(
            [
                np.full((4, 4), 100, dtype=np.uint16),
                np.full((4, 4), 4, dtype=np.uint16),
            ],
            axis=0,
        )
        snow_data = np.stack(
            [
                np.full((4, 4), 200, dtype=np.uint16),
                np.full((4, 4), 11, dtype=np.uint16),
            ],
            axis=0,
        )
        _write_items(
            store,
            [item_default_clear, item_snow],
            [default_clear_data, snow_data],
        )

        compositor = Sentinel2SCLBestClear(scl_band="SCL", clear_values=[11])
        result = compositor.build_composite(
            group=[item_default_clear, item_snow],
            nodata_val=0,
            bands=OUTPUT_BANDS,
            bounds=BOUNDS,
            band_dtype=np.uint16,
            tile_store=TileStoreWithLayer(store, LAYER_NAME),
            projection=PROJECTION,
            resampling_method=Resampling.bilinear,
            remapper=None,
        )

        assert np.all(result.get_chw_array() == 200)

    def test_min_clear_fraction_filters_candidates(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Items below the valid-pixel clear fraction threshold should be skipped."""
        store = DefaultTileStore()
        store.set_dataset_path(UPath(tmp_path))

        item_more_clear_total = _make_item("more_clear_total")
        item_more_clear_valid = _make_item("more_clear_valid")

        more_clear_total_scl = np.array(
            [
                [4, 4, 4, 4],
                [4, 4, 4, 4],
                [4, 4, 4, 4],
                [9, 9, 9, 9],
            ],
            dtype=np.uint16,
        )
        more_clear_valid_scl = np.array(
            [
                [4, 4, 4, 4],
                [4, 4, 4, 4],
                [4, 9, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint16,
        )
        more_clear_total_data = np.stack(
            [np.full((4, 4), 100, dtype=np.uint16), more_clear_total_scl],
            axis=0,
        )
        more_clear_valid_data = np.stack(
            [np.full((4, 4), 200, dtype=np.uint16), more_clear_valid_scl],
            axis=0,
        )
        _write_items(
            store,
            [item_more_clear_total, item_more_clear_valid],
            [more_clear_total_data, more_clear_valid_data],
        )

        compositor = Sentinel2SCLBestClear(scl_band="SCL", min_clear_fraction=0.8)
        result = compositor.build_composite(
            group=[item_more_clear_total, item_more_clear_valid],
            nodata_val=0,
            bands=OUTPUT_BANDS,
            bounds=BOUNDS,
            band_dtype=np.uint16,
            tile_store=TileStoreWithLayer(store, LAYER_NAME),
            projection=PROJECTION,
            resampling_method=Resampling.bilinear,
            remapper=None,
        )

        assert np.all(result.get_chw_array() == 200)

    def test_min_valid_cover_filters_tiny_valid_footprints(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Valid-cover threshold should reject all-clear but tiny footprints."""
        store = DefaultTileStore()
        store.set_dataset_path(UPath(tmp_path))

        item_tiny_valid = _make_item("tiny_valid")
        item_enough_valid = _make_item("enough_valid")

        tiny_valid_scl = np.array(
            [
                [4, 4, 4, 4],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint16,
        )
        enough_valid_scl = np.array(
            [
                [4, 4, 9, 9],
                [9, 9, 9, 9],
                [9, 9, 9, 9],
                [9, 9, 9, 9],
            ],
            dtype=np.uint16,
        )
        tiny_valid_data = np.stack(
            [np.full((4, 4), 100, dtype=np.uint16), tiny_valid_scl], axis=0
        )
        enough_valid_data = np.stack(
            [np.full((4, 4), 200, dtype=np.uint16), enough_valid_scl], axis=0
        )
        _write_items(
            store,
            [item_tiny_valid, item_enough_valid],
            [tiny_valid_data, enough_valid_data],
        )

        compositor = Sentinel2SCLBestClear(scl_band="SCL", min_valid_cover=0.5)
        result = compositor.build_composite(
            group=[item_tiny_valid, item_enough_valid],
            nodata_val=0,
            bands=OUTPUT_BANDS,
            bounds=BOUNDS,
            band_dtype=np.uint16,
            tile_store=TileStoreWithLayer(store, LAYER_NAME),
            projection=PROJECTION,
            resampling_method=Resampling.bilinear,
            remapper=None,
        )

        assert np.all(result.get_chw_array() == 200)

    def test_all_items_below_min_clear_fraction_returns_nodata(
        self, tmp_path: pathlib.Path
    ) -> None:
        """If every candidate fails filtering, the output should be nodata."""
        store = DefaultTileStore()
        store.set_dataset_path(UPath(tmp_path))

        item_a = _make_item("a")
        item_b = _make_item("b")
        data_a = np.stack(
            [
                np.full((4, 4), 100, dtype=np.uint16),
                np.full((4, 4), 9, dtype=np.uint16),
            ],
            axis=0,
        )
        data_b = np.stack(
            [
                np.full((4, 4), 200, dtype=np.uint16),
                np.full((4, 4), 8, dtype=np.uint16),
            ],
            axis=0,
        )
        _write_items(store, [item_a, item_b], [data_a, data_b])

        compositor = Sentinel2SCLBestClear(scl_band="SCL", min_clear_fraction=0.1)
        result = compositor.build_composite(
            group=[item_a, item_b],
            nodata_val=255,
            bands=OUTPUT_BANDS,
            bounds=BOUNDS,
            band_dtype=np.uint16,
            tile_store=TileStoreWithLayer(store, LAYER_NAME),
            projection=PROJECTION,
            resampling_method=Resampling.bilinear,
            remapper=None,
        )

        assert np.all(result.get_chw_array() == 255)
        assert result.metadata.nodata_value == 255

    def test_invalid_thresholds_raise(self) -> None:
        """Thresholds should be fractions between zero and one."""
        with pytest.raises(ValueError, match="min_clear_fraction"):
            Sentinel2SCLBestClear(min_clear_fraction=-0.1)

        with pytest.raises(ValueError, match="min_valid_cover"):
            Sentinel2SCLBestClear(min_valid_cover=1.1)

        with pytest.raises(ValueError, match="nodata"):
            Sentinel2SCLBestClear(clear_values=[0, 4])

    def test_missing_scl_band_raises(self, tmp_path: pathlib.Path) -> None:
        """Scoring should fail if the configured SCL band is unavailable."""
        store = DefaultTileStore()
        store.set_dataset_path(UPath(tmp_path))

        item_a = _make_item("a")
        item_b = _make_item("b")
        for item, value in [(item_a, 10), (item_b, 20)]:
            data = np.full((1, 4, 4), value, dtype=np.uint16)  # only B04, no SCL
            store.write_raster(
                LAYER_NAME,
                item,
                OUTPUT_BANDS,
                PROJECTION,
                BOUNDS,
                RasterArray(chw_array=data),
            )

        compositor = Sentinel2SCLBestClear(scl_band="SCL")
        with pytest.raises(ValueError, match="missing scoring bands"):
            compositor.build_composite(
                group=[item_a, item_b],
                nodata_val=0,
                bands=OUTPUT_BANDS,
                bounds=BOUNDS,
                band_dtype=np.uint16,
                tile_store=TileStoreWithLayer(store, LAYER_NAME),
                projection=PROJECTION,
                resampling_method=Resampling.bilinear,
                remapper=None,
            )
