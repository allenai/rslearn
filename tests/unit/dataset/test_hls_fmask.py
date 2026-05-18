"""Tests for rslearn.dataset.hls_fmask (HlsFmaskFirstValid compositor)."""

import pathlib

import numpy as np
import pytest
from rasterio.enums import Resampling
from shapely.geometry import Polygon
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.data_source import Item
from rslearn.dataset.hls_fmask import HlsFmaskFirstValid
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.tile_stores.default import DefaultTileStore
from rslearn.utils.geometry import PixelBounds, STGeometry
from rslearn.utils.raster_array import RasterArray

LAYER_NAME = "layer"
BOUNDS: PixelBounds = (0, 0, 4, 4)
PROJECTION = WGS84_PROJECTION
OUTPUT_BANDS = ["red"]
STORE_BANDS = OUTPUT_BANDS + ["fmask"]


def _make_item(name: str) -> Item:
    bbox = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    return Item(name, STGeometry(PROJECTION, bbox, None))


class TestHlsFmaskFirstValid:
    """Tests for HlsFmaskFirstValid compositor."""

    def _write_items(
        self,
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

    def test_sorts_items_by_fmask_cloud_score(self, tmp_path: pathlib.Path) -> None:
        """Items should be reordered so the least cloudy item is first."""
        store = DefaultTileStore()
        store.set_dataset_path(UPath(tmp_path))

        item_cloudy = _make_item("cloudy")
        item_clear = _make_item("clear")

        cloudy_data = np.stack(
            [
                np.full((4, 4), 100, dtype=np.uint16),  # red
                np.full((4, 4), 2, dtype=np.uint16),  # cloud bit set
            ],
            axis=0,
        )
        clear_data = np.stack(
            [
                np.full((4, 4), 200, dtype=np.uint16),  # red
                np.zeros((4, 4), dtype=np.uint16),  # clear
            ],
            axis=0,
        )
        self._write_items(store, [item_cloudy, item_clear], [cloudy_data, clear_data])

        compositor = HlsFmaskFirstValid()
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

        assert np.all(result.get_chw_array() == 200)

    def test_single_item_no_scoring(self, tmp_path: pathlib.Path) -> None:
        """With a single item, Fmask scoring should not run."""
        store = DefaultTileStore()
        store.set_dataset_path(UPath(tmp_path))

        item = _make_item("single")
        data = np.stack(
            [
                np.full((4, 4), 42, dtype=np.uint16),
                np.zeros((4, 4), dtype=np.uint16),
            ],
            axis=0,
        )
        self._write_items(store, [item], [data])

        compositor = HlsFmaskFirstValid()

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

    def test_missing_fmask_band_raises(self, tmp_path: pathlib.Path) -> None:
        """Scoring should fail if the configured Fmask band is unavailable."""
        store = DefaultTileStore()
        store.set_dataset_path(UPath(tmp_path))

        item_a = _make_item("a")
        item_b = _make_item("b")
        for item, value in [(item_a, 10), (item_b, 20)]:
            data = np.full((1, 4, 4), value, dtype=np.uint16)  # only red, no fmask
            store.write_raster(
                LAYER_NAME,
                item,
                OUTPUT_BANDS,
                PROJECTION,
                BOUNDS,
                RasterArray(chw_array=data),
            )

        compositor = HlsFmaskFirstValid()
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

    def test_missing_fmask_band_can_skip_item(self, tmp_path: pathlib.Path) -> None:
        """Missing Fmask can be configured to drop an item instead of raising."""
        store = DefaultTileStore()
        store.set_dataset_path(UPath(tmp_path))

        missing_fmask = _make_item("missing")
        clear = _make_item("clear")

        store.write_raster(
            LAYER_NAME,
            missing_fmask,
            OUTPUT_BANDS,
            PROJECTION,
            BOUNDS,
            RasterArray(chw_array=np.full((1, 4, 4), 25, dtype=np.uint16)),
        )
        self._write_items(
            store,
            [clear],
            [
                np.stack(
                    [
                        np.full((4, 4), 75, dtype=np.uint16),
                        np.zeros((4, 4), dtype=np.uint16),
                    ],
                    axis=0,
                )
            ],
        )

        compositor = HlsFmaskFirstValid(on_missing_fmask="skip_item")
        result = compositor.build_composite(
            group=[missing_fmask, clear],
            nodata_val=0,
            bands=OUTPUT_BANDS,
            bounds=BOUNDS,
            band_dtype=np.uint16,
            tile_store=TileStoreWithLayer(store, LAYER_NAME),
            projection=PROJECTION,
            resampling_method=Resampling.bilinear,
            remapper=None,
        )
        assert np.all(result.get_chw_array() == 75)

    def test_all_items_dropped_returns_all_nodata(self, tmp_path: pathlib.Path) -> None:
        """If all scored items are dropped, FIRST_VALID should return nodata image."""
        store = DefaultTileStore()
        store.set_dataset_path(UPath(tmp_path))

        item_a = _make_item("a")
        item_b = _make_item("b")
        data_a = np.stack(
            [
                np.full((4, 4), 10, dtype=np.uint16),
                np.zeros((4, 4), dtype=np.uint16),
            ],
            axis=0,
        )
        data_b = np.stack(
            [
                np.full((4, 4), 20, dtype=np.uint16),
                np.zeros((4, 4), dtype=np.uint16),
            ],
            axis=0,
        )
        self._write_items(store, [item_a, item_b], [data_a, data_b])

        compositor = HlsFmaskFirstValid()

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

    def test_cirrus_weight_affects_ranking(self, tmp_path: pathlib.Path) -> None:
        """Cirrus should affect ranking when cirrus_weight is non-zero."""
        store = DefaultTileStore()
        store.set_dataset_path(UPath(tmp_path))

        cirrus = _make_item("cirrus")
        clear = _make_item("clear")
        cirrus_data = np.stack(
            [
                np.full((4, 4), 100, dtype=np.uint16),
                np.full((4, 4), 0b00000001, dtype=np.uint16),  # cirrus bit set
            ],
            axis=0,
        )
        clear_data = np.stack(
            [
                np.full((4, 4), 200, dtype=np.uint16),
                np.full((4, 4), 0, dtype=np.uint16),
            ],
            axis=0,
        )
        self._write_items(store, [cirrus, clear], [cirrus_data, clear_data])

        compositor = HlsFmaskFirstValid(cloud_weight=0, cirrus_weight=1)
        result = compositor.build_composite(
            group=[cirrus, clear],
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
