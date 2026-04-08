"""Tests for rslearn.dataset.sentinel2_scl (Sentinel2SCLFirstValid compositor)."""

import pathlib

import numpy as np
import pytest
from rasterio.enums import Resampling
from shapely.geometry import Polygon
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.data_source import Item
from rslearn.dataset.sentinel2_scl import Sentinel2SCLFirstValid
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


class TestSentinel2SCLFirstValid:
    """Tests for Sentinel2SCLFirstValid compositor."""

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
        self._write_items(store, [item_cloudy, item_clear], [cloudy_data, clear_data])

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
        self._write_items(store, [item], [data])

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
