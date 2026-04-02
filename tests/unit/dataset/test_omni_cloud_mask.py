"""Tests for rslearn.dataset.omni_cloud_mask (OmniCloudMaskFirstValid compositor)."""

import pathlib
from typing import Any
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pytest
from rasterio.enums import Resampling
from shapely.geometry import Polygon
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.data_source import Item
from rslearn.dataset.omni_cloud_mask import OmniCloudMaskFirstValid
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.tile_stores.default import DefaultTileStore
from rslearn.utils.geometry import PixelBounds, STGeometry
from rslearn.utils.raster_array import RasterArray

LAYER_NAME = "layer"
BANDS = ["B04", "B03", "B8A"]
BOUNDS: PixelBounds = (0, 0, 4, 4)
PROJECTION = WGS84_PROJECTION


def _make_item(name: str) -> Item:
    bbox = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    return Item(name, STGeometry(PROJECTION, bbox, None))


class TestOmniCloudMaskFirstValid:
    """Tests for OmniCloudMaskFirstValid compositor with mocked omnicloudmask."""

    def _write_items(
        self,
        store: DefaultTileStore,
        items: list[Item],
        arrays: list[npt.NDArray[np.uint8]],
    ) -> None:
        for item, arr in zip(items, arrays):
            store.write_raster(
                LAYER_NAME, item, BANDS, PROJECTION, BOUNDS, RasterArray(chw_array=arr)
            )

    def test_sorts_items_by_cloud_score(self, tmp_path: pathlib.Path) -> None:
        """Items should be reordered so the least cloudy is first."""
        store = DefaultTileStore()
        store.set_dataset_path(UPath(tmp_path))

        item_cloudy = _make_item("cloudy")
        item_clear = _make_item("clear")

        cloudy_data = np.full((3, 4, 4), 100, dtype=np.uint8)
        clear_data = np.full((3, 4, 4), 200, dtype=np.uint8)
        self._write_items(store, [item_cloudy, item_clear], [cloudy_data, clear_data])

        compositor = OmniCloudMaskFirstValid(
            red_band=BANDS[0], green_band=BANDS[1], nir_band=BANDS[2]
        )

        # Mock predict_from_array: cloudy item gets thick cloud, clear item gets all clear.
        # Check the first real pixel (not padding) to distinguish items.
        def mock_predict(input_array: Any) -> np.ndarray:
            h, w = input_array.shape[1], input_array.shape[2]
            if input_array[0, 0, 0] == 100:
                return np.ones((h, w), dtype=np.uint8)  # class 1 = thick cloud
            return np.zeros((h, w), dtype=np.uint8)  # class 0 = clear

        with patch("rslearn.dataset.omni_cloud_mask.predict_from_array", mock_predict):
            # Pass cloudy first, clear second -- compositor should reorder.
            result = compositor.build_composite(
                group=[item_cloudy, item_clear],
                nodata_vals=[0, 0, 0],
                bands=BANDS,
                bounds=BOUNDS,
                band_dtype=np.uint8,
                tile_store=TileStoreWithLayer(store, LAYER_NAME),
                projection=PROJECTION,
                resampling_method=Resampling.bilinear,
                remapper=None,
            )

        # Clear item (200) should come first in FIRST_VALID compositing.
        assert np.all(result.get_chw_array() == 200)

    def test_single_item_no_scoring(self, tmp_path: pathlib.Path) -> None:
        """With a single item, no cloud scoring should happen."""
        store = DefaultTileStore()
        store.set_dataset_path(UPath(tmp_path))

        item = _make_item("single")
        data = np.full((3, 4, 4), 42, dtype=np.uint8)
        self._write_items(store, [item], [data])

        compositor = OmniCloudMaskFirstValid(
            red_band=BANDS[0], green_band=BANDS[1], nir_band=BANDS[2]
        )

        # predict_from_array should NOT be called for a single-item group.
        with patch(
            "rslearn.dataset.omni_cloud_mask.predict_from_array",
            side_effect=AssertionError("should not be called"),
        ):
            result = compositor.build_composite(
                group=[item],
                nodata_vals=[0, 0, 0],
                bands=BANDS,
                bounds=BOUNDS,
                band_dtype=np.uint8,
                tile_store=TileStoreWithLayer(store, LAYER_NAME),
                projection=PROJECTION,
                resampling_method=Resampling.bilinear,
                remapper=None,
            )

        assert np.all(result.get_chw_array() == 42)

    def test_scoring_failure_raises(self, tmp_path: pathlib.Path) -> None:
        """If scoring fails for an item, the error should propagate."""
        store = DefaultTileStore()
        store.set_dataset_path(UPath(tmp_path))

        item_ok = _make_item("ok")
        item_fail = _make_item("fail")

        ok_data = np.full((3, 4, 4), 50, dtype=np.uint8)
        fail_data = np.full((3, 4, 4), 99, dtype=np.uint8)
        self._write_items(store, [item_ok, item_fail], [ok_data, fail_data])

        compositor = OmniCloudMaskFirstValid(
            red_band=BANDS[0], green_band=BANDS[1], nir_band=BANDS[2]
        )

        def mock_predict(input_array: Any) -> np.ndarray:
            if input_array[0, 0, 0] == 99:
                raise RuntimeError("simulated failure")
            h, w = input_array.shape[1], input_array.shape[2]
            return np.zeros((h, w), dtype=np.uint8)

        with (
            patch("rslearn.dataset.omni_cloud_mask.predict_from_array", mock_predict),
            pytest.raises(RuntimeError, match="simulated failure"),
        ):
            compositor.build_composite(
                group=[item_fail, item_ok],
                nodata_vals=[0, 0, 0],
                bands=BANDS,
                bounds=BOUNDS,
                band_dtype=np.uint8,
                tile_store=TileStoreWithLayer(store, LAYER_NAME),
                projection=PROJECTION,
                resampling_method=Resampling.bilinear,
                remapper=None,
            )
