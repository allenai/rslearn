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
from rslearn.dataset.compositing import BandSetCompositeRequest
from rslearn.dataset.omni_cloud_mask import OmniCloudMaskFirstValid
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.tile_stores.default import DefaultTileStore
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry
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
                nodata_val=0,
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
                nodata_val=0,
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
                nodata_val=0,
                bands=BANDS,
                bounds=BOUNDS,
                band_dtype=np.uint8,
                tile_store=TileStoreWithLayer(store, LAYER_NAME),
                projection=PROJECTION,
                resampling_method=Resampling.bilinear,
                remapper=None,
            )

    def test_scoring_resolution_uses_window_grid_once_per_window(self) -> None:
        """Explicit scoring resolution should reuse one window-level scoring grid."""
        base_projection = Projection(PROJECTION.crs, 10, 10)
        base_bounds: PixelBounds = (0, 0, 8, 8)
        coarse_projection = Projection(PROJECTION.crs, 40, 40)
        coarse_bounds: PixelBounds = (0, 0, 2, 2)
        scoring_projection = Projection(PROJECTION.crs, 20, 20)
        scoring_bounds: PixelBounds = (0, 0, 4, 4)

        item_cloudy = _make_item("cloudy")
        item_clear = _make_item("clear")
        compositor = OmniCloudMaskFirstValid(
            red_band=BANDS[0],
            green_band=BANDS[1],
            nir_band=BANDS[2],
            min_inference_size=1,
            scoring_resolution=20,
        )

        scoring_reads: list[tuple[str, Projection, PixelBounds]] = []
        sorted_groups: list[list[str]] = []

        def mock_read_raster_window_from_tiles(
            tile_store: TileStoreWithLayer,
            item: Item,
            bands: list[str],
            projection: Projection,
            bounds: PixelBounds,
            nodata_val: int | float | None,
            band_dtype: npt.DTypeLike,
            remapper: Any = None,
            resampling: Resampling = Resampling.bilinear,
            dst: RasterArray | None = None,
        ) -> RasterArray:
            del tile_store, nodata_val, band_dtype, remapper, resampling, dst
            scoring_reads.append((item.name, projection, bounds))
            fill_value = 100 if item.name == "cloudy" else 200
            height = bounds[3] - bounds[1]
            width = bounds[2] - bounds[0]
            return RasterArray(
                array=np.full(
                    (len(bands), 1, height, width), fill_value, dtype=np.float32
                )
            )

        def mock_predict(input_array: Any) -> np.ndarray:
            if input_array.shape != (3, 4, 4):
                raise AssertionError(
                    f"expected 20 m scoring shape (3, 4, 4), got {input_array.shape}"
                )
            if input_array[0, 0, 0] == 100:
                return np.ones((4, 4), dtype=np.uint8)
            return np.zeros((4, 4), dtype=np.uint8)

        def mock_first_valid_build_composite(
            self: Any,
            group: list[Item],
            nodata_val: int | float | None,
            bands: list[str],
            bounds: PixelBounds,
            band_dtype: npt.DTypeLike,
            tile_store: TileStoreWithLayer,
            projection: Projection,
            resampling_method: Resampling,
            remapper: Any,
            request_time_range: tuple[Any, Any] | None = None,
        ) -> RasterArray:
            del (
                self,
                nodata_val,
                band_dtype,
                tile_store,
                projection,
                resampling_method,
                remapper,
                request_time_range,
            )
            sorted_groups.append([item.name for item in group])
            height = bounds[3] - bounds[1]
            width = bounds[2] - bounds[0]
            return RasterArray(
                array=np.zeros((len(bands), 1, height, width), dtype=np.uint8)
            )

        tile_store = TileStoreWithLayer(DefaultTileStore(), LAYER_NAME)
        with (
            patch(
                "rslearn.dataset.omni_cloud_mask.get_needed_band_sets_and_indexes",
                return_value=[(BANDS, [0, 1, 2], [0, 1, 2])],
            ),
            patch(
                "rslearn.dataset.omni_cloud_mask.read_raster_window_from_tiles",
                side_effect=mock_read_raster_window_from_tiles,
            ),
            patch("rslearn.dataset.omni_cloud_mask.predict_from_array", mock_predict),
            patch(
                "rslearn.dataset.omni_cloud_mask.FirstValidCompositor.build_composite",
                new=mock_first_valid_build_composite,
            ),
        ):
            compositor.build_composites(
                group=[item_cloudy, item_clear],
                requests=[
                    BandSetCompositeRequest(
                        nodata_val=0,
                        bands=["B04"],
                        bounds=coarse_bounds,
                        band_dtype=np.uint8,
                        projection=coarse_projection,
                        resampling_method=Resampling.bilinear,
                        remapper=None,
                    ),
                    BandSetCompositeRequest(
                        nodata_val=0,
                        bands=["B04"],
                        bounds=base_bounds,
                        band_dtype=np.uint8,
                        projection=base_projection,
                        resampling_method=Resampling.bilinear,
                        remapper=None,
                    ),
                ],
                tile_store=tile_store,
                window_projection=base_projection,
                window_bounds=base_bounds,
            )

        assert scoring_reads == [
            ("cloudy", scoring_projection, scoring_bounds),
            ("clear", scoring_projection, scoring_bounds),
        ]
        assert sorted_groups == [["clear", "cloudy"], ["clear", "cloudy"]]

    def test_default_scoring_uses_each_materialization_grid(self) -> None:
        """Unset scoring resolution should rank on each materialization grid."""
        base_projection = Projection(PROJECTION.crs, 10, 10)
        base_bounds: PixelBounds = (0, 0, 8, 8)
        coarse_projection = Projection(PROJECTION.crs, 40, 40)
        coarse_bounds: PixelBounds = (0, 0, 2, 2)

        item_cloudy = _make_item("cloudy")
        item_clear = _make_item("clear")
        compositor = OmniCloudMaskFirstValid(
            red_band=BANDS[0],
            green_band=BANDS[1],
            nir_band=BANDS[2],
            min_inference_size=1,
        )

        scoring_reads: list[tuple[str, Projection, PixelBounds]] = []

        def mock_read_raster_window_from_tiles(
            tile_store: TileStoreWithLayer,
            item: Item,
            bands: list[str],
            projection: Projection,
            bounds: PixelBounds,
            nodata_val: int | float | None,
            band_dtype: npt.DTypeLike,
            remapper: Any = None,
            resampling: Resampling = Resampling.bilinear,
            dst: RasterArray | None = None,
        ) -> RasterArray:
            del tile_store, nodata_val, band_dtype, remapper, resampling, dst
            scoring_reads.append((item.name, projection, bounds))
            fill_value = 100 if item.name == "cloudy" else 200
            height = bounds[3] - bounds[1]
            width = bounds[2] - bounds[0]
            return RasterArray(
                array=np.full(
                    (len(bands), 1, height, width), fill_value, dtype=np.float32
                )
            )

        def mock_predict(input_array: Any) -> np.ndarray:
            h, w = input_array.shape[1], input_array.shape[2]
            if input_array[0, 0, 0] == 100:
                return np.ones((h, w), dtype=np.uint8)
            return np.zeros((h, w), dtype=np.uint8)

        tile_store = TileStoreWithLayer(DefaultTileStore(), LAYER_NAME)
        with (
            patch(
                "rslearn.dataset.omni_cloud_mask.get_needed_band_sets_and_indexes",
                return_value=[(BANDS, [0, 1, 2], [0, 1, 2])],
            ),
            patch(
                "rslearn.dataset.omni_cloud_mask.read_raster_window_from_tiles",
                side_effect=mock_read_raster_window_from_tiles,
            ),
            patch("rslearn.dataset.omni_cloud_mask.predict_from_array", mock_predict),
            patch(
                "rslearn.dataset.omni_cloud_mask.FirstValidCompositor.build_composite",
                return_value=RasterArray(array=np.zeros((1, 1, 1, 1), dtype=np.uint8)),
            ),
        ):
            compositor.build_composites(
                group=[item_cloudy, item_clear],
                requests=[
                    BandSetCompositeRequest(
                        nodata_val=0,
                        bands=["B04"],
                        bounds=coarse_bounds,
                        band_dtype=np.uint8,
                        projection=coarse_projection,
                        resampling_method=Resampling.bilinear,
                        remapper=None,
                    ),
                    BandSetCompositeRequest(
                        nodata_val=0,
                        bands=["B04"],
                        bounds=base_bounds,
                        band_dtype=np.uint8,
                        projection=base_projection,
                        resampling_method=Resampling.bilinear,
                        remapper=None,
                    ),
                ],
                tile_store=tile_store,
                window_projection=base_projection,
                window_bounds=base_bounds,
            )

        assert scoring_reads == [
            ("cloudy", coarse_projection, coarse_bounds),
            ("clear", coarse_projection, coarse_bounds),
            ("cloudy", base_projection, base_bounds),
            ("clear", base_projection, base_bounds),
        ]

    def test_scoring_resolution_supports_non_b8a_sensors(self) -> None:
        """Explicit scoring resolution should also work for non-B8A sensors."""
        base_projection = Projection(PROJECTION.crs, 3, 3)
        base_bounds: PixelBounds = (0, 0, 20, 20)
        coarse_projection = Projection(PROJECTION.crs, 6, 6)
        coarse_bounds: PixelBounds = (0, 0, 10, 10)
        scoring_projection = Projection(PROJECTION.crs, 10, 10)
        scoring_bounds: PixelBounds = (0, 0, 6, 6)

        item_cloudy = _make_item("cloudy")
        item_clear = _make_item("clear")
        compositor = OmniCloudMaskFirstValid(
            red_band="B04",
            green_band="B03",
            nir_band="B08",
            min_inference_size=1,
            scoring_resolution=10,
        )

        scoring_reads: list[tuple[str, Projection, PixelBounds]] = []

        def mock_read_raster_window_from_tiles(
            tile_store: TileStoreWithLayer,
            item: Item,
            bands: list[str],
            projection: Projection,
            bounds: PixelBounds,
            nodata_val: int | float | None,
            band_dtype: npt.DTypeLike,
            remapper: Any = None,
            resampling: Resampling = Resampling.bilinear,
            dst: RasterArray | None = None,
        ) -> RasterArray:
            del tile_store, nodata_val, band_dtype, remapper, resampling, dst
            scoring_reads.append((item.name, projection, bounds))
            fill_value = 100 if item.name == "cloudy" else 200
            height = bounds[3] - bounds[1]
            width = bounds[2] - bounds[0]
            return RasterArray(
                array=np.full(
                    (len(bands), 1, height, width), fill_value, dtype=np.float32
                )
            )

        def mock_predict(input_array: Any) -> np.ndarray:
            if input_array.shape != (3, 6, 6):
                raise AssertionError(
                    f"expected explicit window scoring shape (3, 6, 6), got {input_array.shape}"
                )
            if input_array[0, 0, 0] == 100:
                return np.ones((6, 6), dtype=np.uint8)
            return np.zeros((6, 6), dtype=np.uint8)

        tile_store = TileStoreWithLayer(DefaultTileStore(), LAYER_NAME)
        with (
            patch(
                "rslearn.dataset.omni_cloud_mask.get_needed_band_sets_and_indexes",
                return_value=[(["B04", "B03", "B08"], [0, 1, 2], [0, 1, 2])],
            ),
            patch(
                "rslearn.dataset.omni_cloud_mask.read_raster_window_from_tiles",
                side_effect=mock_read_raster_window_from_tiles,
            ),
            patch("rslearn.dataset.omni_cloud_mask.predict_from_array", mock_predict),
            patch(
                "rslearn.dataset.omni_cloud_mask.FirstValidCompositor.build_composite",
                return_value=RasterArray(array=np.zeros((1, 1, 1, 1), dtype=np.uint8)),
            ),
        ):
            compositor.build_composites(
                group=[item_cloudy, item_clear],
                requests=[
                    BandSetCompositeRequest(
                        nodata_val=0,
                        bands=["B04"],
                        bounds=coarse_bounds,
                        band_dtype=np.uint8,
                        projection=coarse_projection,
                        resampling_method=Resampling.bilinear,
                        remapper=None,
                    ),
                    BandSetCompositeRequest(
                        nodata_val=0,
                        bands=["B04"],
                        bounds=base_bounds,
                        band_dtype=np.uint8,
                        projection=base_projection,
                        resampling_method=Resampling.bilinear,
                        remapper=None,
                    ),
                ],
                tile_store=tile_store,
                window_projection=base_projection,
                window_bounds=base_bounds,
            )

        assert scoring_reads == [
            ("cloudy", scoring_projection, scoring_bounds),
            ("clear", scoring_projection, scoring_bounds),
        ]

    def test_scoring_resolution_must_be_positive(self) -> None:
        """Explicit scoring_resolution should be positive."""
        with pytest.raises(ValueError, match="scoring_resolution must be positive"):
            OmniCloudMaskFirstValid(
                red_band="B04",
                green_band="B03",
                nir_band="B08",
                scoring_resolution=0,
            )
