"""Tests for rslearn.dataset.omni_cloud_mask (OmniCloudMaskFirstValid compositor)."""

import pathlib
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pytest
from rasterio.enums import Resampling
from shapely.geometry import Polygon
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.data_source import Item
from rslearn.dataset import Window
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

    @staticmethod
    def _make_request(bounds: PixelBounds = BOUNDS) -> BandSetCompositeRequest:
        return BandSetCompositeRequest(
            nodata_val=0,
            bands=BANDS,
            bounds=bounds,
            band_dtype=np.uint8,
            projection=PROJECTION,
            resampling_method=Resampling.bilinear,
            remapper=None,
        )

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

        # Mock predict_from_batch: cloudy item gets thick cloud, clear item gets all clear.
        # Check the first real pixel (not padding) to distinguish items.
        def mock_predict_one(input_array: Any) -> np.ndarray:
            h, w = input_array.shape[1], input_array.shape[2]
            if input_array[0, 0, 0] == 100:
                return np.ones((h, w), dtype=np.uint8)  # class 1 = thick cloud
            return np.zeros((h, w), dtype=np.uint8)  # class 0 = clear

        def mock_predict(data: Any, batch_size: int) -> np.ndarray:
            return np.stack([mock_predict_one(arr)[None] for arr in data])

        with patch("rslearn.dataset.omni_cloud_mask.predict_from_batch", mock_predict):
            # Pass cloudy first, clear second -- compositor should reorder.
            result = list(
                compositor.build_composites(
                    group=[item_cloudy, item_clear],
                    requests=[self._make_request()],
                    tile_store=TileStoreWithLayer(store, LAYER_NAME),
                )
            )[0]

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

        # predict_from_batch should NOT be called for a single-item group.
        with patch(
            "rslearn.dataset.omni_cloud_mask.predict_from_batch",
            side_effect=AssertionError("should not be called"),
        ):
            result = list(
                compositor.build_composites(
                    group=[item],
                    requests=[self._make_request()],
                    tile_store=TileStoreWithLayer(store, LAYER_NAME),
                )
            )[0]

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

        def mock_predict_one(input_array: Any) -> np.ndarray:
            if input_array[0, 0, 0] == 99:
                raise RuntimeError("simulated failure")
            h, w = input_array.shape[1], input_array.shape[2]
            return np.zeros((h, w), dtype=np.uint8)

        def mock_predict(data: Any, batch_size: int) -> np.ndarray:
            return np.stack([mock_predict_one(arr)[None] for arr in data])

        with (
            patch(
                "rslearn.dataset.omni_cloud_mask.predict_from_batch",
                side_effect=mock_predict,
            ),
            pytest.raises(RuntimeError, match="simulated failure"),
        ):
            list(
                compositor.build_composites(
                    group=[item_fail, item_ok],
                    requests=[self._make_request()],
                    tile_store=TileStoreWithLayer(store, LAYER_NAME),
                )
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
        window = cast(
            Window,
            SimpleNamespace(projection=base_projection, bounds=base_bounds),
        )
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

        def mock_predict_one(input_array: Any) -> np.ndarray:
            if input_array.shape != (3, 4, 4):
                raise AssertionError(
                    f"expected 20 m scoring shape (3, 4, 4), got {input_array.shape}"
                )
            if input_array[0, 0, 0] == 100:
                return np.ones((4, 4), dtype=np.uint8)
            return np.zeros((4, 4), dtype=np.uint8)

        def mock_predict(data: Any, batch_size: int) -> np.ndarray:
            return np.stack([mock_predict_one(arr)[None] for arr in data])

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
            patch(
                "rslearn.dataset.omni_cloud_mask.predict_from_batch",
                side_effect=mock_predict,
            ) as predict,
            patch(
                "rslearn.dataset.omni_cloud_mask.FirstValidCompositor.build_composite",
                new=mock_first_valid_build_composite,
            ),
        ):
            list(
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
                    window=window,
                )
            )

        assert scoring_reads == [
            ("cloudy", scoring_projection, scoring_bounds),
            ("clear", scoring_projection, scoring_bounds),
        ]
        assert sorted_groups == [["clear", "cloudy"], ["clear", "cloudy"]]
        predict.assert_called_once()

    def test_default_scoring_uses_each_materialization_grid(self) -> None:
        """Unset scoring resolution should rank on each materialization grid."""
        base_projection = Projection(PROJECTION.crs, 10, 10)
        base_bounds: PixelBounds = (0, 0, 8, 8)
        coarse_projection = Projection(PROJECTION.crs, 40, 40)
        coarse_bounds: PixelBounds = (0, 0, 2, 2)

        item_cloudy = _make_item("cloudy")
        item_clear = _make_item("clear")
        window = cast(
            Window,
            SimpleNamespace(projection=base_projection, bounds=base_bounds),
        )
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

        def mock_predict_one(input_array: Any) -> np.ndarray:
            h, w = input_array.shape[1], input_array.shape[2]
            if input_array[0, 0, 0] == 100:
                return np.ones((h, w), dtype=np.uint8)
            return np.zeros((h, w), dtype=np.uint8)

        def mock_predict(data: Any, batch_size: int) -> np.ndarray:
            return np.stack([mock_predict_one(arr)[None] for arr in data])

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
            patch(
                "rslearn.dataset.omni_cloud_mask.predict_from_batch",
                side_effect=mock_predict,
            ) as predict,
            patch(
                "rslearn.dataset.omni_cloud_mask.FirstValidCompositor.build_composite",
                return_value=RasterArray(array=np.zeros((1, 1, 1, 1), dtype=np.uint8)),
            ),
        ):
            list(
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
                    window=window,
                )
            )

        assert scoring_reads == [
            ("cloudy", coarse_projection, coarse_bounds),
            ("clear", coarse_projection, coarse_bounds),
            ("cloudy", base_projection, base_bounds),
            ("clear", base_projection, base_bounds),
        ]

        assert predict.call_count == 2

    def test_scoring_resolution_must_be_positive(self) -> None:
        """Explicit scoring_resolution should be positive."""
        with pytest.raises(ValueError, match="scoring_resolution must be positive"):
            OmniCloudMaskFirstValid(
                red_band="B04",
                green_band="B03",
                nir_band="B08",
                scoring_resolution=0,
            )

    @pytest.mark.parametrize("shape", [(96, 48), (48, 96), (8, 16), (16, 8)])
    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_batch_scoring_crops_spatial_axes(
        self, shape: tuple[int, int], batch_size: int
    ) -> None:
        """Clouds below the first width rows and padding must not change the crop."""
        height, width = shape
        items = [_make_item("half_cloudy"), _make_item("thin_cloud")]
        arrays = [np.full((3, height, width), v, dtype=np.float32) for v in [100, 200]]
        compositor = OmniCloudMaskFirstValid(batch_size=batch_size)

        def mock_predict(data: Any, batch_size: int) -> np.ndarray:
            assert batch_size == compositor.batch_size
            assert len(data) == 2
            h, w = max(height, 32), max(width, 32)
            assert all(arr.shape == (3, h, w) for arr in data)
            for arr, original in zip(data, arrays, strict=True):
                np.testing.assert_array_equal(arr[:, :height, :width], original)
                assert not arr[:, height:, :].any()
                assert not arr[:, :, width:].any()
            masks = np.zeros((2, 1, h, w), dtype=np.uint8)
            masks[0, 0, height // 2 : height, :width] = 1
            # Give padding a different class to detect inclusion in the score.
            masks[1] = 1
            masks[1, 0, :height, :width] = 2
            return masks

        with (
            patch(
                "rslearn.dataset.omni_cloud_mask.get_needed_band_sets_and_indexes",
                return_value=[(BANDS, [0, 1, 2], [0, 1, 2])],
            ),
            patch(
                "rslearn.dataset.omni_cloud_mask.read_raster_window_from_tiles",
                side_effect=[RasterArray(chw_array=arr) for arr in arrays],
            ),
            patch(
                "rslearn.dataset.omni_cloud_mask.predict_from_batch",
                side_effect=mock_predict,
            ) as predict,
        ):
            result = compositor._sort_group(
                items,
                TileStoreWithLayer(DefaultTileStore(), LAYER_NAME),
                PROJECTION,
                (0, 0, width, height),
                Resampling.bilinear,
            )
        assert result == items[::-1]
        predict.assert_called_once()

    def test_partial_coverage_does_not_score_nodata_as_clear(self) -> None:
        items = [_make_item("partial_cloudy"), _make_item("thin_cloud")]
        # A pixel is valid even if only one of the scoring bands is nonzero.
        partial = np.zeros((3, 4, 4), dtype=np.float32)
        partial[1, 0, 0] = 100
        full = np.full((3, 4, 4), 200, dtype=np.float32)
        masks = np.zeros((2, 1, 32, 32), dtype=np.uint8)
        masks[0, 0, 0, 0] = 1
        masks[1, 0, :4, :4] = 2
        with (
            patch(
                "rslearn.dataset.omni_cloud_mask.get_needed_band_sets_and_indexes",
                return_value=[(BANDS, [0, 1, 2], [0, 1, 2])],
            ),
            patch(
                "rslearn.dataset.omni_cloud_mask.read_raster_window_from_tiles",
                side_effect=[RasterArray(chw_array=arr) for arr in [partial, full]],
            ),
            patch(
                "rslearn.dataset.omni_cloud_mask.predict_from_batch", return_value=masks
            ),
        ):
            result = OmniCloudMaskFirstValid()._sort_group(
                items,
                TileStoreWithLayer(DefaultTileStore(), LAYER_NAME),
                PROJECTION,
                BOUNDS,
                Resampling.bilinear,
            )
        assert result == items[::-1]

    @pytest.mark.parametrize("has_valid_items", [False, True])
    def test_skips_missing_and_all_nodata_items(self, has_valid_items: bool) -> None:
        items = [_make_item(name) for name in ["missing", "nodata", "first", "second"]]
        reads = [None, RasterArray(chw_array=np.zeros((3, 4, 4), dtype=np.float32))]
        if has_valid_items:
            reads.extend(
                [RasterArray(chw_array=np.ones((3, 4, 4), dtype=np.float32))] * 2
            )
        else:
            items = items[:2]
        with (
            patch(
                "rslearn.dataset.omni_cloud_mask.get_needed_band_sets_and_indexes",
                return_value=[(BANDS, [0, 1, 2], [0, 1, 2])],
            ),
            patch(
                "rslearn.dataset.omni_cloud_mask.read_raster_window_from_tiles",
                side_effect=reads,
            ),
            patch(
                "rslearn.dataset.omni_cloud_mask.predict_from_batch",
                return_value=np.zeros((2, 1, 32, 32), dtype=np.uint8),
            ) as predict,
        ):
            result = OmniCloudMaskFirstValid()._sort_group(
                items,
                TileStoreWithLayer(DefaultTileStore(), LAYER_NAME),
                PROJECTION,
                BOUNDS,
                Resampling.bilinear,
            )
        # Equal scores preserve input order, including after skipping missing items.
        assert result == items[2:]
        if has_valid_items:
            predict.assert_called_once()
            assert len(predict.call_args.args[0]) == 2
        else:
            predict.assert_not_called()

    def test_custom_class_weights(self) -> None:
        compositor = OmniCloudMaskFirstValid(
            clear_weight=2,
            thick_cloud_weight=8,
            thin_cloud_weight=4,
            cloud_shadow_weight=6,
        )
        mask = np.array([[[0, 1, 2, 3, 0]]], dtype=np.uint8)
        valid = np.array([[True, True, True, True, False]])
        assert compositor._score_mask(_make_item("weighted"), mask, valid) == 5

    @pytest.mark.parametrize("batch_size", [0, -1])
    def test_batch_size_must_be_positive(self, batch_size: int) -> None:
        with pytest.raises(ValueError, match="batch_size must be positive"):
            OmniCloudMaskFirstValid(batch_size=batch_size)

    def test_missing_scoring_bands_raises(self) -> None:
        with (
            patch(
                "rslearn.dataset.omni_cloud_mask.get_needed_band_sets_and_indexes",
                return_value=[],
            ),
            patch("rslearn.dataset.omni_cloud_mask.predict_from_batch") as predict,
            pytest.raises(ValueError, match="missing scoring bands.*missing_bands"),
        ):
            OmniCloudMaskFirstValid()._sort_group(
                [_make_item("missing_bands")],
                TileStoreWithLayer(DefaultTileStore(), LAYER_NAME),
                PROJECTION,
                BOUNDS,
                Resampling.bilinear,
            )
        predict.assert_not_called()
