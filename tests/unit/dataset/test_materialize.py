"""Tests for rslearn.dataset.tile_utils (read_raster_window_from_tiles)."""

import pathlib

import numpy as np
import pytest
from shapely.geometry import Polygon
from upath import UPath

from rslearn.config import LayerConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.data_source import Item, ItemType
from rslearn.dataset import Window
from rslearn.dataset.compositing import BandSetCompositeRequest, Compositor
from rslearn.dataset.materialize import RasterMaterializer, resolve_nodata_value
from rslearn.dataset.storage.file import FileWindowStorage
from rslearn.dataset.tile_utils import read_raster_window_from_tiles
from rslearn.tile_stores.default import DefaultTileStore
from rslearn.tile_stores.tile_store import TileStoreWithLayer
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry
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
            nodata_val=0,
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

        With scalar nodata, dst is only overwritten where all bands equal nodata_val.
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

        nodata_val = 1
        dst_arr = np.zeros((2, 1, 4, 4), dtype=np.uint8)
        # Topleft 2x2: both bands nodata (1).
        dst_arr[0, 0, 0:2, 0:2] = 1
        dst_arr[1, 0, 0:2, 0:2] = 1
        # Top-right: band0 nodata, band1 not nodata.
        dst_arr[0, 0, 0:2, 2:4] = 1
        dst_arr[1, 0, 0:2, 2:4] = 7
        # Bottom-left: band0 not nodata, band1 nodata.
        dst_arr[0, 0, 2:4, 0:2] = 7
        dst_arr[1, 0, 2:4, 0:2] = 1
        # Bottom-right: neither band is nodata.
        dst_arr[0, 0, 2:4, 2:4] = 0
        dst_arr[1, 0, 2:4, 2:4] = 0
        dst = RasterArray(array=dst_arr)
        read_raster_window_from_tiles(
            tile_store=TileStoreWithLayer(tile_store, self.LAYER_NAME),
            item=item,
            bands=self.BANDS,
            projection=self.PROJECTION,
            bounds=self.BOUNDS,
            nodata_val=nodata_val,
            band_dtype=np.uint8,
            dst=dst,
        )
        result = dst.get_chw_array()
        # Top-right: band1 not nodata -> not overwritten.
        assert np.all(result[0, 0:2, 2:4] == 1)
        assert np.all(result[1, 0:2, 2:4] == 7)
        # Bottom-left: band0 not nodata -> not overwritten.
        assert np.all(result[0, 2:4, 0:2] == 7)
        assert np.all(result[1, 2:4, 0:2] == 1)
        # Topleft: both nodata -> updated to 3.
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
            nodata_val=np.nan,
            band_dtype=np.float32,
            dst=dst,
        )
        result = dst.get_chw_array()
        assert np.all(result[0, 0:2, :] == 1.0)
        assert np.all(result[0, 2:4, :] == 5.0)

    def test_none_nodata_unconditional_overwrite(self, tmp_path: pathlib.Path) -> None:
        """When nodata_val is None, source pixels overwrite dst unconditionally."""
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
            nodata_val=None,
            band_dtype=np.uint8,
            dst=dst,
        )
        result = dst.get_chw_array()
        assert np.all(result == 5)


class TestResolveNodataValue:
    """Tests for resolve_nodata_value."""

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
        """resolve_nodata_value should return None when source has no nodata."""
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
        result = resolve_nodata_value(
            TileStoreWithLayer(tile_store, self.LAYER_NAME),
            [item],
            self.BANDS,
        )
        assert result is None

    def test_returns_scalar_when_source_has_nodata(
        self, tmp_path: pathlib.Path
    ) -> None:
        """resolve_nodata_value returns the scalar when source metadata has nodata."""
        tile_store = DefaultTileStore()
        tile_store.set_dataset_path(UPath(tmp_path))
        item = self._make_item()
        nodata_meta = RasterMetadata(nodata_value=255.0)
        src = np.ones((2, 4, 4), dtype=np.uint8)
        tile_store.write_raster(
            self.LAYER_NAME,
            item,
            self.BANDS,
            self.PROJECTION,
            self.BOUNDS,
            RasterArray(chw_array=src, metadata=nodata_meta),
        )
        result = resolve_nodata_value(
            TileStoreWithLayer(tile_store, self.LAYER_NAME),
            [item],
            self.BANDS,
        )
        assert result == 255.0


class RecordingCompositor(Compositor):
    """Test compositor that records whole-window band-set requests."""

    def __init__(self) -> None:
        self.calls: list[
            tuple[
                list[ItemType],
                list[BandSetCompositeRequest],
                Projection | None,
                PixelBounds | None,
            ]
        ] = []

    def build_composites(
        self,
        group: list[ItemType],
        requests: list[BandSetCompositeRequest],
        tile_store: TileStoreWithLayer,
        window_projection: Projection | None = None,
        window_bounds: PixelBounds | None = None,
        request_time_range: tuple | None = None,
    ) -> list[RasterArray]:
        del tile_store, request_time_range
        self.calls.append((group, requests, window_projection, window_bounds))
        return [
            RasterArray(
                array=np.zeros(
                    (
                        len(request.bands),
                        1,
                        request.bounds[3] - request.bounds[1],
                        request.bounds[2] - request.bounds[0],
                    ),
                    dtype=request.band_dtype,
                )
            )
            for request in requests
        ]

    def build_composite(
        self,
        group: list[ItemType],
        nodata_val: int | float | None,
        bands: list[str],
        bounds: PixelBounds,
        band_dtype: np.dtype,
        tile_store: TileStoreWithLayer,
        projection: Projection,
        resampling_method,
        remapper,
        request_time_range: tuple | None = None,
    ) -> RasterArray:
        del (
            group,
            nodata_val,
            bands,
            bounds,
            band_dtype,
            tile_store,
            projection,
            resampling_method,
            remapper,
            request_time_range,
        )
        raise AssertionError(
            "build_composite should not be called by RasterMaterializer"
        )


def test_raster_materializer_passes_all_band_sets_to_compositor(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layer_cfg = LayerConfig.model_validate(
        {
            "type": "raster",
            "band_sets": [
                {"bands": ["B1"], "dtype": "uint8"},
                {"bands": ["B2"], "dtype": "uint8"},
            ],
        }
    )
    compositor = RecordingCompositor()
    monkeypatch.setattr(LayerConfig, "instantiate_compositor", lambda self: compositor)

    window = Window(
        storage=FileWindowStorage(UPath(tmp_path / "dataset")),
        group="default",
        name="window",
        projection=WGS84_PROJECTION,
        bounds=(0, 0, 4, 4),
        time_range=None,
    )
    window.save()

    tile_store = DefaultTileStore()
    tile_store.set_dataset_path(UPath(tmp_path / "tile_store"))

    RasterMaterializer().materialize(
        tile_store=TileStoreWithLayer(tile_store, "layer"),
        window=window,
        layer_name="layer",
        layer_cfg=layer_cfg,
        item_groups=[[]],
    )

    assert len(compositor.calls) == 1
    group, requests, window_projection, window_bounds = compositor.calls[0]
    assert group == []
    assert [request.bands for request in requests] == [["B1"], ["B2"]]
    assert window_projection == WGS84_PROJECTION
    assert window_bounds == (0, 0, 4, 4)
