"""Tests for rslearn.dataset.tile_utils."""

import pathlib

import numpy as np
import pytest
from shapely.geometry import Polygon
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.data_source import Item
from rslearn.dataset.remap import LinearRemapper
from rslearn.dataset.tile_utils import read_raster_window_from_tiles
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.tile_stores.default import DefaultTileStore
from rslearn.utils.geometry import STGeometry
from rslearn.utils.raster_array import RasterArray

LAYER_NAME = "layer"
BOUNDS = (0, 0, 4, 4)
PROJECTION = WGS84_PROJECTION


@pytest.fixture
def tile_store(tmp_path: pathlib.Path) -> DefaultTileStore:
    store = DefaultTileStore()
    store.set_dataset_path(UPath(tmp_path))
    return store


def make_item(name: str) -> Item:
    return Item(
        name=name,
        geometry=STGeometry(
            projection=PROJECTION,
            shp=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            time_range=None,
        ),
    )


def write(
    tile_store: DefaultTileStore,
    item: Item,
    bands: list[str],
    data: np.ndarray,
    bounds: tuple[int, int, int, int] = BOUNDS,
) -> None:
    tile_store.write_raster(
        LAYER_NAME,
        item,
        bands,
        PROJECTION,
        bounds,
        RasterArray(chw_array=data),
    )


def read(
    tile_store: DefaultTileStore,
    item: Item,
    bands: list[str],
    nodata_val: int | float | None,
    band_dtype: type = np.uint8,
    dst: RasterArray | None = None,
    remapper: LinearRemapper | None = None,
) -> RasterArray | None:
    return read_raster_window_from_tiles(
        tile_store=TileStoreWithLayer(tile_store, LAYER_NAME),
        item=item,
        bands=bands,
        projection=PROJECTION,
        bounds=BOUNDS,
        nodata_val=nodata_val,
        band_dtype=band_dtype,
        remapper=remapper,
        dst=dst,
    )


def test_first_valid_only_overwrites_nodata(tile_store: DefaultTileStore) -> None:
    """A second item must only fill pixels still at nodata after the first."""
    bands = ["band1", "band2"]
    nodata_val = 0

    array1 = np.zeros((2, 4, 4), dtype=np.uint8)
    array1[:, :2, :] = 7  # top half valid
    array2 = np.full((2, 4, 4), 9, dtype=np.uint8)  # fully valid

    item1 = make_item("item1")
    item2 = make_item("item2")
    write(tile_store, item1, bands, array1)
    write(tile_store, item2, bands, array2)

    dst = read(tile_store, item1, bands, nodata_val)
    assert dst is not None
    dst = read(tile_store, item2, bands, nodata_val, dst=dst)
    assert dst is not None

    result = dst.get_chw_array()
    assert np.array_equal(result[:, :2, :], np.full((2, 2, 4), 7))  # kept
    assert np.array_equal(result[:, 2:, :], np.full((2, 2, 4), 9))  # filled


def test_partially_valid_pixel_is_not_overwritten(
    tile_store: DefaultTileStore,
) -> None:
    """First-valid masks on ALL bands being nodata: a pixel valid in one band only is kept."""
    bands = ["band1", "band2"]
    nodata_val = 0

    array1 = np.zeros((2, 4, 4), dtype=np.uint8)
    array1[0, :, :] = 5  # band1 valid everywhere; band2 all nodata
    array2 = np.full((2, 4, 4), 9, dtype=np.uint8)

    item1 = make_item("item1")
    item2 = make_item("item2")
    write(tile_store, item1, bands, array1)
    write(tile_store, item2, bands, array2)

    dst = read(tile_store, item1, bands, nodata_val)
    dst = read(tile_store, item2, bands, nodata_val, dst=dst)
    assert dst is not None

    result = dst.get_chw_array()
    # Pixel was not all-nodata, so item2 must not have touched either band.
    assert np.array_equal(result[0], np.full((4, 4), 5))
    assert np.array_equal(result[1], np.zeros((4, 4)))


def test_dtype_conversion(tile_store: DefaultTileStore) -> None:
    """Stored uint8 data must be converted to the requested band_dtype."""
    bands = ["band1"]
    item = make_item("item")
    write(tile_store, item, bands, np.full((1, 4, 4), 3, dtype=np.uint8))

    dst = read(tile_store, item, bands, nodata_val=0, band_dtype=np.float32)
    assert dst is not None
    assert dst.array.dtype == np.float32
    assert np.array_equal(dst.get_chw_array(), np.full((1, 4, 4), 3.0))


def test_no_nodata_overwrites_everything(tile_store: DefaultTileStore) -> None:
    """With nodata_val=None every source pixel overwrites the destination."""
    bands = ["band1"]
    item1 = make_item("item1")
    item2 = make_item("item2")
    write(tile_store, item1, bands, np.full((1, 4, 4), 5, dtype=np.uint8))
    write(tile_store, item2, bands, np.full((1, 4, 4), 9, dtype=np.uint8))

    dst = read(tile_store, item1, bands, nodata_val=None)
    dst = read(tile_store, item2, bands, nodata_val=None, dst=dst)
    assert dst is not None
    assert np.array_equal(dst.get_chw_array(), np.full((1, 4, 4), 9))


def test_band_reorder_across_band_sets(tile_store: DefaultTileStore) -> None:
    """Requested band order must be honored when bands live in separate band sets."""
    item = make_item("item")
    write(tile_store, item, ["band1"], np.full((1, 4, 4), 1, dtype=np.uint8))
    write(tile_store, item, ["band2"], np.full((1, 4, 4), 2, dtype=np.uint8))

    dst = read(tile_store, item, ["band2", "band1"], nodata_val=0)
    assert dst is not None
    result = dst.get_chw_array()
    assert np.array_equal(result[0], np.full((4, 4), 2))
    assert np.array_equal(result[1], np.full((4, 4), 1))


def test_partial_intersection_offset(tile_store: DefaultTileStore) -> None:
    """An item covering part of the window writes at the right offset, rest stays nodata."""
    bands = ["band1"]
    nodata_val = 0
    item = make_item("item")
    # Item only covers the bottom-right 2x2 of the 4x4 window.
    write(
        tile_store,
        item,
        bands,
        np.full((1, 2, 2), 8, dtype=np.uint8),
        bounds=(2, 2, 4, 4),
    )

    dst = read(tile_store, item, bands, nodata_val)
    assert dst is not None
    result = dst.get_chw_array()
    assert np.array_equal(result[0, 2:, 2:], np.full((2, 2), 8))
    assert np.array_equal(result[0, :2, :], np.zeros((2, 4)))
    assert np.array_equal(result[0, 2:, :2], np.zeros((2, 2)))


def test_remapper_applied(tile_store: DefaultTileStore) -> None:
    """The remapper must be applied to source values before writing."""
    bands = ["band1"]
    item = make_item("item")
    write(tile_store, item, bands, np.full((1, 4, 4), 100, dtype=np.uint8))

    remapper = LinearRemapper({"src": (0, 200), "dst": (0, 20)})
    dst = read(tile_store, item, bands, nodata_val=0, remapper=remapper)
    assert dst is not None
    assert np.array_equal(dst.get_chw_array(), np.full((1, 4, 4), 10))


def test_nan_nodata(tile_store: DefaultTileStore) -> None:
    """NaN nodata sentinels must be matched by the first-valid mask."""
    bands = ["band1"]
    item1 = make_item("item1")
    item2 = make_item("item2")
    array1 = np.full((1, 4, 4), np.nan, dtype=np.float32)
    array1[0, :2, :] = 1.0
    write(tile_store, item1, bands, array1)
    write(tile_store, item2, bands, np.full((1, 4, 4), 2.0, dtype=np.float32))

    dst = read(tile_store, item1, bands, nodata_val=float("nan"), band_dtype=np.float32)
    dst = read(
        tile_store,
        item2,
        bands,
        nodata_val=float("nan"),
        band_dtype=np.float32,
        dst=dst,
    )
    assert dst is not None
    result = dst.get_chw_array()
    assert np.array_equal(result[0, :2, :], np.full((2, 4), 1.0))
    assert np.array_equal(result[0, 2:, :], np.full((2, 4), 2.0))
