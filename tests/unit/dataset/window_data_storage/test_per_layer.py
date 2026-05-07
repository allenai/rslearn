"""Tests for rslearn.dataset.window_data_storage.per_layer."""

import pathlib

import numpy as np
import pytest
from shapely.geometry import box
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.dataset.storage.file import FileWindowStorage
from rslearn.dataset.window_data_storage.per_layer import (
    PER_LAYER_STORAGE_META_FNAME,
    PerLayerStorage,
    PerLayerStorageFactory,
)
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import STGeometry
from rslearn.utils.raster_array import RasterArray, RasterMetadata
from rslearn.utils.raster_format import GeotiffRasterFormat
from rslearn.utils.vector_format import GeojsonVectorFormat

LAYER_NAME = "layer"
BANDS = ["B1", "B2"]
BOUNDS = (0, 0, 4, 4)
PROJECTION = WGS84_PROJECTION


def _make_window(tmp_path: pathlib.Path) -> Window:
    storage = FileWindowStorage(UPath(tmp_path / "dataset"))
    window = Window(
        storage=storage,
        group="default",
        name="w0",
        projection=PROJECTION,
        bounds=BOUNDS,
        time_range=None,
        data_storage=PerLayerStorage(),
    )
    window.save()
    return window


def test_factory_returns_per_layer_storage(tmp_path: pathlib.Path) -> None:
    """The factory returns a fresh PerLayerStorage."""
    storage = PerLayerStorageFactory().get_storage(UPath(tmp_path))
    assert isinstance(storage, PerLayerStorage)


def test_raster_roundtrip(tmp_path: pathlib.Path) -> None:
    """Per-layer raster writes produce a single combined file and round-trip cleanly."""
    window = _make_window(tmp_path)
    raster_format = GeotiffRasterFormat()

    rasters = [
        RasterArray(
            chw_array=np.full((2, 4, 4), i + 1, dtype=np.uint8),
            metadata=RasterMetadata(nodata_value=0),
        )
        for i in range(3)
    ]

    with window.open_layer_writer(LAYER_NAME) as writer:
        for i, raster in enumerate(rasters):
            writer.write_raster(
                BANDS, raster_format, PROJECTION, BOUNDS, raster, group_idx=i
            )

    # Single combined directory should exist; per-group dirs should not.
    layer_dir = window.window_root / "layers" / LAYER_NAME / "B1_B2"
    assert layer_dir.exists()
    assert (layer_dir / PER_LAYER_STORAGE_META_FNAME).exists()
    assert not (window.window_root / f"layers/{LAYER_NAME}.1/B1_B2").exists()
    assert not (window.window_root / f"layers/{LAYER_NAME}.2/B1_B2").exists()

    for i in range(3):
        arr = window.read_raster(LAYER_NAME, BANDS, raster_format, group_idx=i)
        assert np.all(arr.get_chw_array() == i + 1)

    all_rasters = window.read_rasters(LAYER_NAME, BANDS, [0, 1, 2], raster_format)
    assert len(all_rasters) == 3
    for i, arr in enumerate(all_rasters):
        assert np.all(arr.get_chw_array() == i + 1)


def test_groups_can_be_written_out_of_order(tmp_path: pathlib.Path) -> None:
    """Buffered groups are sorted by group_idx on flush."""
    window = _make_window(tmp_path)
    raster_format = GeotiffRasterFormat()

    raster_for_group_0 = RasterArray(
        chw_array=np.full((2, 4, 4), 9, dtype=np.uint8),
        metadata=RasterMetadata(nodata_value=0),
    )
    raster_for_group_1 = RasterArray(
        chw_array=np.full((2, 4, 4), 5, dtype=np.uint8),
        metadata=RasterMetadata(nodata_value=0),
    )

    with window.open_layer_writer(LAYER_NAME) as writer:
        writer.write_raster(
            BANDS, raster_format, PROJECTION, BOUNDS, raster_for_group_1, group_idx=1
        )
        writer.write_raster(
            BANDS, raster_format, PROJECTION, BOUNDS, raster_for_group_0, group_idx=0
        )

    arr0 = window.read_raster(LAYER_NAME, BANDS, raster_format, group_idx=0)
    arr1 = window.read_raster(LAYER_NAME, BANDS, raster_format, group_idx=1)
    assert np.all(arr0.get_chw_array() == 9)
    assert np.all(arr1.get_chw_array() == 5)


def test_inconsistent_bounds_rejected(tmp_path: pathlib.Path) -> None:
    """All groups must share the same bounds in PerLayerStorage."""
    window = _make_window(tmp_path)
    raster_format = GeotiffRasterFormat()
    other_bounds = (0, 0, 8, 8)
    raster_4 = RasterArray(
        chw_array=np.full((2, 4, 4), 1, dtype=np.uint8),
        metadata=RasterMetadata(nodata_value=0),
    )
    raster_8 = RasterArray(
        chw_array=np.full((2, 8, 8), 1, dtype=np.uint8),
        metadata=RasterMetadata(nodata_value=0),
    )
    with window.open_layer_writer(LAYER_NAME) as writer:
        writer.write_raster(
            BANDS, raster_format, PROJECTION, BOUNDS, raster_4, group_idx=0
        )
        with pytest.raises(ValueError, match="consistent bounds"):
            writer.write_raster(
                BANDS, raster_format, PROJECTION, other_bounds, raster_8, group_idx=1
            )


def test_vector_falls_back_to_per_item_group(tmp_path: pathlib.Path) -> None:
    """PerLayerStorage delegates vector ops to per-item-group on-disk layout."""
    window = _make_window(tmp_path)
    vector_format = GeojsonVectorFormat()

    feat0 = Feature(
        geometry=STGeometry(PROJECTION, box(0, 0, 1, 1), None),
        properties={"label": "a"},
    )
    feat1 = Feature(
        geometry=STGeometry(PROJECTION, box(2, 2, 3, 3), None),
        properties={"label": "b"},
    )

    with window.open_layer_writer(LAYER_NAME) as writer:
        writer.write_vector(vector_format, [feat0], group_idx=0)
        writer.write_vector(vector_format, [feat1], group_idx=1)

    # Vector data should land at the per-item-group dir, not at the per-layer dir.
    assert (window.window_root / "layers" / LAYER_NAME / "data.geojson").exists()
    assert (window.window_root / f"layers/{LAYER_NAME}.1" / "data.geojson").exists()

    out0 = window.read_vector(LAYER_NAME, vector_format, group_idx=0)
    out1 = window.read_vector(LAYER_NAME, vector_format, group_idx=1)
    assert len(out0) == 1 and out0[0].properties["label"] == "a"
    assert len(out1) == 1 and out1[0].properties["label"] == "b"


def test_writer_skips_flush_on_exception(tmp_path: pathlib.Path) -> None:
    """If the with-block raises, no combined file should be flushed."""
    window = _make_window(tmp_path)
    raster_format = GeotiffRasterFormat()

    raster = RasterArray(
        chw_array=np.full((2, 4, 4), 1, dtype=np.uint8),
        metadata=RasterMetadata(nodata_value=0),
    )

    class BoomError(RuntimeError):
        pass

    with pytest.raises(BoomError):
        with window.open_layer_writer(LAYER_NAME) as writer:
            writer.write_raster(BANDS, raster_format, PROJECTION, BOUNDS, raster)
            raise BoomError

    # Nothing should have been flushed.
    layer_dir = window.window_root / "layers" / LAYER_NAME / "B1_B2"
    assert not (layer_dir / PER_LAYER_STORAGE_META_FNAME).exists()
