"""Tests for rslearn.dataset.window_data_storage.per_item_group."""

import pathlib

import numpy as np
from shapely.geometry import box
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.dataset.storage.file import FileWindowStorage
from rslearn.dataset.window_data_storage.per_item_group import (
    PerItemGroupStorage,
    _per_item_group_layer_dir,
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
        data_storage=PerItemGroupStorage(),
    )
    window.save()
    return window


def test_raster_roundtrip(tmp_path: pathlib.Path) -> None:
    """Raster writes land at per-group on-disk paths and round-trip cleanly."""
    window = _make_window(tmp_path)
    storage = PerItemGroupStorage()
    raster_format = GeotiffRasterFormat()

    raster0 = RasterArray(
        chw_array=np.full((2, 4, 4), 1, dtype=np.uint8),
        metadata=RasterMetadata(nodata_value=0),
    )
    raster1 = RasterArray(
        chw_array=np.full((2, 4, 4), 7, dtype=np.uint8),
        metadata=RasterMetadata(nodata_value=0),
    )

    with storage.open_layer_writer(window, LAYER_NAME) as writer:
        writer.write_raster(
            BANDS, raster_format, PROJECTION, BOUNDS, raster0, group_idx=0
        )
        writer.write_raster(
            BANDS, raster_format, PROJECTION, BOUNDS, raster1, group_idx=1
        )

    # Files should land at layers/{LAYER_NAME}.{group_idx}/{bandset}/
    assert (window.window_root / "layers" / LAYER_NAME / "B1_B2").exists()
    assert (window.window_root / f"layers/{LAYER_NAME}.1/B1_B2").exists()

    out0 = storage.read_raster(
        window, LAYER_NAME, BANDS, raster_format, PROJECTION, BOUNDS, group_idx=0
    )
    out1 = storage.read_raster(
        window, LAYER_NAME, BANDS, raster_format, PROJECTION, BOUNDS, group_idx=1
    )
    assert np.all(out0.get_chw_array() == 1)
    assert np.all(out1.get_chw_array() == 7)

    all_rasters = storage.read_rasters(
        window, LAYER_NAME, BANDS, [0, 1], raster_format, PROJECTION, BOUNDS
    )
    assert len(all_rasters) == 2
    assert np.all(all_rasters[0].get_chw_array() == 1)
    assert np.all(all_rasters[1].get_chw_array() == 7)


def test_underscore_band_name(tmp_path: pathlib.Path) -> None:
    """A band name that contains an underscore still produces one bandset directory."""
    window = _make_window(tmp_path)
    storage = PerItemGroupStorage()
    raster_format = GeotiffRasterFormat()

    raster = RasterArray(
        chw_array=np.zeros((1, 4, 4), dtype=np.uint8),
        metadata=RasterMetadata(nodata_value=0),
    )
    with storage.open_layer_writer(window, "layer") as writer:
        writer.write_raster(["_"], raster_format, PROJECTION, (0, 0, 4, 4), raster)
    window.mark_layer_completed("layer")
    assert window.is_layer_completed("layer")

    # There should be exactly one bandset subfolder under the layer directory.
    subfolders = [
        entry
        for entry in _per_item_group_layer_dir(window.window_root, "layer").iterdir()
        if entry.is_dir()
    ]
    assert len(subfolders) == 1


def test_vector_roundtrip(tmp_path: pathlib.Path) -> None:
    """Vector features are written and read back per item group."""
    window = _make_window(tmp_path)
    storage = PerItemGroupStorage()
    vector_format = GeojsonVectorFormat()

    feat0 = Feature(
        geometry=STGeometry(PROJECTION, box(0, 0, 1, 1), None),
        properties={"label": "a"},
    )
    feat1 = Feature(
        geometry=STGeometry(PROJECTION, box(2, 2, 3, 3), None),
        properties={"label": "b"},
    )

    with storage.open_layer_writer(window, LAYER_NAME) as writer:
        writer.write_vector(vector_format, [feat0], group_idx=0)
        writer.write_vector(vector_format, [feat1], group_idx=1)

    out0 = storage.read_vector(
        window, LAYER_NAME, vector_format, PROJECTION, BOUNDS, group_idx=0
    )
    out1 = storage.read_vector(
        window, LAYER_NAME, vector_format, PROJECTION, BOUNDS, group_idx=1
    )
    assert len(out0) == 1 and out0[0].properties["label"] == "a"
    assert len(out1) == 1 and out1[0].properties["label"] == "b"
