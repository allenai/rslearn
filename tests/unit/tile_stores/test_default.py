import pathlib

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.tile_stores.default import DefaultTileStore
from rslearn.utils.fsspec import open_atomic
from rslearn.utils.geometry import Projection

LAYER_NAME = "layer"
ITEM_NAME = "item"
BANDS = ["B1"]
PROJECTION = Projection(CRS.from_epsg(3857), 1, -1)


@pytest.fixture
def tile_store_with_ones(tmp_path: pathlib.Path) -> DefaultTileStore:
    ds_path = UPath(tmp_path)
    tile_store = DefaultTileStore()
    tile_store.set_dataset_path(ds_path)
    # Write square.
    raster_size = 4
    tile_store.write_raster(
        LAYER_NAME,
        ITEM_NAME,
        BANDS,
        PROJECTION,
        (0, 0, raster_size, raster_size),
        np.ones((len(BANDS), raster_size, raster_size), dtype=np.uint8),
    )
    return tile_store


def test_rectangle_read(tile_store_with_ones: DefaultTileStore) -> None:
    # Make sure that when we read a rectangle with different width/height it returns
    # the right shape.
    width = 2
    height = 3
    result = tile_store_with_ones.read_raster(
        LAYER_NAME, ITEM_NAME, BANDS, PROJECTION, (0, 0, width, height)
    )
    assert result.shape == (len(BANDS), height, width)


def test_partial_read(tile_store_with_ones: DefaultTileStore) -> None:
    # Make sure that if we read an array that partially overlaps the raster, the
    # portion overlapping the raster has right value while the rest is zero.
    result = tile_store_with_ones.read_raster(
        LAYER_NAME, ITEM_NAME, BANDS, PROJECTION, (2, 2, 6, 6)
    )
    # This portion matches the raster which is all ones.
    assert np.all(result[:, 0:2, 0:2] == 1)
    # These portions do not.
    assert np.all(result[:, :, 2:4] == 0)
    assert np.all(result[:, 2:4, :] == 0)


def test_zstd_compression(tmp_path: pathlib.Path) -> None:
    # Make sure we can correctly write a GeoTIFF with ZSTD compression.
    ds_path = UPath(tmp_path)
    tile_store = DefaultTileStore(
        geotiff_options=dict(
            compress="zstd",
        )
    )
    tile_store.set_dataset_path(ds_path)
    raster_size = 4
    tile_store.write_raster(
        LAYER_NAME,
        ITEM_NAME,
        BANDS,
        PROJECTION,
        (0, 0, raster_size, raster_size),
        np.zeros((len(BANDS), raster_size, raster_size), dtype=np.uint8),
    )

    assert tile_store.path is not None
    fname = tile_store.path / LAYER_NAME / ITEM_NAME / "_".join(BANDS) / "geotiff.tif"
    with rasterio.open(fname) as raster:
        assert raster.profile["compress"] == "zstd"


def test_leftover_tmp_file(tmp_path: pathlib.Path) -> None:
    """Ensure that leftover files from open_atomic do not cause issues.

    Previously DefaultTileStore would raise error if there was one of these leftover
    files along with an actual raster written. Now the tmp files are ignored.
    """

    tile_store = DefaultTileStore()
    tile_store.set_dataset_path(UPath(tmp_path))
    raster_size = 4
    bounds = (0, 0, raster_size, raster_size)
    raster_dir = tile_store._get_raster_dir(LAYER_NAME, ITEM_NAME, BANDS)
    raster_dir.mkdir(parents=True)

    # Create the tmp file by writing halfway with open_atomic.
    class TestException(Exception):
        pass

    with pytest.raises(TestException):
        with open_atomic(raster_dir / "geotiff.tif", "wb") as f:
            f.write(b"123")
            raise TestException()

    # Double check that there is a tmp file.
    fnames = list(raster_dir.iterdir())
    assert len(fnames) == 1
    assert ".tmp." in fnames[0].name

    # Read should throw ValueError because there's no raster.
    with pytest.raises(ValueError):
        tile_store.read_raster(LAYER_NAME, ITEM_NAME, BANDS, PROJECTION, bounds)

    # Now write actual raster.
    tile_store.write_raster(
        LAYER_NAME,
        ITEM_NAME,
        BANDS,
        PROJECTION,
        bounds,
        np.ones((len(BANDS), raster_size, raster_size), dtype=np.uint8),
    )

    # And make sure this time the read succeeds.
    array = tile_store.read_raster(LAYER_NAME, ITEM_NAME, BANDS, PROJECTION, bounds)
    assert array.min() == 1 and array.max() == 1


class TestGetRasterBands:
    """Tests for DefaultTileStore.get_raster_bands."""

    def test_simple_bands(self, tmp_path: pathlib.Path) -> None:
        """Test with two bands with standard names."""
        tile_store = DefaultTileStore()
        tile_store.set_dataset_path(UPath(tmp_path))
        tile_store.write_raster(
            "layer",
            "item",
            ["B01", "B02"],
            WGS84_PROJECTION,
            (0, 0, 4, 4),
            np.ones((2, 4, 4), dtype=np.uint8),
        )
        assert tile_store.get_raster_bands("layer", "item") == [["B01", "B02"]]

    def test_band_with_underscore(self, tmp_path: pathlib.Path) -> None:
        """Verify that the tile store works when a band contains underscore.

        The tile store needs to use bands.json to check the band names, since it would
        not be encoded in the directory name.
        """
        tile_store = DefaultTileStore()
        tile_store.set_dataset_path(UPath(tmp_path))
        tile_store.write_raster(
            "layer",
            "item",
            ["B01", "_"],
            WGS84_PROJECTION,
            (0, 0, 4, 4),
            np.ones((2, 4, 4), dtype=np.uint8),
        )
        assert tile_store.get_raster_bands("layer", "item") == [["B01", "_"]]

    def test_multiple_files(self, tmp_path: pathlib.Path) -> None:
        """Test when there are multiple files with different subsets of bands."""
        tile_store = DefaultTileStore()
        tile_store.set_dataset_path(UPath(tmp_path))
        tile_store.write_raster(
            "layer",
            "item",
            ["B01", "B02"],
            WGS84_PROJECTION,
            (0, 0, 4, 4),
            np.ones((2, 4, 4), dtype=np.uint8),
        )
        tile_store.write_raster(
            "layer",
            "item",
            ["_"],
            WGS84_PROJECTION,
            (0, 0, 4, 4),
            np.ones((2, 4, 4), dtype=np.uint8),
        )
        assert list(sorted(tile_store.get_raster_bands("layer", "item"))) == list(
            sorted([["B01", "B02"], ["_"]])
        )
