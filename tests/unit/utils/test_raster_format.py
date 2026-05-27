import pathlib
from datetime import UTC, datetime

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from upath import UPath

from rslearn.utils.geometry import Projection
from rslearn.utils.raster_array import RasterArray, RasterMetadata
from rslearn.utils.raster_format import GeotiffRasterFormat, NumpyRasterFormat


def test_geotiff_tiling(tmp_path: pathlib.Path) -> None:
    path = UPath(tmp_path)
    block_size = 128
    projection = Projection(CRS.from_epsg(3857), 1, -1)

    # If always_enable_tiling=False, it should create tiled GeoTIFF only if one
    # of the dimensions exceeds the block size.
    # For some reason the GeoTIFF still ends up being tiled if the dimensions are the
    # same in some cases so here we set them different.
    array = np.zeros((1, 60, 64), dtype=np.uint8)
    GeotiffRasterFormat(
        block_size=block_size, always_enable_tiling=False
    ).encode_raster(path, projection, (0, 0, 64, 60), RasterArray(chw_array=array))
    with (path / "geotiff.tif").open("rb") as f:
        with rasterio.open(f) as raster:
            assert not raster.profile["tiled"]

    array = np.zeros((1, 252, 256), dtype=np.uint8)
    GeotiffRasterFormat(
        block_size=block_size, always_enable_tiling=False
    ).encode_raster(path, projection, (0, 0, 256, 252), RasterArray(chw_array=array))
    with (path / "geotiff.tif").open("rb") as f:
        with rasterio.open(f) as raster:
            assert raster.profile["tiled"]

    # If always_enable_tiling=True it should create tiled GeoTIFF either way.
    array = np.zeros((1, 60, 64), dtype=np.uint8)
    GeotiffRasterFormat(block_size=block_size, always_enable_tiling=True).encode_raster(
        path, projection, (0, 0, 64, 60), RasterArray(chw_array=array)
    )
    with (path / "geotiff.tif").open("rb") as f:
        with rasterio.open(f) as raster:
            assert raster.profile["tiled"]


class TestGeotiffBoundsAndNodata:
    """Test GeotiffRasterFormat bounds and nodata handling."""

    PROJECTION = Projection(CRS.from_epsg(3857), 1, -1)

    @pytest.fixture
    def encoded_raster_path(self, tmp_path: pathlib.Path) -> UPath:
        path = UPath(tmp_path)
        array = np.ones((1, 8, 8), dtype=np.uint8)
        GeotiffRasterFormat().encode_raster(
            path, self.PROJECTION, (0, 0, 8, 8), RasterArray(chw_array=array)
        )
        return path

    def test_geotiff_in_bounds(self, encoded_raster_path: UPath) -> None:
        """In-bounds read from GeoTIFF of 1s should yield 1s."""
        ra = GeotiffRasterFormat().decode_raster(
            encoded_raster_path, self.PROJECTION, (2, 2, 6, 6)
        )
        assert ra.array.shape == (1, 1, 4, 4)
        assert np.all(ra.array == 1)

    def test_geotiff_partial_overlap(self, encoded_raster_path: UPath) -> None:
        """Read with partial overlap should yield nodata (0) for out of bounds pixels."""
        ra = GeotiffRasterFormat().decode_raster(
            encoded_raster_path, self.PROJECTION, (4, 4, 12, 12)
        )
        assert ra.array.shape == (1, 1, 8, 8)
        assert np.all(ra.array[:, :, 0:4, 0:4] == 1)
        assert np.all(ra.array[:, :, 0:8, 4:8] == 0)

    def test_geotiff_out_of_bounds(self, encoded_raster_path: UPath) -> None:
        """Fully out-of-bounds read should yield nodata (0) for all pixels."""
        ra = GeotiffRasterFormat().decode_raster(
            encoded_raster_path, self.PROJECTION, (8, 8, 12, 12)
        )
        assert ra.array.shape == (1, 1, 4, 4)
        assert np.all(ra.array == 0)

    def test_geotiff_write_nodata_value(self, tmp_path: pathlib.Path) -> None:
        """Test that nodata_value on metadata is correctly written to the GeoTIFF."""
        path = UPath(tmp_path)
        projection = Projection(CRS.from_epsg(3857), 1, -1)
        array = np.zeros((1, 4, 4), dtype=np.float32)
        nodata_val = -9999.0

        GeotiffRasterFormat().encode_raster(
            path,
            projection,
            (0, 0, 4, 4),
            RasterArray(
                chw_array=array,
                metadata=RasterMetadata(nodata_value=nodata_val),
            ),
        )

        with rasterio.open(path / "geotiff.tif") as raster:
            assert raster.nodata == nodata_val

    def test_geotiff_out_of_bounds_uses_source_nodata(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Out-of-bounds pixels should use the source file's nodata value when nodata_val is not set.

        This verifies the source nodata value is honored even when it is non-zero.
        """
        # Create a GeoTIFF that is 42 in (0, 0, 4, 4), with nodata value 255.
        path = UPath(tmp_path)
        projection = Projection(CRS.from_epsg(3857), 1, -1)
        array = np.full((1, 4, 4), 42, dtype=np.uint8)
        GeotiffRasterFormat().encode_raster(
            path,
            projection,
            (0, 0, 4, 4),
            RasterArray(chw_array=array, metadata=RasterMetadata(nodata_value=255)),
        )

        # Read partially out of bounds without specifying nodata_val.
        raster = GeotiffRasterFormat().decode_raster(path, projection, (2, 2, 8, 8))
        result = raster.get_chw_array()
        assert result.shape == (1, 6, 6)
        # In-bounds region should keep original data.
        assert np.all(result[:, 0:2, 0:2] == 42)
        # Out-of-bounds region should be filled with the source nodata (255).
        assert np.all(result[:, 2:6, :] == 255)
        assert np.all(result[:, :, 2:6] == 255)
        # nodata_value should be populated from the GeoTIFF.
        assert raster.metadata.nodata_value == 255

    def test_geotiff_read_nodata_val_out_of_bounds(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Out-of-bounds pixels are filled with the nodata_val override on read."""
        path = UPath(tmp_path)
        projection = Projection(CRS.from_epsg(3857), 1, -1)
        nodata_val = -9999.0

        # Create a raster with value 1 and nodata=0.
        array = np.ones((1, 4, 4), dtype=np.float32)
        GeotiffRasterFormat().encode_raster(
            path,
            projection,
            (0, 0, 4, 4),
            RasterArray(
                chw_array=array,
                metadata=RasterMetadata(nodata_value=0),
            ),
        )

        # Read a region that partially overlaps the source raster.
        # We override the nodata_val to -9999.
        # In the out-of-bounds portions it should be filled in as -9999.
        raster_array = GeotiffRasterFormat().decode_raster(
            path, projection, (2, 2, 8, 8), nodata_val=nodata_val
        )

        assert raster_array.array.shape == (1, 1, 6, 6)
        result = raster_array.get_chw_array()
        # Top-left 2x2 region overlaps source and should have value 1.
        assert np.all(result[:, 0:2, 0:2] == 1)
        # Pixels outside source bounds should be filled with nodata_val.
        assert np.all(result[:, 2:6, :] == nodata_val)
        assert np.all(result[:, :, 2:6] == nodata_val)
        # The nodata on the RasterArray should also be overridden to nodata_val instead of 0.
        assert raster_array.metadata.nodata_value == nodata_val

    def test_geotiff_read_nodata_val_orig_nodata(self, tmp_path: pathlib.Path) -> None:
        """Test reading from GeoTIFF that has nodata values, but we set different nodata_val.

        Since we override the nodata value in the read operation, the source pixels that
        were a different nodata value should still have the original value.
        """
        path = UPath(tmp_path)
        projection = Projection(CRS.from_epsg(3857), 1, -1)
        original_nodata = -9999.0
        new_nodata = -1.0

        # Create a raster where some pixels have the nodata value -9999.
        # Other pixels are valid (1).
        array = np.ones((1, 4, 4), dtype=np.float32)
        array[:, 2:4, 2:4] = original_nodata
        GeotiffRasterFormat().encode_raster(
            path,
            projection,
            (0, 0, 4, 4),
            RasterArray(
                chw_array=array,
                metadata=RasterMetadata(nodata_value=original_nodata),
            ),
        )

        # Decode with a different nodata_val.
        # The pixels that were originally nodata should be unchanged.
        raster_array = GeotiffRasterFormat().decode_raster(
            path, projection, (0, 0, 4, 4), nodata_val=new_nodata
        )

        assert raster_array.array.shape == (1, 1, 4, 4)
        result = raster_array.get_chw_array()
        # Valid data pixels should still have value 1.
        assert np.all(result[:, 0:2, :] == 1)
        assert np.all(result[:, :, 0:2] == 1)
        # Original nodata pixels should still be as they were.
        assert np.all(result[:, 2:4, 2:4] == original_nodata)
        # The RasterArray should use the nodata value from the override.
        assert raster_array.metadata.nodata_value == new_nodata

    def test_geotiff_decode_no_nodata(self, tmp_path: pathlib.Path) -> None:
        """GeoTIFF without nodata should decode with nodata_value=None."""
        path = UPath(tmp_path)
        projection = Projection(CRS.from_epsg(3857), 1, -1)
        array = np.ones((1, 4, 4), dtype=np.uint8)
        GeotiffRasterFormat().encode_raster(
            path, projection, (0, 0, 4, 4), RasterArray(chw_array=array)
        )
        decoded = GeotiffRasterFormat().decode_raster(path, projection, (0, 0, 4, 4))
        assert decoded.metadata.nodata_value is None


def test_geotiff_compress_zstd(tmp_path: pathlib.Path) -> None:
    # Make sure we can use ZSTD compression successfully.
    path = UPath(tmp_path)
    projection = Projection(CRS.from_epsg(3857), 1, -1)
    array = np.zeros((1, 4, 4))
    raster_format = GeotiffRasterFormat(
        geotiff_options=dict(
            compress="zstd",
        )
    )
    raster_format.encode_raster(
        path, projection, (0, 0, 4, 4), RasterArray(chw_array=array)
    )
    with rasterio.open(path / "geotiff.tif") as raster:
        assert raster.profile["compress"] == "zstd"


def test_geotiff_multi_timestep_roundtrip(tmp_path: pathlib.Path) -> None:
    """Test encoding and decoding a multi-timestep CTHW raster."""
    path = UPath(tmp_path)
    projection = Projection(CRS.from_epsg(3857), 1, -1)

    c, t, h, w = 3, 5, 8, 8
    data = np.random.randint(0, 255, (c, t, h, w), dtype=np.uint8)
    timestamps = [
        (datetime(2024, 1, i + 1, tzinfo=UTC), datetime(2024, 1, i + 2, tzinfo=UTC))
        for i in range(t)
    ]
    raster = RasterArray(array=data, timestamps=timestamps)
    GeotiffRasterFormat().encode_raster(path, projection, (0, 0, w, h), raster)

    # Decode and verify shape + data.
    decoded = GeotiffRasterFormat().decode_raster(path, projection, (0, 0, w, h))
    assert decoded.array.shape == (c, t, h, w)
    assert np.array_equal(decoded.array, data)
    assert decoded.timestamps == timestamps


def test_geotiff_single_timestep_with_timestamps(tmp_path: pathlib.Path) -> None:
    """Test that T=1 with timestamps writes metadata.json and reads it back."""
    path = UPath(tmp_path)
    projection = Projection(CRS.from_epsg(3857), 1, -1)

    data = np.ones((2, 1, 4, 4), dtype=np.float32)
    ts = [(datetime(2024, 6, 1, tzinfo=UTC), datetime(2024, 6, 2, tzinfo=UTC))]
    raster = RasterArray(array=data, timestamps=ts)
    GeotiffRasterFormat().encode_raster(path, projection, (0, 0, 4, 4), raster)

    decoded = GeotiffRasterFormat().decode_raster(path, projection, (0, 0, 4, 4))
    assert decoded.array.shape == (2, 1, 4, 4)
    assert decoded.timestamps == ts


def test_raster_array_validation() -> None:
    """Test RasterArray validation."""
    # Must be 4D.
    with pytest.raises(ValueError, match="4D CTHW"):
        RasterArray(array=np.zeros((3, 4, 4)))

    # Timestamps length must match T.
    with pytest.raises(ValueError, match="timestamps length"):
        RasterArray(
            array=np.zeros((1, 2, 4, 4)),
            timestamps=[
                (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC))
            ],
        )

    # Valid with matching timestamps.
    ra = RasterArray(
        array=np.zeros((1, 2, 4, 4)),
        timestamps=[
            (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)),
            (datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 1, 3, tzinfo=UTC)),
        ],
    )
    assert ra.array.shape == (1, 2, 4, 4)


def test_raster_image_from_raster_array() -> None:
    """Test RasterImage.from_raster_array conversion."""
    from rslearn.train.model_context import RasterImage

    data = np.random.rand(3, 2, 8, 8).astype(np.float32)
    ts = [
        (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)),
        (datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 1, 3, tzinfo=UTC)),
    ]
    ra = RasterArray(array=data, timestamps=ts)
    ri = RasterImage.from_raster_array(ra)

    assert ri.image.shape == (3, 2, 8, 8)
    assert ri.timestamps == ts
    assert ri.image.dtype.is_floating_point


class TestNumpyRasterFormat:
    """Tests for NumpyRasterFormat."""

    PROJECTION = Projection(CRS.from_epsg(3857), 1, -1)

    def test_single_timestep_roundtrip(self, tmp_path: pathlib.Path) -> None:
        """Encode then decode a single-timestep (C, 1, H, W) array."""
        path = UPath(tmp_path)
        data = np.random.rand(3, 1, 4, 4).astype(np.float32)
        raster = RasterArray(array=data)

        fmt = NumpyRasterFormat()
        fmt.encode_raster(path, self.PROJECTION, (0, 0, 4, 4), raster)
        decoded = fmt.decode_raster(path, self.PROJECTION, (0, 0, 4, 4))

        assert decoded.array.shape == (3, 1, 4, 4)
        np.testing.assert_array_equal(decoded.array, data)
        assert decoded.timestamps is None

    def test_multi_timestep_roundtrip(self, tmp_path: pathlib.Path) -> None:
        """Encode then decode a multi-timestep (C, T, H, W) array with timestamps."""
        path = UPath(tmp_path)
        c, t, h, w = 2, 10, 1, 1
        data = np.arange(c * t * h * w, dtype=np.float32).reshape(c, t, h, w)
        timestamps = [
            (
                datetime(2024, 1, i + 1, tzinfo=UTC),
                datetime(2024, 1, i + 2, tzinfo=UTC),
            )
            for i in range(t)
        ]
        raster = RasterArray(array=data, timestamps=timestamps)

        fmt = NumpyRasterFormat()
        fmt.encode_raster(path, self.PROJECTION, (0, 0, w, h), raster)
        decoded = fmt.decode_raster(path, self.PROJECTION, (0, 0, w, h))

        assert decoded.array.shape == (c, t, h, w)
        np.testing.assert_array_equal(decoded.array, data)
        assert decoded.timestamps == timestamps

    def test_dtype_preserved(self, tmp_path: pathlib.Path) -> None:
        """Test that the stored dtype is preserved on decode."""
        path = UPath(tmp_path)
        data = np.array([[[[1, 2], [3, 4]]]], dtype=np.int16)  # (1, 1, 2, 2)
        raster = RasterArray(array=data)

        fmt = NumpyRasterFormat()
        fmt.encode_raster(path, self.PROJECTION, (0, 0, 2, 2), raster)
        decoded = fmt.decode_raster(path, self.PROJECTION, (0, 0, 2, 2))

        assert decoded.array.dtype == np.int16
        np.testing.assert_array_equal(decoded.array, data)

    def test_files_created(self, tmp_path: pathlib.Path) -> None:
        """Test that encode creates data.npy and metadata.json."""
        path = UPath(tmp_path)
        data = np.zeros((1, 1, 2, 2), dtype=np.float32)
        fmt = NumpyRasterFormat()
        fmt.encode_raster(path, self.PROJECTION, (0, 0, 2, 2), RasterArray(array=data))
        assert (path / "data.npy").exists()
        assert (path / "metadata.json").exists()

    def test_bounds_mismatch_raises(self, tmp_path: pathlib.Path) -> None:
        """Decoding with different bounds should raise ValueError."""
        path = UPath(tmp_path)
        data = np.ones((1, 1, 2, 2), dtype=np.float32)
        fmt = NumpyRasterFormat()
        fmt.encode_raster(path, self.PROJECTION, (0, 0, 2, 2), RasterArray(array=data))

        with pytest.raises(ValueError, match="bounds .* differ"):
            fmt.decode_raster(path, self.PROJECTION, (10, 10, 12, 12))

    def test_nodata_value_roundtrip(self, tmp_path: pathlib.Path) -> None:
        """nodata_value should round-trip through NumpyRasterFormat."""
        path = UPath(tmp_path)
        data = np.ones((2, 1, 3, 3), dtype=np.float32)
        nodata_value = -9999.0
        raster = RasterArray(
            array=data, metadata=RasterMetadata(nodata_value=nodata_value)
        )

        fmt = NumpyRasterFormat()
        fmt.encode_raster(path, self.PROJECTION, (0, 0, 3, 3), raster)
        decoded = fmt.decode_raster(path, self.PROJECTION, (0, 0, 3, 3))

        assert decoded.metadata.nodata_value == nodata_value
        np.testing.assert_array_equal(decoded.array, data)
