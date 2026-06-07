"""Tests for the rslearn.config.dataset module."""

import warnings
from datetime import timedelta

import pytest
from pydantic import ValidationError
from rasterio.crs import CRS

from rslearn.config.dataset import (
    BandSetConfig,
    DType,
    LayerConfig,
    QueryConfig,
    TimeMode,
)
from rslearn.data_sources.planetary_computer import Sentinel2
from rslearn.utils.geometry import Projection
from rslearn.utils.raster_format import NumpyRasterFormat, SingleImageRasterFormat
from rslearn.utils.vector_format import TileVectorFormat


class TestLayerConfig:
    """Tests for LayerConfig."""

    def test_custom_vector_format(self) -> None:
        """Test layer configuration that specifies a custom vector format."""
        layer_config = LayerConfig.model_validate(
            {
                "type": "vector",
                "vector_format": {
                    "class_path": "rslearn.utils.vector_format.TileVectorFormat",
                    "init_args": {
                        "tile_size": 256,
                    },
                },
            }
        )
        vector_format = layer_config.instantiate_vector_format()
        assert isinstance(vector_format, TileVectorFormat)
        assert vector_format.tile_size == 256

    def test_data_source(self) -> None:
        """Test layer configuration that specifies a data source."""
        layer_config = LayerConfig.model_validate(
            {
                "type": "raster",
                "band_sets": [
                    {
                        "dtype": "uint8",
                        "bands": ["R", "G", "B"],
                        "format": {
                            "class_path": "rslearn.utils.raster_format.SingleImageRasterFormat",
                            "init_args": {
                                "format": "png",
                            },
                        },
                    },
                ],
                "data_source": {
                    "class_path": "rslearn.data_sources.planetary_computer.Sentinel2",
                    "init_args": {
                        "harmonize": True,
                    },
                    "ingest": False,
                    "query_config": {
                        "min_matches": 4,
                        "max_matches": 4,
                    },
                },
            },
        )

        band_set = layer_config.band_sets[0]
        assert band_set.dtype == DType.UINT8
        raster_format = band_set.instantiate_raster_format()
        assert isinstance(raster_format, SingleImageRasterFormat)
        assert raster_format.format == "png"

        assert layer_config.data_source is not None
        assert not layer_config.data_source.ingest
        assert layer_config.data_source.query_config.min_matches == 4

        data_source = layer_config.instantiate_data_source()
        assert isinstance(data_source, Sentinel2)
        assert data_source.harmonize

    def test_timedeltas(self) -> None:
        """Test timedelta parsing."""
        layer_config = LayerConfig.model_validate(
            {
                "type": "raster",
                "band_sets": [
                    {
                        "dtype": "uint8",
                        "bands": ["R", "G", "B"],
                    },
                ],
                "data_source": {
                    "duration": "180d",
                    "time_offset": "-5d",
                    "class_path": "rslearn.data_sources.planetary_computer.Sentinel2",
                },
            }
        )
        assert layer_config.data_source is not None
        assert layer_config.data_source.duration == timedelta(days=180)
        assert layer_config.data_source.time_offset == timedelta(days=-5)

    def test_missing_bandsets(self) -> None:
        """An error should be raised if band sets are missing for a raster layer."""
        with pytest.raises(ValidationError):
            LayerConfig.model_validate({"type": "raster"})


class TestQueryConfigTimeMode:
    """Tests for the deprecated time_mode field in QueryConfig."""

    def test_no_warning_when_time_mode_not_set(self) -> None:
        """No warning should be emitted when time_mode is not in the config."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            QueryConfig()
            assert len(w) == 0

    def test_warning_when_time_mode_set(self) -> None:
        """A warning should be emitted when time_mode is set, and it should be excluded from dump."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            query_config = QueryConfig(time_mode=TimeMode.WITHIN)
            assert len(w) == 1
            assert issubclass(w[0].category, FutureWarning)
            assert "time_mode" in str(w[0].message)

        # time_mode should be excluded from the dump
        dumped = query_config.model_dump()
        assert "space_mode" in dumped
        assert "time_mode" not in dumped


class TestBandSetConfigSpatialSize:
    """Tests for the spatial_size option on BandSetConfig."""

    def test_spatial_size_default_none(self) -> None:
        """Default spatial_size should be None."""
        bs = BandSetConfig(dtype=DType.FLOAT32, bands=["a"])
        assert bs.spatial_size is None

    def test_spatial_size_projection_and_bounds(self) -> None:
        """spatial_size should adjust projection and bounds to target dimensions."""
        bs = BandSetConfig(dtype=DType.FLOAT32, bands=["a"], spatial_size=(1, 1))
        projection = Projection(CRS.from_epsg(3857), 10.0, -10.0)
        bounds = (100, 200, 228, 328)  # 128 x 128 pixels

        new_proj, new_bounds = bs.get_final_projection_and_bounds(projection, bounds)

        # Output should be 1x1 pixels.
        assert new_bounds[2] - new_bounds[0] == 1
        assert new_bounds[3] - new_bounds[1] == 1

        # The resolution should scale up by the original pixel count.
        assert new_proj.x_resolution == pytest.approx(10.0 / (1 / 128))
        # x_resolution: 10 / (1/128) = 1280 -- each output pixel covers 128 original pixels
        assert new_proj.x_resolution == pytest.approx(10.0 * 128)

    def test_spatial_size_non_square(self) -> None:
        """spatial_size with non-square dimensions should work correctly."""
        bs = BandSetConfig(dtype=DType.FLOAT32, bands=["a"], spatial_size=(2, 4))
        projection = Projection(CRS.from_epsg(3857), 1.0, -1.0)
        bounds = (0, 0, 100, 200)  # 100 x 200 pixels

        new_proj, new_bounds = bs.get_final_projection_and_bounds(projection, bounds)

        assert new_bounds[2] - new_bounds[0] == 4  # width
        assert new_bounds[3] - new_bounds[1] == 2  # height

    def test_spatial_size_mutually_exclusive_with_zoom_offset(self) -> None:
        """spatial_size and non-zero zoom_offset should raise an error."""
        with pytest.raises(ValidationError, match="mutually exclusive"):
            BandSetConfig(
                dtype=DType.FLOAT32, bands=["a"], spatial_size=(1, 1), zoom_offset=1
            )

    def test_spatial_size_zero_rejected(self) -> None:
        """spatial_size with zero value should raise an error."""
        with pytest.raises(ValidationError, match="positive integers"):
            BandSetConfig(dtype=DType.FLOAT32, bands=["a"], spatial_size=(0, 1))

    def test_spatial_size_negative_rejected(self) -> None:
        """spatial_size with negative value should raise an error."""
        with pytest.raises(ValidationError, match="positive integers"):
            BandSetConfig(dtype=DType.FLOAT32, bands=["a"], spatial_size=(-1, 1))

    def test_numpy_raster_format_from_config(self) -> None:
        """NumpyRasterFormat should be instantiable from config via jsonargparse."""
        bs = BandSetConfig(
            dtype=DType.FLOAT32,
            bands=["a"],
            format={
                "class_path": "rslearn.utils.raster_format.NumpyRasterFormat",
            },
        )
        fmt = bs.instantiate_raster_format()
        assert isinstance(fmt, NumpyRasterFormat)
