"""Tests for the rslearn.config.dataset module."""

from datetime import timedelta

import pytest
from pydantic import ValidationError

from rslearn.config.dataset import DType, LayerConfig
from rslearn.data_sources.planetary_computer import Sentinel1, Sentinel2
from rslearn.utils.raster_format import SingleImageRasterFormat
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


class TestBackwardsCompatibility:
    """Test backwards compatibility with old dataset config version.

    The previous version used custom parsing instead of a combination of pydantic
    models (for the base config) + jsonargparse (for parsing components with multiple
    selectable implementations).

    The backwards compatibility should be maintained through 1 March 2026.
    """

    def test_data_source_compat(self) -> None:
        """Test previous data source format.

        In the old format, the source-specific init_args are mixed with the base
        data source config options.
        """
        layer_config = LayerConfig.model_validate(
            {
                "band_sets": [
                    {"bands": ["B02", "B03", "B04"], "dtype": "uint16"},
                    {
                        "bands": ["B01", "B09"],
                        "dtype": "uint16",
                        "zoom_offset": -2,
                    },
                ],
                "data_source": {
                    # Base options.
                    "ingest": False,
                    "query_config": {
                        "max_matches": 12,
                        "period_duration": "30d",
                        "space_mode": "PER_PERIOD_MOSAIC",
                    },
                    # Class path is specified with "name" instead of "class_path".
                    "name": "rslearn.data_sources.planetary_computer.Sentinel2",
                    # Source-specific options are here instead of init_args dict.
                    "cache_dir": "cache/planetary_computer",
                    "harmonize": True,
                    "sort_by": "eo:cloud_cover",
                },
                "type": "raster",
            }
        )

        # Check band sets.
        assert len(layer_config.band_sets) == 2
        assert layer_config.band_sets[0].bands == ["B02", "B03", "B04"]
        assert layer_config.band_sets[0].zoom_offset == 0
        assert layer_config.band_sets[1].bands == ["B01", "B09"]
        assert layer_config.band_sets[1].zoom_offset == -2

        # Check data source.
        assert layer_config.data_source is not None
        assert not layer_config.data_source.ingest
        assert layer_config.data_source.query_config.max_matches == 12
        assert (
            layer_config.data_source.class_path
            == "rslearn.data_sources.planetary_computer.Sentinel2"
        )
        ds = layer_config.instantiate_data_source()
        assert isinstance(ds, Sentinel2)
        assert ds.harmonize
        assert ds.sort_by == "eo:cloud_cover"

    def test_invalid_cloud_cover_options(self) -> None:
        """Make sure invalid cloud cover options is ignored.

        We provide backwards compatibility specifically with "max_cloud_cover" being
        erroneously provided to the Planetary Computer Sentinel-2 data source, because
        there are some dataset configs that do this that we want to maintain backwards
        compatibility with.
        """
        layer_config = LayerConfig.model_validate(
            {
                "band_sets": [
                    {"bands": ["B02", "B03", "B04"], "dtype": "uint16"},
                ],
                "data_source": {
                    "name": "rslearn.data_sources.planetary_computer.Sentinel2",
                    # Valid init arg.
                    "sort_by": "eo:cloud_cover",
                    # Invalid init arg.
                    "max_cloud_cover": 50,
                },
                "type": "raster",
            }
        )
        assert layer_config.data_source is not None
        assert layer_config.data_source.init_args["sort_by"] == "eo:cloud_cover"
        assert "max_cloud_cover" not in layer_config.data_source.init_args

    def test_sentinel1_layer(self) -> None:
        """Make sure a Sentinel-1 layer is parsed correctly too.

        This example is taken from LFMC config in olmoearth_projects.
        """
        layer_config = LayerConfig.model_validate(
            {
                "band_sets": [
                    {
                        "bands": ["vv", "vh"],
                        "dtype": "float32",
                    }
                ],
                "data_source": {
                    # Base options.
                    "ingest": False,
                    "duration": "168d",
                    "time_offset": "-168d",
                    "query_config": {
                        "max_matches": 12,
                        "period_duration": "14d",
                        "space_mode": "PER_PERIOD_MOSAIC",
                    },
                    # Class path and source-specific options.
                    "name": "rslearn.data_sources.planetary_computer.Sentinel1",
                    "cache_dir": "cache/planetary_computer",
                    "query": {
                        "sar:instrument_mode": {"eq": "IW"},
                        "sar:polarizations": {"eq": ["VV", "VH"]},
                    },
                },
                "type": "raster",
            }
        )

        assert len(layer_config.band_sets) == 1
        assert layer_config.band_sets[0].dtype == DType.FLOAT32
        assert layer_config.data_source is not None
        assert not layer_config.data_source.ingest
        assert layer_config.data_source.duration == timedelta(days=168)

        ds = layer_config.instantiate_data_source()
        assert isinstance(ds, Sentinel1)
        assert ds.query is not None
        assert ds.query["sar:instrument_mode"] == {"eq": "IW"}
        assert ds.cache_dir is not None
        assert ds.cache_dir.path == "cache/planetary_computer"
