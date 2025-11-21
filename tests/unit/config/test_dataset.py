"""Tests for the rslearn.config.dataset module."""

from rslearn.config.dataset import DType, LayerConfig
from rslearn.data_sources.planetary_computer import Sentinel2
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
                },
            }
        )

        band_set = layer_config.band_sets[0]
        assert band_set.dtype == DType.UINT8
        raster_format = band_set.instantiate_raster_format()
        assert isinstance(raster_format, SingleImageRasterFormat)
        assert raster_format.format == "png"

        data_source = layer_config.instantiate_data_source()
        assert isinstance(data_source, Sentinel2)
        assert data_source.harmonize
