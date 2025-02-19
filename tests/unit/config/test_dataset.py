"""Test the dataset configuration file.

Mostly just makes sure there aren't runtime errors with the parsing.
"""

from rslearn.config import RasterLayerConfig, VectorLayerConfig


class TestBandSetConfig:
    """Test BandSetConfig."""

    def test_class_names_option(self) -> None:
        """Verify that config parsing works when class_names option is set."""
        class_names = ["class0", "class1", "class2"]
        layer_cfg_dict = {
            "type": "raster",
            "band_sets": [
                {
                    "dtype": "uint8",
                    "bands": ["class"],
                    "class_names": [class_names],
                }
            ],
        }
        layer_cfg = RasterLayerConfig.from_config(layer_cfg_dict)
        assert len(layer_cfg.band_sets) == 1
        band_set = layer_cfg.band_sets[0]
        assert len(band_set.bands) == 1
        assert band_set.class_names is not None
        assert band_set.class_names[0] == class_names


class TestVectorLayerConfig:
    """Test VectorLayerConfig."""

    def test_class_names_option(self) -> None:
        """Verify that config parsing works when property_name/class_names are set."""
        property_name = "my_class_prop"
        class_names = ["class0", "class1", "class2"]
        layer_cfg_dict = {
            "type": "vector",
            "class_property_name": property_name,
            "class_names": class_names,
        }
        layer_cfg = VectorLayerConfig.from_config(layer_cfg_dict)
        assert layer_cfg.class_property_name == property_name
        assert layer_cfg.class_names == class_names
