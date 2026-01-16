import pytest

from rslearn.config import BandSetConfig, DType, LayerConfig, LayerType
from rslearn.data_sources import DataSourceContext
from rslearn.data_sources.planetary_computer import CopDemGlo30


def test_cop_dem_glo_30_uses_context_layer_config_band_name() -> None:
    layer_cfg = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[BandSetConfig(dtype=DType.FLOAT32, bands=["elevation"])],
    )
    context = DataSourceContext(layer_config=layer_cfg)

    data_source = CopDemGlo30(band_name="DEM", context=context)
    assert data_source.asset_bands[CopDemGlo30.DATA_ASSET] == ["elevation"]


def test_cop_dem_glo_30_rejects_multiple_band_sets_in_context() -> None:
    layer_cfg = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[
            BandSetConfig(dtype=DType.FLOAT32, bands=["elevation"]),
            BandSetConfig(dtype=DType.FLOAT32, bands=["slope"]),
        ],
    )
    context = DataSourceContext(layer_config=layer_cfg)

    with pytest.raises(ValueError, match="expected a single band set"):
        CopDemGlo30(context=context)


def test_cop_dem_glo_30_rejects_multiple_bands_in_context_band_set() -> None:
    layer_cfg = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[
            BandSetConfig(dtype=DType.FLOAT32, bands=["elevation", "slope"]),
        ],
    )
    context = DataSourceContext(layer_config=layer_cfg)

    with pytest.raises(ValueError, match="expected band set to have a single band"):
        CopDemGlo30(context=context)

