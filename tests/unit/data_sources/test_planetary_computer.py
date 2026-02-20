from datetime import datetime
from unittest.mock import patch

import pytest

from rslearn.config import BandSetConfig, DType, LayerConfig, LayerType
from rslearn.data_sources import DataSourceContext
from rslearn.data_sources.planetary_computer import (
    CopDemGlo30,
    Hls2S30,
    Hls2L30,
    LandsatC2L2,
    PlanetaryComputer,
    Sentinel3SlstrLST,
)
from rslearn.utils.stac import StacAsset, StacItem


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


def test_landsat_c2_l2_defaults_to_reflectance_common_names() -> None:
    data_source = LandsatC2L2()
    assert set(data_source.asset_bands.keys()) == {
        "coastal",
        "blue",
        "green",
        "red",
        "nir08",
        "swir16",
        "swir22",
        "lwir11",
    }
    assert data_source.asset_bands["coastal"] == ["B1"]
    assert data_source.asset_bands["red"] == ["B4"]
    assert data_source.asset_bands["nir08"] == ["B5"]
    assert data_source.asset_bands["lwir11"] == ["B10"]
    assert data_source.query is not None
    assert data_source.query["platform"]["in"] == ["landsat-8", "landsat-9"]


def test_landsat_c2_l2_normalizes_stac_band_name_aliases_from_context() -> None:
    layer_cfg = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[
            BandSetConfig(dtype=DType.UINT16, bands=["OLI_B4", "OLI_B5", "TIRS_B10"])
        ],
    )
    context = DataSourceContext(layer_config=layer_cfg)

    data_source = LandsatC2L2(context=context)
    assert set(data_source.asset_bands.keys()) == {"red", "nir08", "lwir11"}
    assert data_source.asset_bands["red"] == ["B4"]
    assert data_source.asset_bands["nir08"] == ["B5"]
    assert data_source.asset_bands["lwir11"] == ["B10"]


def test_landsat_c2_l2_accepts_landsat_band_names_from_context() -> None:
    layer_cfg = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[BandSetConfig(dtype=DType.UINT16, bands=["B2", "B3", "B4"])],
    )
    context = DataSourceContext(layer_config=layer_cfg)

    data_source = LandsatC2L2(context=context)
    assert set(data_source.asset_bands.keys()) == {"blue", "green", "red"}
    assert data_source.asset_bands["blue"] == ["B2"]
    assert data_source.asset_bands["green"] == ["B3"]
    assert data_source.asset_bands["red"] == ["B4"]


def test_landsat_c2_l2_allows_overriding_platform_query() -> None:
    data_source = LandsatC2L2(query={"platform": {"in": ["landsat-8"]}})
    assert data_source.query is not None
    assert data_source.query["platform"]["in"] == ["landsat-8"]


def test_landsat_c2_l2_uses_user_query_unmodified() -> None:
    query = {"eo:cloud_cover": {"lt": 5}}
    data_source = LandsatC2L2(query=query)
    assert data_source.query == query


def test_hls2_s30_defaults_to_reflectance_bands() -> None:
    data_source = Hls2S30()
    assert set(data_source.asset_bands.keys()) == set(Hls2S30.DEFAULT_BANDS)


def test_hls2_s30_rejects_unknown_band() -> None:
    with pytest.raises(ValueError, match="unsupported HLS2 S30 band"):
        Hls2S30(band_names=["B01", "NOT_A_BAND"])


def test_hls2_s30_accepts_common_name_aliases() -> None:
    data_source = Hls2S30(band_names=["coastal", "red", "nir"])
    assert set(data_source.asset_bands.keys()) == {"B01", "B04", "B08"}


def test_hls2_l30_defaults_to_reflectance_bands() -> None:
    data_source = Hls2L30()
    assert set(data_source.asset_bands.keys()) == set(Hls2L30.DEFAULT_BANDS)


def test_hls2_l30_accepts_common_name_aliases() -> None:
    data_source = Hls2L30(band_names=["coastal", "red", "nir"])
    assert set(data_source.asset_bands.keys()) == {"B01", "B04", "B05"}


def test_planetary_computer_get_item_by_name_delegates_to_stac_data_source() -> None:
    """Ensure get_item_by_name works and doesn't raise NotImplementedError."""
    data_source = PlanetaryComputer(
        collection_name="sentinel-2-l2a",
        asset_bands={"visual": ["R", "G", "B"]},
    )

    mock_stac_item = StacItem(
        id="test-item-id",
        properties={"datetime": "2024-01-01T00:00:00Z"},
        collection="sentinel-2-l2a",
        bbox=(-122.0, 37.0, -121.0, 38.0),
        geometry={
            "type": "Polygon",
            "coordinates": [
                [[-122, 37], [-121, 37], [-121, 38], [-122, 38], [-122, 37]]
            ],
        },
        assets={
            "visual": StacAsset(
                href="https://example.com/visual.tif",
                title="Visual",
                type="image/tiff",
                roles=["data"],
            )
        },
        time_range=(datetime(2024, 1, 1), datetime(2024, 1, 1)),
    )

    with patch.object(data_source.client, "search", return_value=[mock_stac_item]):
        item = data_source.get_item_by_name("test-item-id")

    assert item.name == "test-item-id"
    assert "visual" in item.asset_urls


def test_sentinel3_slstr_lst_uses_fixed_band() -> None:
    layer_cfg = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[BandSetConfig(dtype=DType.FLOAT32, bands=["LST"])],
    )
    context = DataSourceContext(layer_config=layer_cfg)

    data_source = Sentinel3SlstrLST(context=context)
    assert data_source.asset_bands["lst-in"] == ["LST"]
    assert data_source.band_names == ["LST"]


def test_sentinel3_slstr_lst_rejects_non_lst_band_in_context() -> None:
    layer_cfg = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[BandSetConfig(dtype=DType.FLOAT32, bands=["LST_uncertainty"])],
    )
    context = DataSourceContext(layer_config=layer_cfg)

    with pytest.raises(ValueError, match="only supports the LST band"):
        Sentinel3SlstrLST(context=context)
