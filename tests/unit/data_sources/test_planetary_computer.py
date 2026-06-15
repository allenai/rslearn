from datetime import UTC, datetime
from typing import cast
from unittest.mock import patch

import pytest
import shapely

from rslearn.config import (
    BandSetConfig,
    DType,
    LayerConfig,
    LayerType,
    QueryConfig,
    SpaceMode,
)
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import DataSourceContext
from rslearn.data_sources.planetary_computer import (
    CopDemGlo30,
    Hls2L30,
    Hls2S30,
    LandsatC2L2,
    PlanetaryComputer,
    Sentinel2,
    Sentinel3SlstrLST,
)
from rslearn.data_sources.stac import SourceItem
from rslearn.utils.geometry import STGeometry
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


def test_sentinel2_context_rgb_without_cloud_ranking_uses_visual_only() -> None:
    layer_cfg = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[BandSetConfig(dtype=DType.UINT8, bands=["R", "G", "B"])],
    )
    context = DataSourceContext(layer_config=layer_cfg)

    data_source = Sentinel2(context=context)
    assert set(data_source.asset_bands.keys()) == {"visual"}


def test_sentinel2_context_rgb_with_scl_ranking_keeps_visual_only() -> None:
    layer_cfg = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[BandSetConfig(dtype=DType.UINT8, bands=["R", "G", "B"])],
        compositing_method={
            "class_path": "rslearn.dataset.sentinel2_scl.Sentinel2SCLFirstValid",
            "init_args": {"scl_band": "SCL"},
        },
    )
    context = DataSourceContext(layer_config=layer_cfg)

    data_source = Sentinel2(context=context)
    assert set(data_source.asset_bands.keys()) == {"visual"}


def test_sentinel2_context_rgb_with_omni_ranking_keeps_visual_only() -> None:
    layer_cfg = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[BandSetConfig(dtype=DType.UINT8, bands=["R", "G", "B"])],
        compositing_method={
            "class_path": "rslearn.dataset.omni_cloud_mask.OmniCloudMaskFirstValid",
            "init_args": {"red_band": "B04", "green_band": "B03", "nir_band": "B8A"},
        },
    )
    context = DataSourceContext(layer_config=layer_cfg)

    data_source = Sentinel2(context=context)
    assert set(data_source.asset_bands.keys()) == {"visual"}


def test_sentinel2_context_explicit_scoring_bands_are_included() -> None:
    layer_cfg = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[
            BandSetConfig(dtype=DType.UINT8, bands=["R", "G", "B"]),
            BandSetConfig(dtype=DType.UINT8, bands=["SCL"]),
            BandSetConfig(dtype=DType.UINT16, bands=["B04", "B03", "B8A"]),
        ],
    )
    context = DataSourceContext(layer_config=layer_cfg)

    data_source = Sentinel2(context=context)
    assert set(data_source.asset_bands.keys()) == {"visual", "SCL", "B04", "B03", "B8A"}


def test_sentinel2_get_read_callback_skips_scl_harmonization() -> None:
    data_source = Sentinel2(harmonize=True)
    item = cast(SourceItem, object())

    with patch.object(
        Sentinel2,
        "_get_product_xml",
        side_effect=AssertionError("SCL should not request product XML"),
    ):
        callback = data_source.get_read_callback(item=item, asset_key="SCL")

    assert callback is None


def _make_geoparquet_row(
    *,
    item_id: str,
    shp: shapely.Geometry,
    dt: datetime,
    cloud_cover: float = 0,
) -> dict:
    """Create a mock Planetary Computer STAC GeoParquet row."""
    west, south, east, north = shp.bounds
    return {
        "id": item_id,
        "bbox": {"xmin": west, "ymin": south, "xmax": east, "ymax": north},
        "datetime": dt,
        "collection": Sentinel2.COLLECTION_NAME,
        "geometry": shapely.to_wkb(shp),
        "assets": {
            "B04": {
                "href": f"https://example.com/{item_id}/B04.tif",
                "title": "Band 4",
                "type": "image/tiff",
                "roles": ["data"],
            },
            "product-metadata": {
                "href": f"https://example.com/{item_id}/metadata.xml",
                "title": "Product metadata",
                "type": "application/xml",
                "roles": ["metadata"],
            },
        },
        "eo:cloud_cover": cloud_cover,
    }


def test_sentinel2_geoparquet_batches_prepare_window_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_source = Sentinel2(
        assets=["B04"],
        metadata_backend="geoparquet",
        geoparquet_href="file:///unused.parquet",
        sort_by="eo:cloud_cover",
    )

    geom1 = STGeometry(
        WGS84_PROJECTION,
        shapely.box(-1, 50, 0, 51),
        (
            datetime(2020, 1, 1, tzinfo=UTC),
            datetime(2021, 1, 1, tzinfo=UTC),
        ),
    )
    geom2 = STGeometry(
        WGS84_PROJECTION,
        shapely.box(-5, 55, -4, 56),
        (
            datetime(2021, 1, 1, tzinfo=UTC),
            datetime(2022, 1, 1, tzinfo=UTC),
        ),
    )

    captured = {}

    def fake_read_geoparquet_rows(**kwargs: object) -> list[dict]:
        captured.update(kwargs)
        return [
            _make_geoparquet_row(
                item_id="S2A_MSIL2A_20200601T000000_R000_T30UXB_20200601T000000",
                shp=shapely.box(-1, 50, 0, 51),
                dt=datetime(2020, 6, 1, tzinfo=UTC),
                cloud_cover=80,
            ),
            _make_geoparquet_row(
                item_id="S2A_MSIL2A_20200602T000000_R000_T30UXB_20200602T000000",
                shp=shapely.box(-1, 50, 0, 51),
                dt=datetime(2020, 6, 2, tzinfo=UTC),
                cloud_cover=5,
            ),
            _make_geoparquet_row(
                item_id="S2A_MSIL2A_20210601T000000_R000_T30UWA_20210601T000000",
                shp=shapely.box(-5, 55, -4, 56),
                dt=datetime(2021, 6, 1, tzinfo=UTC),
                cloud_cover=10,
            ),
        ]

    monkeypatch.setattr(
        data_source.client, "_read_geoparquet_rows", fake_read_geoparquet_rows
    )

    groups = data_source.get_items(
        [geom1, geom2], QueryConfig(space_mode=SpaceMode.INTERSECTS)
    )

    assert captured["bbox"] == (-5.0, 50.0, 0.0, 56.0)
    assert captured["date_time"] == (
        datetime(2020, 1, 1, tzinfo=UTC),
        datetime(2022, 1, 1, tzinfo=UTC),
    )
    assert len(groups) == 2
    assert groups[0][0].items[0].name == (
        "S2A_MSIL2A_20200602T000000_R000_T30UXB_20200602T000000"
    )
    assert groups[1][0].items[0].name == (
        "S2A_MSIL2A_20210601T000000_R000_T30UWA_20210601T000000"
    )


def test_sentinel2_geoparquet_get_item_by_name_infers_time_from_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_source = Sentinel2(
        assets=["B04"],
        metadata_backend="geoparquet",
        geoparquet_href="file:///unused.parquet",
    )
    item_id = "S2A_MSIL2A_20200718T130301_R038_T27VVL_20210515T015108"
    captured = {}

    def fake_read_geoparquet_rows(**kwargs: object) -> list[dict]:
        captured.update(kwargs)
        return [
            _make_geoparquet_row(
                item_id=item_id,
                shp=shapely.box(-23, 63, -21, 64),
                dt=datetime(2020, 7, 18, 13, 3, 1, tzinfo=UTC),
            )
        ]

    monkeypatch.setattr(
        data_source.client, "_read_geoparquet_rows", fake_read_geoparquet_rows
    )

    item = data_source.get_item_by_name(item_id)

    assert captured["ids"] == [item_id]
    assert captured["date_time"] == (
        datetime(2020, 7, 18, tzinfo=UTC),
        datetime(2020, 7, 19, tzinfo=UTC),
    )
    assert item.name == item_id
    assert item.asset_urls["B04"] == f"https://example.com/{item_id}/B04.tif"


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
