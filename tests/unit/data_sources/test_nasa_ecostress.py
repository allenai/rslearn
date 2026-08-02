"""Unit tests for the NASA ECOSTRESS L2T LSTE data source."""

from datetime import UTC, datetime

import pytest
import shapely

from rslearn.data_sources.nasa_ecostress import EcostressLSTE
from rslearn.utils.stac import StacAsset, StacItem

SEATTLE_WGS84_BOUNDS = (-122.34, 47.60, -122.32, 47.62)
GRANULE = "ECOv002_L2T_LSTE_00376_004_13TDE_20180731T000421_0712_01"


def _asset(href: str) -> StacAsset:
    return StacAsset(href=href, title=None, type="image/tiff", roles=["data"])


def _make_stac_item(layers: list[str], include_s3: bool = True) -> StacItem:
    """Build a STAC item with ECOSTRESS-style path-like asset keys."""
    assets: dict[str, StacAsset] = {
        "browse": _asset("https://example.com/browse.png"),
    }
    for layer in layers:
        http_key = f"002/{GRANULE}/{GRANULE}_{layer}"
        assets[http_key] = _asset(f"https://example.com/{layer}.tif")
        if include_s3:
            assets[f"s3_{http_key}"] = _asset(f"s3://lp-prod-protected/{layer}.tif")

    return StacItem(
        id=GRANULE,
        properties={"datetime": "2018-07-31T00:04:21Z"},
        collection=EcostressLSTE.COLLECTION_NAME,
        bbox=SEATTLE_WGS84_BOUNDS,
        geometry=shapely.geometry.mapping(shapely.box(*SEATTLE_WGS84_BOUNDS)),
        assets=assets,
        time_range=(
            datetime(2018, 7, 31, tzinfo=UTC),
            datetime(2018, 7, 31, 0, 5, tzinfo=UTC),
        ),
    )


def test_defaults_to_lst_band() -> None:
    data_source = EcostressLSTE()
    assert list(data_source.asset_bands.keys()) == ["LST"]


def test_accepts_supported_bands() -> None:
    data_source = EcostressLSTE(band_names=["LST", "cloud"])
    assert set(data_source.asset_bands.keys()) == {"LST", "cloud"}


def test_rejects_unsupported_band() -> None:
    with pytest.raises(ValueError, match="unsupported EcostressLSTE band"):
        EcostressLSTE(band_names=["NOT_A_BAND"])


def test_stac_item_maps_lst_by_suffix_and_prefers_s3() -> None:
    data_source = EcostressLSTE(band_names=["LST"])
    item = data_source._stac_item_to_item(_make_stac_item(["LST", "LST_err", "QC"]))

    assert item.asset_urls["LST"] == "s3://lp-prod-protected/LST.tif"
    assert item.properties["_http_url_LST"] == "https://example.com/LST.tif"
    # The _LST suffix must not accidentally match the _LST_err asset.
    assert "LST_err" not in item.asset_urls


def test_stac_item_falls_back_to_http_when_no_s3() -> None:
    data_source = EcostressLSTE(band_names=["LST"])
    item = data_source._stac_item_to_item(_make_stac_item(["LST"], include_s3=False))
    assert item.asset_urls["LST"] == "https://example.com/LST.tif"


def test_should_include_item_requires_all_bands() -> None:
    data_source = EcostressLSTE(band_names=["LST", "cloud"])
    with_both = data_source._stac_item_to_item(_make_stac_item(["LST", "cloud"]))
    missing_cloud = data_source._stac_item_to_item(_make_stac_item(["LST"]))

    assert data_source._should_include_item(with_both)
    assert not data_source._should_include_item(missing_cloud)
