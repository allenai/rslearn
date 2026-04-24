from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
import shapely

from rslearn.config import BandSetConfig, DType, LayerConfig, LayerType
from rslearn.data_sources import DataSourceContext
from rslearn.data_sources.nasa_hls import Hls2L30, Hls2S30
from rslearn.utils.stac import StacAsset, StacItem

SEATTLE_WGS84_BOUNDS = (-122.34, 47.60, -122.32, 47.62)


def _make_stac_item(collection: str, asset_keys: list[str]) -> StacItem:
    assets: dict[str, StacAsset] = {}
    for asset_key in asset_keys:
        assets[asset_key] = StacAsset(
            href=f"https://example.com/{asset_key}.tif",
            title=asset_key,
            type="image/tiff",
            roles=["data"],
        )

    return StacItem(
        id=f"{collection}.example",
        properties={
            "datetime": "2020-07-20T00:00:00Z",
            "eo:cloud_cover": 12,
        },
        collection=collection,
        bbox=SEATTLE_WGS84_BOUNDS,
        geometry=shapely.geometry.mapping(shapely.box(*SEATTLE_WGS84_BOUNDS)),
        assets=assets,
        time_range=(
            datetime(2020, 7, 20, tzinfo=UTC),
            datetime(2020, 7, 21, tzinfo=UTC),
        ),
    )


def test_hls2_s30_defaults_to_reflectance_bands() -> None:
    data_source = Hls2S30()
    assert list(data_source.asset_bands.keys()) == Hls2S30.DEFAULT_BANDS


def test_hls2_s30_accepts_common_name_aliases() -> None:
    data_source = Hls2S30(band_names=["coastal", "red", "nir", "fmask"])
    assert set(data_source.asset_bands.keys()) == {"B01", "B04", "B08", "Fmask"}


def test_hls2_s30_accepts_context_band_aliases() -> None:
    layer_cfg = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[
            BandSetConfig(dtype=DType.UINT16, bands=["red", "nir08a"]),
            BandSetConfig(dtype=DType.UINT8, bands=["qa"]),
        ],
    )
    data_source = Hls2S30(context=DataSourceContext(layer_config=layer_cfg))
    assert list(data_source.asset_bands.keys()) == ["B04", "B8A", "Fmask"]


def test_hls2_l30_accepts_common_name_aliases() -> None:
    data_source = Hls2L30(band_names=["coastal", "red", "nir08", "fmask"])
    assert set(data_source.asset_bands.keys()) == {"B01", "B04", "B05", "Fmask"}


def test_hls2_rejects_unknown_band() -> None:
    with pytest.raises(ValueError, match="unsupported Hls2S30 band"):
        Hls2S30(band_names=["B01", "NOT_A_BAND"])


def test_hls2_prefers_s3_assets_in_stac_items() -> None:
    data_source = Hls2S30(band_names=["B04", "Fmask"])
    stac_item = _make_stac_item(
        data_source.COLLECTION_NAME,
        ["B04", "s3_B04", "Fmask", "s3_Fmask"],
    )

    item = data_source._stac_item_to_item(stac_item)

    assert item.asset_urls["B04"] == "https://example.com/s3_B04.tif"
    assert item.asset_urls["Fmask"] == "https://example.com/s3_Fmask.tif"
    assert item.properties["eo:cloud_cover"] == 12


def test_hls2_auth_uses_token_before_username_password() -> None:
    data_source = Hls2S30(
        earthdata_token="token",
        earthdata_username="user",
        earthdata_password="pass",
    )
    creds = data_source.auth._parse_credentials(
        {
            "accessKeyId": "key",
            "secretAccessKey": "secret",
            "sessionToken": "session",
            "expiration": "2030-01-01 00:00:00+00:00",
        }
    )

    with patch.object(data_source.auth, "_refresh_with_token", return_value=creds) as token:
        with patch.object(
            data_source.auth,
            "_refresh_with_username_password",
            side_effect=AssertionError("unexpected username/password fallback"),
        ):
            actual = data_source.auth.get_s3_credentials(refresh=True)

    token.assert_called_once()
    assert actual.access_key_id == "key"


def test_hls2_auth_falls_back_to_username_password() -> None:
    data_source = Hls2S30(earthdata_token="token", earthdata_username="user", earthdata_password="pass")
    creds = data_source.auth._parse_credentials(
        {
            "accessKeyId": "key",
            "secretAccessKey": "secret",
            "sessionToken": "session",
            "expiration": "2030-01-01 00:00:00+00:00",
        }
    )

    with patch.object(
        data_source.auth, "_refresh_with_token", side_effect=RuntimeError("bad token")
    ):
        with patch.object(
            data_source.auth,
            "_refresh_with_username_password",
            return_value=creds,
        ) as username_password:
            actual = data_source.auth.get_s3_credentials(refresh=True)

    username_password.assert_called_once()
    assert actual.access_key_id == "key"


def test_hls2_auth_reuses_cached_credentials() -> None:
    data_source = Hls2S30(earthdata_token="token")
    creds = data_source.auth._parse_credentials(
        {
            "accessKeyId": "key",
            "secretAccessKey": "secret",
            "sessionToken": "session",
            "expiration": (
                datetime.now(tz=UTC) + timedelta(hours=1)
            ).isoformat(sep=" "),
        }
    )
    data_source.auth._credentials = creds

    with patch.object(
        data_source.auth,
        "_refresh_with_token",
        side_effect=AssertionError("should use cached credentials"),
    ):
        actual = data_source.auth.get_s3_credentials()

    assert actual is creds
