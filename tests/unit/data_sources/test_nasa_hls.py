from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import numpy as np
import pytest
import shapely
from rasterio.errors import RasterioIOError

from rslearn.config import BandSetConfig, DType, LayerConfig, LayerType
from rslearn.data_sources import DataSourceContext
from rslearn.data_sources.direct_materialize_data_source import (
    DirectMaterializeDataSource,
)
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
    assert item.properties["_http_url_B04"] == "https://example.com/B04.tif"
    assert item.properties["_http_url_Fmask"] == "https://example.com/Fmask.tif"
    assert item.properties["eo:cloud_cover"] == 12


def test_hls2_download_falls_back_to_http_asset() -> None:
    data_source = Hls2S30(band_names=["B04"])
    stac_item = _make_stac_item(data_source.COLLECTION_NAME, ["B04", "s3_B04"])
    assert stac_item.assets is not None
    stac_item = StacItem(
        id=stac_item.id,
        properties=stac_item.properties,
        collection=stac_item.collection,
        bbox=stac_item.bbox,
        geometry=stac_item.geometry,
        assets={
            **stac_item.assets,
            "s3_B04": StacAsset(
                href="s3://lp-prod-protected/example/B04.tif",
                title="s3_B04",
                type="image/tiff",
                roles=["data"],
            ),
        },
        time_range=stac_item.time_range,
    )
    item = data_source._stac_item_to_item(stac_item)

    with patch.object(
        data_source,
        "_download_asset",
        side_effect=[RuntimeError("denied"), None],
    ) as download_asset:
        data_source._download_asset_with_fallback(item, "B04", "/tmp/out.tif")

    assert (
        download_asset.call_args_list[0].args[0]
        == "s3://lp-prod-protected/example/B04.tif"
    )
    assert download_asset.call_args_list[1].args[0] == "https://example.com/B04.tif"


def test_hls2_read_raster_falls_back_to_http_asset() -> None:
    data_source = Hls2S30(band_names=["B04"])
    stac_item = _make_stac_item(data_source.COLLECTION_NAME, ["B04", "s3_B04"])
    assert stac_item.assets is not None
    stac_item = StacItem(
        id=stac_item.id,
        properties=stac_item.properties,
        collection=stac_item.collection,
        bbox=stac_item.bbox,
        geometry=stac_item.geometry,
        assets={
            **stac_item.assets,
            "s3_B04": StacAsset(
                href="s3://lp-prod-protected/example/B04.tif",
                title="s3_B04",
                type="image/tiff",
                roles=["data"],
            ),
        },
        time_range=stac_item.time_range,
    )
    item = data_source._stac_item_to_item(stac_item)
    expected_array = np.ones((1, 2, 2), dtype=np.uint16)
    projection = object()
    bounds = object()
    resampling = object()

    with patch.object(data_source, "_rasterio_env", return_value=nullcontext()):
        with patch.object(
            DirectMaterializeDataSource,
            "_read_raster_from_url",
            side_effect=RasterioIOError("denied"),
        ):
            with patch.object(
                data_source,
                "_read_raster_from_local_copy",
                return_value=(expected_array, 0),
            ) as local_copy:
                actual_array, actual_nodata = data_source._read_raster_for_item(
                    item,
                    "B04",
                    projection,
                    bounds,
                    resampling,
                )

    local_copy.assert_called_once_with(
        "https://example.com/B04.tif", projection, bounds, resampling
    )
    assert actual_nodata == 0
    np.testing.assert_array_equal(actual_array, expected_array)


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

    with patch.object(
        data_source.auth, "_refresh_with_token", return_value=creds
    ) as token:
        with patch.object(
            data_source.auth,
            "_refresh_with_username_password",
            side_effect=AssertionError("unexpected username/password fallback"),
        ):
            actual = data_source.auth.get_s3_credentials(refresh=True)

    token.assert_called_once()
    assert actual.access_key_id == "key"


def test_hls2_auth_falls_back_to_username_password() -> None:
    data_source = Hls2S30(
        earthdata_token="token", earthdata_username="user", earthdata_password="pass"
    )
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
            "expiration": (datetime.now(tz=UTC) + timedelta(hours=1)).isoformat(
                sep=" "
            ),
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
