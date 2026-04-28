from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import numpy as np
import pytest
import shapely
from rasterio.errors import RasterioIOError

from rslearn.config import QueryConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.direct_materialize_data_source import (
    DirectMaterializeDataSource,
)
from rslearn.data_sources.nasa_hls import Hls2, Hls2L30, Hls2S30
from rslearn.utils.geometry import STGeometry
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


def test_hls2_s30_accepts_asset_keys() -> None:
    data_source = Hls2S30(band_names=["B01", "B04", "B08", "Fmask"])
    assert set(data_source.asset_bands.keys()) == {"B01", "B04", "B08", "Fmask"}


def test_hls2_s30_rejects_common_names() -> None:
    with pytest.raises(ValueError, match="unsupported Hls2S30 band"):
        Hls2S30(band_names=["red"])


def test_hls2_l30_accepts_asset_keys() -> None:
    data_source = Hls2L30(band_names=["B01", "B04", "B05", "Fmask"])
    assert set(data_source.asset_bands.keys()) == {"B01", "B04", "B05", "Fmask"}


def test_hls2_l30_rejects_common_names() -> None:
    with pytest.raises(ValueError, match="unsupported Hls2L30 band"):
        Hls2L30(band_names=["nir"])


def test_hls2_rejects_unknown_band() -> None:
    with pytest.raises(ValueError, match="unsupported Hls2S30 band"):
        Hls2S30(band_names=["B01", "NOT_A_BAND"])


def test_combined_hls2_rejects_raw_asset_keys() -> None:
    with pytest.raises(ValueError, match="unsupported Hls2 band"):
        Hls2(band_names=["B04"])


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


def test_hls2_auth_uses_token() -> None:
    data_source = Hls2S30(earthdata_token="token")
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
        actual = data_source.auth.get_s3_credentials(refresh=True)

    token.assert_called_once()
    assert actual.access_key_id == "key"


def test_hls2_auth_requires_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
    data_source = Hls2S30(earthdata_token=None)

    with pytest.raises(ValueError, match="EARTHDATA_TOKEN"):
        data_source.auth.get_s3_credentials(refresh=True)


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


def test_combined_hls2_defaults_to_common_semantic_bands() -> None:
    data_source = Hls2()
    assert list(data_source.asset_bands.keys()) == Hls2.DEFAULT_BANDS
    assert data_source.sources == ["sentinel", "landsat"]


def test_combined_hls2_accepts_source_filter() -> None:
    data_source = Hls2(sources=["landsat"], band_names=["red", "nir", "fmask"])
    assert data_source.collection_names == [Hls2L30.COLLECTION_NAME]
    assert list(data_source.asset_bands.keys()) == ["red", "nir", "fmask"]


def test_combined_hls2_rejects_unknown_source() -> None:
    with pytest.raises(ValueError, match="unsupported Hls2 source"):
        Hls2(sources=["sentinel", "planet"])


def test_combined_hls2_maps_semantic_bands_per_collection() -> None:
    data_source = Hls2(band_names=["red", "nir", "swir16", "fmask"])

    sentinel_item = StacItem(
        id="HLSS30.example",
        properties={"datetime": "2020-07-20T00:00:00Z"},
        collection=Hls2S30.COLLECTION_NAME,
        bbox=SEATTLE_WGS84_BOUNDS,
        geometry=shapely.geometry.mapping(shapely.box(*SEATTLE_WGS84_BOUNDS)),
        assets={
            "B04": StacAsset(
                href="https://example.com/s_red.tif", title=None, type=None, roles=None
            ),
            "s3_B04": StacAsset(
                href="s3://bucket/s_red.tif", title=None, type=None, roles=None
            ),
            "B08": StacAsset(
                href="https://example.com/s_nir.tif", title=None, type=None, roles=None
            ),
            "B11": StacAsset(
                href="https://example.com/s_swir16.tif",
                title=None,
                type=None,
                roles=None,
            ),
            "Fmask": StacAsset(
                href="https://example.com/s_fmask.tif",
                title=None,
                type=None,
                roles=None,
            ),
        },
        time_range=(
            datetime(2020, 7, 20, tzinfo=UTC),
            datetime(2020, 7, 21, tzinfo=UTC),
        ),
    )
    landsat_item = StacItem(
        id="HLSL30.example",
        properties={"datetime": "2020-07-21T00:00:00Z"},
        collection=Hls2L30.COLLECTION_NAME,
        bbox=SEATTLE_WGS84_BOUNDS,
        geometry=shapely.geometry.mapping(shapely.box(*SEATTLE_WGS84_BOUNDS)),
        assets={
            "B04": StacAsset(
                href="https://example.com/l_red.tif", title=None, type=None, roles=None
            ),
            "B05": StacAsset(
                href="https://example.com/l_nir.tif", title=None, type=None, roles=None
            ),
            "B06": StacAsset(
                href="https://example.com/l_swir16.tif",
                title=None,
                type=None,
                roles=None,
            ),
            "Fmask": StacAsset(
                href="https://example.com/l_fmask.tif",
                title=None,
                type=None,
                roles=None,
            ),
        },
        time_range=(
            datetime(2020, 7, 21, tzinfo=UTC),
            datetime(2020, 7, 22, tzinfo=UTC),
        ),
    )

    sentinel_source_item = data_source._stac_item_to_item(sentinel_item)
    landsat_source_item = data_source._stac_item_to_item(landsat_item)

    assert sentinel_source_item.asset_urls["red"] == "s3://bucket/s_red.tif"
    assert sentinel_source_item.asset_urls["nir"] == "https://example.com/s_nir.tif"
    assert sentinel_source_item.properties["sensor"] == "sentinel"
    assert landsat_source_item.asset_urls["nir"] == "https://example.com/l_nir.tif"
    assert (
        landsat_source_item.asset_urls["swir16"] == "https://example.com/l_swir16.tif"
    )
    assert landsat_source_item.properties["sensor"] == "landsat"


def test_combined_hls2_get_item_by_name_searches_selected_collections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_source = Hls2(
        sources=["landsat"],
        band_names=["red"],
    )
    stac_item = _make_stac_item(Hls2L30.COLLECTION_NAME, ["B04"])
    captured_kwargs: dict[str, object] = {}

    def fake_search(**kwargs: object) -> list[StacItem]:
        captured_kwargs.update(kwargs)
        return [stac_item]

    monkeypatch.setattr(data_source.client, "search", fake_search)

    item = data_source.get_item_by_name(stac_item.id)

    assert item.name == stac_item.id
    assert item.asset_urls["red"] == "https://example.com/B04.tif"
    assert captured_kwargs["collections"] == [Hls2L30.COLLECTION_NAME]


def test_combined_hls2_get_items_merges_and_sorts_chronologically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_source = Hls2(band_names=["red"], sort_by="datetime", sort_ascending=True)
    geometry = STGeometry(
        WGS84_PROJECTION,
        shapely.box(*SEATTLE_WGS84_BOUNDS),
        (datetime(2020, 7, 1, tzinfo=UTC), datetime(2020, 8, 1, tzinfo=UTC)),
    )
    later = StacItem(
        id="later",
        properties={"datetime": "2020-07-21T00:00:00Z"},
        collection=Hls2L30.COLLECTION_NAME,
        bbox=SEATTLE_WGS84_BOUNDS,
        geometry=shapely.geometry.mapping(shapely.box(*SEATTLE_WGS84_BOUNDS)),
        assets={
            "B04": StacAsset(
                href="https://example.com/l_red.tif", title=None, type=None, roles=None
            )
        },
        time_range=(
            datetime(2020, 7, 21, tzinfo=UTC),
            datetime(2020, 7, 22, tzinfo=UTC),
        ),
    )
    earlier = StacItem(
        id="earlier",
        properties={"datetime": "2020-07-20T00:00:00Z"},
        collection=Hls2S30.COLLECTION_NAME,
        bbox=SEATTLE_WGS84_BOUNDS,
        geometry=shapely.geometry.mapping(shapely.box(*SEATTLE_WGS84_BOUNDS)),
        assets={
            "B04": StacAsset(
                href="https://example.com/s_red.tif", title=None, type=None, roles=None
            )
        },
        time_range=(
            datetime(2020, 7, 20, tzinfo=UTC),
            datetime(2020, 7, 21, tzinfo=UTC),
        ),
    )

    monkeypatch.setattr(data_source.client, "search", lambda **kw: [later, earlier])
    groups = data_source.get_items(
        [geometry], QueryConfig(space_mode=SpaceMode.INTERSECTS, max_matches=5)
    )[0]

    assert [group.items[0].name for group in groups] == ["earlier", "later"]
