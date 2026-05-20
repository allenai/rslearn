"""NASA HLS v2.0 data sources backed by LP DAAC / CMR STAC."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlparse

import boto3
import rasterio
import shapely
from rasterio.errors import RasterioIOError
from rasterio.session import AWSSession
from typing_extensions import override
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import DataSourceContext
from rslearn.data_sources.data_source import Item
from rslearn.data_sources.direct_materialize_data_source import (
    DirectMaterializeDataSource,
)
from rslearn.data_sources.stac import AxisAlignedStacDataSource, SourceItem
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.geometry import STGeometry
from rslearn.utils.raster_array import RasterMetadata
from rslearn.utils.retry_session import create_retry_session
from rslearn.utils.stac import StacItem

logger = get_logger(__name__)
_HTTP_URL_PROPERTY_PREFIX = "_http_url_"
_EARTHDATA_TOKEN_ENV = "EARTHDATA_TOKEN"  # nosec B105


@dataclass(frozen=True)
class _S3Credentials:
    """Temporary AWS credentials returned by LP DAAC's s3credentials endpoint."""

    access_key_id: str
    secret_access_key: str
    session_token: str
    expiration: datetime


class _EarthdataAuth:
    """Fetch and refresh temporary AWS credentials for LP DAAC protected buckets."""

    REFRESH_BUFFER = timedelta(minutes=5)

    def __init__(
        self,
        earthdata_token: str | None,
        s3_credentials_url: str,
        aws_region: str,
        timeout: timedelta,
    ) -> None:
        self.earthdata_token = earthdata_token or os.environ.get(_EARTHDATA_TOKEN_ENV)
        self.s3_credentials_url = s3_credentials_url
        self.aws_region = aws_region
        self.timeout = timeout
        self.session = create_retry_session()
        self._credentials: _S3Credentials | None = None

    def _parse_credentials(self, data: dict[str, Any]) -> _S3Credentials:
        expiration = datetime.fromisoformat(data["expiration"])
        if expiration.tzinfo is None:
            expiration = expiration.replace(tzinfo=UTC)
        return _S3Credentials(
            access_key_id=data["accessKeyId"],
            secret_access_key=data["secretAccessKey"],
            session_token=data["sessionToken"],
            expiration=expiration,
        )

    def _refresh_with_token(self) -> _S3Credentials:
        if self.earthdata_token is None:
            raise ValueError("earthdata token is not configured")

        response = self.session.get(
            self.s3_credentials_url,
            headers={"Authorization": f"Bearer {self.earthdata_token}"},
            timeout=self.timeout.total_seconds(),
        )
        response.raise_for_status()
        return self._parse_credentials(response.json())

    def get_s3_credentials(self, refresh: bool = False) -> _S3Credentials:
        """Return temporary S3 credentials, refreshing them as needed."""
        now = datetime.now(tz=UTC)
        if (
            not refresh
            and self._credentials is not None
            and now + self.REFRESH_BUFFER < self._credentials.expiration
        ):
            return self._credentials

        if self.earthdata_token is None:
            raise ValueError(
                f"NASA HLS requires an Earthdata bearer token. Set {_EARTHDATA_TOKEN_ENV}."
            )

        self._credentials = self._refresh_with_token()
        return self._credentials

    def get_boto3_session(self) -> boto3.session.Session:
        """Return a boto3 session configured with current temporary credentials."""
        credentials = self.get_s3_credentials()
        return boto3.session.Session(
            aws_access_key_id=credentials.access_key_id,
            aws_secret_access_key=credentials.secret_access_key,
            aws_session_token=credentials.session_token,
            region_name=self.aws_region,
        )

    def get_s3_client(self) -> Any:
        """Return a boto3 S3 client configured with current temporary credentials."""
        return self.get_boto3_session().client("s3", region_name=self.aws_region)


class _NasaHlsBase(DirectMaterializeDataSource[SourceItem], AxisAlignedStacDataSource):
    """Shared implementation for NASA HLS v2.0 raster data sources.

    CMR rejects non-axis-aligned ``intersects`` polygons, so searches use the
    axis-aligned STAC data source variant.
    """

    STAC_ENDPOINT = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"
    S3_CREDENTIALS_URL = "https://data.lpdaac.earthdatacloud.nasa.gov/s3credentials"
    AWS_REGION = "us-west-2"
    PROPERTIES_TO_RECORD = ["eo:cloud_cover"]

    SUPPORTED_BANDS: list[str] = []
    DEFAULT_BANDS: list[str] = []
    COLLECTION_NAME = ""

    @classmethod
    def _normalize_band_name(cls, band: str) -> str:
        if band not in cls.SUPPORTED_BANDS:
            raise ValueError(
                f"unsupported {cls.__name__} band '{band}'. Use one of "
                f"{sorted(cls.SUPPORTED_BANDS)}."
            )
        return band

    def __init__(
        self,
        band_names: list[str] | None = None,
        query: dict[str, Any] | None = None,
        sort_by: str | None = None,
        sort_ascending: bool = True,
        timeout: timedelta = timedelta(seconds=30),
        earthdata_token: str | None = None,
        s3_credentials_url: str = S3_CREDENTIALS_URL,
        context: DataSourceContext = DataSourceContext(),
        collection_name: str | list[str] | None = None,
        require_asset_filter: bool = True,
    ) -> None:
        self.timeout = timeout
        self.auth = _EarthdataAuth(
            earthdata_token=earthdata_token,
            s3_credentials_url=s3_credentials_url,
            aws_region=self.AWS_REGION,
            timeout=timeout,
        )
        # Track which HTTPS fallback messages have already been emitted at INFO.
        # Repeated fallback events are still logged at DEBUG to reduce log noise.
        self._https_fallback_info_keys: set[tuple[str, str]] = set()

        if context.layer_config is not None:
            requested_bands: list[str] = []
            for band_set in context.layer_config.band_sets:
                for band in band_set.bands:
                    normalized_band = self._normalize_band_name(band)
                    if normalized_band not in requested_bands:
                        requested_bands.append(normalized_band)
            band_names = requested_bands
        elif band_names is None:
            band_names = list(self.DEFAULT_BANDS)
        else:
            band_names = [self._normalize_band_name(band) for band in band_names]

        asset_bands = {band: [band] for band in band_names}
        if collection_name is None:
            collection_name = self.COLLECTION_NAME
        required_assets = list(asset_bands.keys()) if require_asset_filter else None

        DirectMaterializeDataSource.__init__(self, asset_bands=asset_bands)
        AxisAlignedStacDataSource.__init__(
            self,
            endpoint=self.STAC_ENDPOINT,
            collection_name=collection_name,
            query=query,
            sort_by=sort_by,
            sort_ascending=sort_ascending,
            required_assets=required_assets,
            properties_to_record=list(self.PROPERTIES_TO_RECORD),
        )

    def _log_https_fallback(
        self,
        fallback_kind: str,
        item_name: str,
        asset_key: str,
    ) -> None:
        """Log HTTPS fallback with reduced repetition.

        The first fallback per (kind, asset_key) is INFO. Subsequent ones are DEBUG.
        """
        log_key = (fallback_kind, asset_key)
        if log_key not in self._https_fallback_info_keys:
            self._https_fallback_info_keys.add(log_key)
            logger.info(
                "falling back to HTTPS %s for %s asset %s (future repeats logged at DEBUG)",
                fallback_kind,
                item_name,
                asset_key,
            )
            return

        logger.debug(
            "falling back to HTTPS %s for %s asset %s",
            fallback_kind,
            item_name,
            asset_key,
        )

    def _get_http_asset_url(self, item: SourceItem, asset_key: str) -> str | None:
        """Return the HTTPS asset URL recorded for the given item and asset key."""
        value = item.properties.get(f"{_HTTP_URL_PROPERTY_PREFIX}{asset_key}")
        if isinstance(value, str):
            return value
        return None

    @contextmanager
    def _rasterio_env(self, asset_url: str) -> Iterator[None]:
        """Create a rasterio environment for the given asset URL."""
        if asset_url.startswith("s3://"):
            with rasterio.Env(session=AWSSession(self.auth.get_boto3_session())):
                yield
            return
        yield

    def _read_raster_from_local_copy(
        self,
        asset_url: str,
        projection: Any,
        bounds: Any,
        resampling: Any,
    ) -> tuple[Any, float | None]:
        """Download an asset locally, then read it with rasterio."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_fname = os.path.join(tmp_dir, "asset.tif")
            self._download_asset(asset_url, local_fname)
            return super()._read_raster_from_url(
                f"file://{local_fname}", projection, bounds, resampling
            )

    def _get_nodata_from_local_copy(self, asset_url: str) -> float | None:
        """Download an asset locally to inspect its nodata metadata."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_fname = os.path.join(tmp_dir, "asset.tif")
            self._download_asset(asset_url, local_fname)
            with rasterio.open(local_fname) as src:
                return src.nodata

    def _download_asset(self, asset_url: str, local_fname: str) -> None:
        """Download an asset to a local path for ingest."""
        if asset_url.startswith("s3://"):
            parsed = urlparse(asset_url)
            self.auth.get_s3_client().download_file(
                parsed.netloc, parsed.path.lstrip("/"), local_fname
            )
            return

        headers = {}
        if self.auth.earthdata_token is not None:
            headers["Authorization"] = f"Bearer {self.auth.earthdata_token}"

        with self.auth.session.get(
            asset_url,
            headers=headers,
            stream=True,
            timeout=self.timeout.total_seconds(),
        ) as response:
            response.raise_for_status()
            with open(local_fname, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)

    @override
    def _stac_item_to_item(self, stac_item: StacItem) -> SourceItem:
        if stac_item.geometry is None:
            raise ValueError("got unexpected item with no geometry")
        if stac_item.time_range is None:
            raise ValueError("got unexpected item with no time range")
        if stac_item.assets is None:
            raise ValueError("got unexpected item with no assets")

        shp = shapely.geometry.shape(stac_item.geometry)
        geometry = STGeometry(WGS84_PROJECTION, shp, stac_item.time_range)
        asset_urls = {}
        properties: dict[str, Any] = {}
        for asset_key in self.asset_bands:
            s3_asset_key = f"s3_{asset_key}"
            if s3_asset_key in stac_item.assets:
                asset_urls[asset_key] = stac_item.assets[s3_asset_key].href
            if asset_key in stac_item.assets:
                properties[f"{_HTTP_URL_PROPERTY_PREFIX}{asset_key}"] = (
                    stac_item.assets[asset_key].href
                )
                if asset_key not in asset_urls:
                    asset_urls[asset_key] = stac_item.assets[asset_key].href

        for prop_name in self.properties_to_record:
            if prop_name in stac_item.properties:
                properties[prop_name] = stac_item.properties[prop_name]

        return SourceItem(stac_item.id, geometry, asset_urls, properties)

    @override
    def get_asset_url(self, item: SourceItem, asset_key: str) -> str:
        return item.asset_urls[asset_key]

    @override
    def get_raster_bands(self, layer_name: str, item: Item) -> list[list[str]]:
        if not isinstance(item, SourceItem):
            raise TypeError(f"expected SourceItem, got {type(item)}")
        return [
            band_names
            for asset_key, band_names in self.asset_bands.items()
            if asset_key in item.asset_urls
        ]

    def _download_asset_with_fallback(
        self, item: SourceItem, asset_key: str, local_fname: str
    ) -> None:
        """Download an asset, retrying via HTTPS if direct S3 access fails."""
        asset_url = self.get_asset_url(item, asset_key)
        try:
            self._download_asset(asset_url, local_fname)
        except Exception:
            http_asset_url = self._get_http_asset_url(item, asset_key)
            if not asset_url.startswith("s3://") or http_asset_url is None:
                raise
            self._log_https_fallback("download", item.name, asset_key)
            self._download_asset(http_asset_url, local_fname)

    def _read_raster_for_item(
        self,
        item: SourceItem,
        asset_key: str,
        projection: Any,
        bounds: Any,
        resampling: Any,
    ) -> tuple[Any, float | None]:
        """Read an asset, retrying via HTTPS download if direct S3 access fails."""
        asset_url = self.get_asset_url(item, asset_key)
        try:
            with self._rasterio_env(asset_url):
                return super()._read_raster_from_url(
                    asset_url, projection, bounds, resampling
                )
        except RasterioIOError:
            http_asset_url = self._get_http_asset_url(item, asset_key)
            if not asset_url.startswith("s3://") or http_asset_url is None:
                raise
            self._log_https_fallback("raster read", item.name, asset_key)
            return self._read_raster_from_local_copy(
                http_asset_url, projection, bounds, resampling
            )

    def _get_nodata_for_item(self, item: SourceItem, asset_key: str) -> float | None:
        """Read nodata metadata, retrying via HTTPS download if needed."""
        asset_url = self.get_asset_url(item, asset_key)
        try:
            with self._rasterio_env(asset_url):
                with rasterio.open(asset_url) as src:
                    return src.nodata
        except RasterioIOError:
            http_asset_url = self._get_http_asset_url(item, asset_key)
            if not asset_url.startswith("s3://") or http_asset_url is None:
                raise
            self._log_https_fallback("nodata read", item.name, asset_key)
            return self._get_nodata_from_local_copy(http_asset_url)

    @override
    def _read_raster_from_url(
        self,
        url: str,
        projection: Any,
        bounds: Any,
        resampling: Any,
    ) -> tuple[Any, float | None]:
        with self._rasterio_env(url):
            return super()._read_raster_from_url(url, projection, bounds, resampling)

    @override
    def get_raster_metadata(
        self, layer_name: str, item: Item, bands: list[str]
    ) -> RasterMetadata:
        typed_item = item
        if not isinstance(typed_item, SourceItem):
            raise TypeError(f"expected SourceItem, got {type(item)}")

        asset_key = self._get_asset_key_by_bands(bands)
        if asset_key not in self._nodata_cache:
            self._nodata_cache[asset_key] = self._get_nodata_for_item(
                typed_item, asset_key
            )

        nodata = self._nodata_cache[asset_key]
        if nodata is not None:
            return RasterMetadata(nodata_value=nodata)
        return RasterMetadata()

    @override
    def read_raster(
        self,
        layer_name: str,
        item: Item,
        bands: list[str],
        projection: Any,
        bounds: Any,
        resampling: Any = rasterio.enums.Resampling.bilinear,
    ) -> Any:
        typed_item = item
        if not isinstance(typed_item, SourceItem):
            raise TypeError(f"expected SourceItem, got {type(item)}")

        asset_key = self._get_asset_key_by_bands(bands)
        raw_data, src_nodata = self._read_raster_for_item(
            typed_item, asset_key, projection, bounds, resampling
        )

        if asset_key not in self._nodata_cache:
            self._nodata_cache[asset_key] = src_nodata

        callback = self.get_read_callback(typed_item, asset_key)
        if callback is not None:
            raw_data = callback(raw_data)

        raster_metadata = None
        if src_nodata is not None:
            raster_metadata = RasterMetadata(nodata_value=src_nodata)

        from rslearn.utils.raster_array import RasterArray

        return RasterArray(
            chw_array=raw_data,
            time_range=item.geometry.time_range,
            metadata=raster_metadata,
        )

    @override
    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[SourceItem],
        geometries: list[list[STGeometry]],
    ) -> None:
        for item in items:
            for asset_key, band_names in self.asset_bands.items():
                if asset_key not in item.asset_urls:
                    continue
                if tile_store.is_raster_ready(item, band_names):
                    continue

                with tempfile.TemporaryDirectory() as tmp_dir:
                    local_fname = os.path.join(tmp_dir, f"{asset_key}.tif")
                    self._download_asset_with_fallback(item, asset_key, local_fname)
                    tile_store.write_raster_file(
                        item,
                        band_names,
                        UPath(local_fname),
                        time_range=item.geometry.time_range,
                    )


class Hls2S30(_NasaHlsBase):
    """NASA LP DAAC HLS v2.0 Sentinel-2 (HLSS30) data source."""

    COLLECTION_NAME = "HLSS30_2.0"
    DEFAULT_BANDS = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B11",
        "B12",
    ]
    SUPPORTED_BANDS = [
        *DEFAULT_BANDS,
        "B09",
        "B10",
        "Fmask",
        "SAA",
        "SZA",
        "VAA",
        "VZA",
    ]


class Hls2L30(_NasaHlsBase):
    """NASA LP DAAC HLS v2.0 Landsat (HLSL30) data source."""

    COLLECTION_NAME = "HLSL30_2.0"
    DEFAULT_BANDS = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B09",
        "B10",
        "B11",
    ]
    SUPPORTED_BANDS = [
        *DEFAULT_BANDS,
        "Fmask",
        "SAA",
        "SZA",
        "VAA",
        "VZA",
    ]


class Hls2(_NasaHlsBase):
    """Combined NASA HLS v2.0 time-series datasource with semantic band names."""

    # Internal label for the combined datasource; CMR STAC searches use
    # SOURCE_TO_COLLECTION below rather than a real "HLS2" collection.
    COLLECTION_NAME = "HLS2"
    SOURCE_TO_COLLECTION = {
        "sentinel": Hls2S30.COLLECTION_NAME,
        "landsat": Hls2L30.COLLECTION_NAME,
    }
    COLLECTION_TO_SOURCE = {
        collection_name: source
        for source, collection_name in SOURCE_TO_COLLECTION.items()
    }
    SENTINEL_SEMANTIC_BANDS = {
        "coastal": "B01",
        "blue": "B02",
        "green": "B03",
        "red": "B04",
        "nir": "B08",
        "cirrus": "B10",
        "swir16": "B11",
        "swir22": "B12",
        "fmask": "Fmask",
        "solar_azimuth": "SAA",
        "solar_zenith": "SZA",
        "view_azimuth": "VAA",
        "view_zenith": "VZA",
    }
    LANDSAT_SEMANTIC_BANDS = {
        "coastal": "B01",
        "blue": "B02",
        "green": "B03",
        "red": "B04",
        "nir": "B05",
        "cirrus": "B09",
        "swir16": "B06",
        "swir22": "B07",
        "fmask": "Fmask",
        "solar_azimuth": "SAA",
        "solar_zenith": "SZA",
        "view_azimuth": "VAA",
        "view_zenith": "VZA",
    }
    DEFAULT_BANDS = [
        "coastal",
        "blue",
        "green",
        "red",
        "nir",
        "swir16",
        "swir22",
    ]

    @classmethod
    def _asset_key_map_for_collection(cls, collection_name: str) -> dict[str, str]:
        if collection_name == Hls2S30.COLLECTION_NAME:
            return cls.SENTINEL_SEMANTIC_BANDS
        if collection_name == Hls2L30.COLLECTION_NAME:
            return cls.LANDSAT_SEMANTIC_BANDS
        raise ValueError(f"unsupported HLS2 collection {collection_name!r}")

    @classmethod
    def _normalize_band_name(cls, band: str) -> str:
        if band not in cls.SENTINEL_SEMANTIC_BANDS:
            raise ValueError(
                f"unsupported Hls2 band '{band}'. Use one of "
                f"{sorted(cls.SENTINEL_SEMANTIC_BANDS.keys())}."
            )
        return band

    def __init__(
        self,
        band_names: list[str] | None = None,
        sources: list[str] | None = None,
        query: dict[str, Any] | None = None,
        sort_by: str | None = "datetime",
        sort_ascending: bool = True,
        timeout: timedelta = timedelta(seconds=30),
        earthdata_token: str | None = None,
        s3_credentials_url: str = _NasaHlsBase.S3_CREDENTIALS_URL,
        context: DataSourceContext = DataSourceContext(),
    ) -> None:
        """Create a combined HLS datasource with semantic bands.

        Args:
            band_names: optional semantic bands to expose.
            sources: optional subset of sources to include: sentinel, landsat, or both.
            query: optional STAC query dict to include in searches.
            sort_by: sort merged STAC results by this property.
            sort_ascending: whether the sort should be ascending.
            timeout: timeout for auth and asset requests.
            earthdata_token: optional Earthdata bearer token override.
            s3_credentials_url: LP DAAC temporary credentials endpoint.
            context: optional datasource context from rslearn.
        """
        if sources is None:
            sources = ["sentinel", "landsat"]
        normalized_sources: list[str] = []
        for source in sources:
            if source not in self.SOURCE_TO_COLLECTION:
                raise ValueError(
                    f"unsupported Hls2 source '{source}'. Use one of "
                    f"{sorted(self.SOURCE_TO_COLLECTION.keys())}."
                )
            if source not in normalized_sources:
                normalized_sources.append(source)
        self.sources = normalized_sources
        self.collection_names = [
            self.SOURCE_TO_COLLECTION[source] for source in self.sources
        ]

        super().__init__(
            band_names=band_names,
            query=query,
            sort_by=sort_by,
            sort_ascending=sort_ascending,
            timeout=timeout,
            earthdata_token=earthdata_token,
            s3_credentials_url=s3_credentials_url,
            context=context,
            collection_name=self.collection_names,
            require_asset_filter=False,
        )

    @override
    def _stac_item_to_item(self, stac_item: StacItem) -> SourceItem:
        if stac_item.collection is None:
            raise ValueError("got unexpected item with no collection")
        if stac_item.geometry is None:
            raise ValueError("got unexpected item with no geometry")
        if stac_item.time_range is None:
            raise ValueError("got unexpected item with no time range")
        if stac_item.assets is None:
            raise ValueError("got unexpected item with no assets")

        asset_key_map = self._asset_key_map_for_collection(stac_item.collection)
        shp = shapely.geometry.shape(stac_item.geometry)
        geometry = STGeometry(WGS84_PROJECTION, shp, stac_item.time_range)
        asset_urls = {}
        properties: dict[str, Any] = {
            "sensor": self.COLLECTION_TO_SOURCE[stac_item.collection],
            "collection": stac_item.collection,
        }
        for semantic_band in self.asset_bands:
            stac_asset_key = asset_key_map.get(semantic_band)
            if stac_asset_key is None:
                continue
            s3_asset_key = f"s3_{stac_asset_key}"
            if s3_asset_key in stac_item.assets:
                asset_urls[semantic_band] = stac_item.assets[s3_asset_key].href
            if stac_asset_key in stac_item.assets:
                properties[f"{_HTTP_URL_PROPERTY_PREFIX}{semantic_band}"] = (
                    stac_item.assets[stac_asset_key].href
                )
                if semantic_band not in asset_urls:
                    asset_urls[semantic_band] = stac_item.assets[stac_asset_key].href

        for prop_name in self.properties_to_record:
            if prop_name in stac_item.properties:
                properties[prop_name] = stac_item.properties[prop_name]

        return SourceItem(stac_item.id, geometry, asset_urls, properties)

    @override
    def _get_sort_key(self, stac_item: StacItem) -> Any:
        assert self.sort_by is not None
        return (
            stac_item.properties.get(self.sort_by) is None,
            stac_item.properties.get(self.sort_by),
        )

    @override
    def _should_include_item(self, item: SourceItem) -> bool:
        return all(band in item.asset_urls for band in self.asset_bands)
