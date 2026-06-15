"""Data on Planetary Computer."""

import hashlib
import json
import os
import re
import tempfile
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any
from urllib.parse import urlparse

import numpy as np
import numpy.typing as npt
import planetary_computer
import rasterio
import requests
import shapely
import xarray as xr
from typing_extensions import override
from upath import UPath

from rslearn.data_sources import DataSourceContext
from rslearn.data_sources.data_source import Item
from rslearn.data_sources.direct_materialize_data_source import (
    DirectMaterializeDataSource,
)
from rslearn.data_sources.stac import SourceItem, StacDataSource
from rslearn.data_sources.utils import MatchedItemGroup, match_candidate_items_to_window
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.fsspec import join_upath
from rslearn.utils.geometry import STGeometry
from rslearn.utils.interpolation import NODATA_VALUE, interpolate_to_grid
from rslearn.utils.raster_array import RasterArray, RasterMetadata
from rslearn.utils.raster_format import get_raster_projection_and_bounds
from rslearn.utils.stac import StacAsset, StacClient, StacItem

from .copernicus import get_harmonize_callback

logger = get_logger(__name__)

# Max limit accepted by Planetary Computer API.
PLANETARY_COMPUTER_LIMIT = 1000


class PlanetaryComputerMetadataBackend(StrEnum):
    """Metadata backend for Planetary Computer item discovery."""

    STAC = "stac"
    GEOPARQUET = "geoparquet"


def _datetime_to_utc(value: datetime) -> datetime:
    """Return a timezone-aware UTC datetime."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _quote_duckdb_identifier(identifier: str) -> str:
    """Quote a DuckDB identifier."""
    return '"' + identifier.replace('"', '""') + '"'


def _quote_duckdb_string(value: str) -> str:
    """Quote a DuckDB string literal."""
    return "'" + value.replace("'", "''") + "'"


class PlanetaryComputerGeoParquetClient:
    """A STAC-like client that searches Planetary Computer STAC GeoParquet items.

    Planetary Computer exposes collection-level ``geoparquet-items`` assets for bulk
    item metadata queries. For large prepare jobs this avoids one STAC API request per
    window, which can otherwise hit Planetary Computer API rate limits.
    """

    GEOPARQUET_ASSET_KEY = "geoparquet-items"
    _PARTITION_RE = re.compile(
        r"/part-\d+_"
        r"(?P<start>\d{4}-\d{2}-\d{2}T[^_]+)"
        r"_"
        r"(?P<end>\d{4}-\d{2}-\d{2}T[^_]+)"
        r"\.parquet$"
    )
    _SENTINEL2_ID_TIME_RE = re.compile(r"_MSIL2A_(?P<date>\d{8})T\d{6}_")

    def __init__(
        self,
        endpoint: str,
        collection_name: str,
        required_assets: list[str] | None,
        query_properties: list[str],
        timeout: timedelta,
        metadata_cache_dir: UPath | None = None,
        geoparquet_href: str | None = None,
    ) -> None:
        """Create a GeoParquet-backed client.

        Args:
            endpoint: Planetary Computer STAC endpoint.
            collection_name: STAC collection name.
            required_assets: asset keys that must exist on returned items.
            query_properties: STAC property columns needed for filtering/sorting.
            timeout: HTTP timeout for collection metadata and Azure listing requests.
            metadata_cache_dir: optional directory for file-list and query-result cache.
            geoparquet_href: optional explicit GeoParquet asset href.
        """
        self.endpoint = endpoint
        self.collection_name = collection_name
        self.required_assets = required_assets
        self.query_properties = query_properties
        self.timeout = timeout
        self.metadata_cache_dir = metadata_cache_dir
        self.geoparquet_href = geoparquet_href

        self._resolved_geoparquet_url: str | None = None
        self._partition_blob_names: list[str] | None = None
        self._items_by_name: dict[str, StacItem] = {}

        if self.metadata_cache_dir is not None:
            self.metadata_cache_dir.mkdir(parents=True, exist_ok=True)

    def search(
        self,
        collections: list[str] | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        intersects: dict[str, Any] | None = None,
        date_time: datetime | tuple[datetime, datetime] | None = None,
        ids: list[str] | None = None,
        limit: int | None = None,
        query: dict[str, Any] | None = None,
        sortby: list[dict[str, str]] | None = None,
    ) -> list[StacItem]:
        """Execute a STAC-like item search against STAC GeoParquet rows."""
        del limit

        if collections is not None and self.collection_name not in collections:
            return []

        if (
            ids is not None
            and bbox is None
            and intersects is None
            and date_time is None
            and query is None
            and sortby is None
            and all(item_id in self._items_by_name for item_id in ids)
        ):
            return [self._items_by_name[item_id] for item_id in ids]

        if bbox is None and intersects is not None:
            bbox = shapely.geometry.shape(intersects).bounds

        search_time_range = self._normalize_search_time_range(date_time)
        if search_time_range is None and ids is not None:
            search_time_range = self._infer_time_range_from_sentinel2_ids(ids)

        rows = self._read_geoparquet_rows(
            bbox=bbox,
            date_time=search_time_range,
            ids=ids,
            query=query,
            sortby=sortby,
        )
        stac_items = [self._row_to_stac_item(row) for row in rows]
        for stac_item in stac_items:
            self._items_by_name[stac_item.id] = stac_item
        return stac_items

    def _normalize_search_time_range(
        self, date_time: datetime | tuple[datetime, datetime] | None
    ) -> tuple[datetime, datetime] | None:
        """Normalize a STAC search datetime argument to a range."""
        if date_time is None:
            return None
        if isinstance(date_time, tuple):
            return (_datetime_to_utc(date_time[0]), _datetime_to_utc(date_time[1]))
        dt = _datetime_to_utc(date_time)
        return (dt, dt)

    def _infer_time_range_from_sentinel2_ids(
        self, ids: list[str]
    ) -> tuple[datetime, datetime] | None:
        """Infer a coarse time range from Sentinel-2 item IDs."""
        dates: list[datetime] = []
        for item_id in ids:
            match = self._SENTINEL2_ID_TIME_RE.search(item_id)
            if match is None:
                continue
            dates.append(
                datetime.strptime(match.group("date"), "%Y%m%d").replace(tzinfo=UTC)
            )
        if not dates:
            return None
        return (min(dates), max(dates) + timedelta(days=1))

    def _get_duckdb(self) -> Any:
        """Import duckdb lazily so GeoParquet support is optional."""
        try:
            import duckdb
        except (
            ImportError
        ) as e:  # pragma: no cover - exercised only without optional dep
            raise ImportError(
                "Planetary Computer GeoParquet metadata requires the optional "
                "'duckdb' dependency. Install rslearn with the 'extra' extras or "
                "install duckdb separately."
            ) from e
        return duckdb

    def _resolve_geoparquet_url(self) -> str:
        """Resolve the configured or collection-level GeoParquet href to HTTPS."""
        if self._resolved_geoparquet_url is not None:
            return self._resolved_geoparquet_url

        href = self.geoparquet_href
        storage_account: str | None = None
        if href is None:
            response = requests.get(
                f"{self.endpoint}/collections/{self.collection_name}",
                timeout=self.timeout.total_seconds(),
            )
            response.raise_for_status()
            collection = response.json()
            asset = collection.get("assets", {}).get(self.GEOPARQUET_ASSET_KEY)
            if asset is None:
                raise ValueError(
                    f"Planetary Computer collection {self.collection_name!r} has no "
                    f"{self.GEOPARQUET_ASSET_KEY!r} asset"
                )
            href = asset["href"]
            storage_account = asset.get("table:storage_options", {}).get("account_name")

        parsed = urlparse(href)
        if parsed.scheme in ("http", "https", "file"):
            self._resolved_geoparquet_url = href
            return href
        if parsed.scheme == "abfs":
            if not storage_account:
                raise ValueError(
                    f"GeoParquet href {href!r} requires table:storage_options.account_name"
                )
            container = parsed.netloc
            blob = parsed.path.lstrip("/")
            self._resolved_geoparquet_url = (
                f"https://{storage_account}.blob.core.windows.net/{container}/{blob}"
            )
            return self._resolved_geoparquet_url

        raise ValueError(f"unsupported GeoParquet href scheme in {href!r}")

    def _get_signed_query_string(self, base_url: str) -> str:
        """Return a Planetary Computer SAS query string for an HTTPS Blob URL."""
        signed = planetary_computer.sign(base_url)
        if "?" not in signed:
            return ""
        return signed.split("?", 1)[1]

    def _list_partition_blob_names(self) -> list[str]:
        """List partition parquet blob names under the GeoParquet directory."""
        if self._partition_blob_names is not None:
            return self._partition_blob_names

        cache_path = None
        if self.metadata_cache_dir is not None:
            cache_path = self.metadata_cache_dir / "geoparquet_partitions.json"
            if cache_path.exists():
                with cache_path.open() as f:
                    self._partition_blob_names = json.load(f)
                return self._partition_blob_names

        base_url = self._resolve_geoparquet_url()
        parsed = urlparse(base_url)
        if parsed.scheme == "file" or parsed.scheme == "":
            self._partition_blob_names = [base_url]
            return self._partition_blob_names
        if parsed.scheme != "https":
            raise ValueError(f"expected HTTPS GeoParquet URL, got {base_url!r}")

        account_url = f"{parsed.scheme}://{parsed.netloc}"
        path_parts = parsed.path.lstrip("/").split("/", 1)
        if len(path_parts) != 2:
            raise ValueError(f"invalid Azure Blob GeoParquet URL {base_url!r}")
        container, blob_prefix = path_parts

        query_string = self._get_signed_query_string(base_url)
        names: list[str] = []
        marker = ""
        while True:
            list_url = (
                f"{account_url}/{container}?restype=container&comp=list"
                f"&prefix={blob_prefix}/&maxresults=5000"
            )
            if marker:
                list_url += f"&marker={marker}"
            if query_string:
                list_url += f"&{query_string}"
            response = requests.get(list_url, timeout=self.timeout.total_seconds())
            response.raise_for_status()

            root = ET.fromstring(response.content)
            for name_el in root.findall(".//Blob/Name"):
                if name_el.text and name_el.text.endswith(".parquet"):
                    names.append(name_el.text)

            marker_el = root.find("NextMarker")
            if marker_el is None or not marker_el.text:
                break
            marker = marker_el.text

        self._partition_blob_names = names
        if cache_path is not None:
            with cache_path.open("w") as f:
                json.dump(names, f)
        return names

    def _partition_time_range(
        self, blob_name_or_url: str
    ) -> tuple[datetime, datetime] | None:
        """Parse the time range encoded in a Planetary Computer partition filename."""
        match = self._PARTITION_RE.search(blob_name_or_url)
        if match is None:
            return None
        return (
            datetime.fromisoformat(match.group("start")),
            datetime.fromisoformat(match.group("end")),
        )

    def _get_parquet_files(
        self, date_time: tuple[datetime, datetime] | None
    ) -> list[str]:
        """Return signed Parquet files relevant to the requested time range."""
        base_url = self._resolve_geoparquet_url()
        parsed = urlparse(base_url)
        if parsed.scheme == "file" or parsed.scheme == "":
            return [base_url]

        path_parts = parsed.path.lstrip("/").split("/", 1)
        if len(path_parts) != 2:
            raise ValueError(f"invalid Azure Blob GeoParquet URL {base_url!r}")
        container, _ = path_parts
        account_url = f"{parsed.scheme}://{parsed.netloc}"
        query_string = self._get_signed_query_string(base_url)

        files = []
        for blob_name in self._list_partition_blob_names():
            partition_time_range = self._partition_time_range(blob_name)
            if date_time is not None and partition_time_range is not None:
                if (
                    partition_time_range[1] < date_time[0]
                    or partition_time_range[0] > date_time[1]
                ):
                    continue
            url = f"{account_url}/{container}/{blob_name}"
            if query_string:
                url += f"?{query_string}"
            files.append(url)
        return files

    def _cache_path(
        self,
        *,
        bbox: tuple[float, float, float, float] | None,
        date_time: tuple[datetime, datetime] | None,
        ids: list[str] | None,
        query: dict[str, Any] | None,
        sortby: list[dict[str, str]] | None,
    ) -> UPath | None:
        """Return a local cache path for a GeoParquet query result."""
        if self.metadata_cache_dir is None:
            return None
        parsed_cache = urlparse(str(self.metadata_cache_dir))
        if parsed_cache.scheme not in ("", "file"):
            return None
        key_payload = {
            "collection": self.collection_name,
            "geoparquet_url": self._resolve_geoparquet_url(),
            "bbox": bbox,
            "date_time": [dt.isoformat() for dt in date_time] if date_time else None,
            "ids": sorted(ids) if ids else None,
            "query": query,
            "sortby": sortby,
            "required_assets": self.required_assets,
            "query_properties": self.query_properties,
            "version": 1,
        }
        key = hashlib.sha256(
            json.dumps(key_payload, sort_keys=True, default=str).encode()
        ).hexdigest()
        query_cache_dir = self.metadata_cache_dir / "queries"
        query_cache_dir.mkdir(parents=True, exist_ok=True)
        return query_cache_dir / f"{key}.parquet"

    def _query_condition_sql(
        self, query: dict[str, Any] | None, params: list[Any]
    ) -> list[str]:
        """Translate supported STAC query predicates to DuckDB SQL."""
        if query is None:
            return []

        conditions = []
        op_to_sql = {
            "eq": "=",
            "lt": "<",
            "lte": "<=",
            "gt": ">",
            "gte": ">=",
        }
        for prop_name, predicate in query.items():
            column = _quote_duckdb_identifier(prop_name)
            if not isinstance(predicate, dict):
                conditions.append(f"{column} = ?")
                params.append(predicate)
                continue
            for op, value in predicate.items():
                if op in op_to_sql:
                    conditions.append(f"{column} {op_to_sql[op]} ?")
                    params.append(value)
                elif op == "in":
                    if not isinstance(value, list) or len(value) == 0:
                        raise ValueError(
                            f"STAC query operator 'in' for {prop_name!r} requires a non-empty list"
                        )
                    placeholders = ", ".join("?" for _ in value)
                    conditions.append(f"{column} IN ({placeholders})")
                    params.extend(value)
                else:
                    raise ValueError(
                        f"unsupported STAC query operator {op!r} for GeoParquet backend"
                    )
        return conditions

    def _build_select_sql(
        self,
        files_expr: str,
        *,
        bbox: tuple[float, float, float, float] | None,
        date_time: tuple[datetime, datetime] | None,
        ids: list[str] | None,
        query: dict[str, Any] | None,
        sortby: list[dict[str, str]] | None,
        params: list[Any],
    ) -> str:
        """Build a DuckDB SELECT over STAC GeoParquet files."""
        property_columns = []
        for prop_name in self.query_properties:
            if prop_name == "datetime" or prop_name in property_columns:
                continue
            property_columns.append(prop_name)

        select_columns = [
            "id",
            "bbox",
            "datetime",
            "collection",
            "assets",
            "geometry",
            *[_quote_duckdb_identifier(prop) for prop in property_columns],
        ]
        conditions = ["collection = ?"]
        params.append(self.collection_name)

        if date_time is not None:
            conditions.append("datetime >= ? AND datetime <= ?")
            params.extend(date_time)
        if bbox is not None:
            west, south, east, north = bbox
            conditions.append(
                "bbox.xmin < ? AND bbox.xmax > ? AND bbox.ymin < ? AND bbox.ymax > ?"
            )
            params.extend([east, west, north, south])
        if ids is not None:
            if len(ids) == 0:
                conditions.append("FALSE")
            else:
                placeholders = ", ".join("?" for _ in ids)
                conditions.append(f"id IN ({placeholders})")
                params.extend(ids)
        if self.required_assets is not None:
            for asset_key in self.required_assets:
                conditions.append(
                    f"assets.{_quote_duckdb_identifier(asset_key)}.href IS NOT NULL"
                )

        conditions.extend(self._query_condition_sql(query, params))

        order_by = ""
        if sortby is not None:
            order_clauses = []
            for spec in sortby:
                direction = spec.get("direction", "asc").upper()
                if direction not in ("ASC", "DESC"):
                    raise ValueError(f"invalid sort direction {direction!r}")
                order_clauses.append(
                    f"{_quote_duckdb_identifier(spec['field'])} {direction}"
                )
            if order_clauses:
                order_by = " ORDER BY " + ", ".join(order_clauses)

        return (
            f"SELECT {', '.join(select_columns)} FROM read_parquet({files_expr}) "
            f"WHERE {' AND '.join(conditions)}{order_by}"
        )

    def _fetch_rows(
        self, con: Any, sql: str, params: list[Any]
    ) -> list[dict[str, Any]]:
        """Fetch DuckDB query results as dictionaries."""
        cursor = con.execute(sql, params)
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def _read_geoparquet_rows(
        self,
        *,
        bbox: tuple[float, float, float, float] | None,
        date_time: tuple[datetime, datetime] | None,
        ids: list[str] | None,
        query: dict[str, Any] | None,
        sortby: list[dict[str, str]] | None,
    ) -> list[dict[str, Any]]:
        """Read matching rows from local cache or remote GeoParquet partitions."""
        cache_path = self._cache_path(
            bbox=bbox, date_time=date_time, ids=ids, query=query, sortby=sortby
        )
        duckdb = self._get_duckdb()
        con = duckdb.connect()

        if (
            cache_path is not None
            and cache_path.exists()
            and (cache_path.parent / f"{cache_path.name}.done").exists()
        ):
            return self._fetch_rows(
                con, "SELECT * FROM read_parquet(?)", [str(cache_path)]
            )

        parquet_files = self._get_parquet_files(date_time)
        if not parquet_files:
            return []

        params: list[Any] = [parquet_files]
        select_sql = self._build_select_sql(
            "?",
            bbox=bbox,
            date_time=date_time,
            ids=ids,
            query=query,
            sortby=sortby,
            params=params,
        )

        if cache_path is None:
            return self._fetch_rows(con, select_sql, params)

        tmp_cache_path = cache_path.parent / f"{cache_path.name}.tmp"
        done_path = cache_path.parent / f"{cache_path.name}.done"
        if tmp_cache_path.exists():
            tmp_cache_path.unlink()
        copy_sql = (
            f"COPY ({select_sql}) TO {_quote_duckdb_string(str(tmp_cache_path))} "
            "(FORMAT PARQUET)"
        )
        con.execute(copy_sql, params)
        tmp_cache_path.rename(cache_path)
        done_path.touch()
        return self._fetch_rows(con, "SELECT * FROM read_parquet(?)", [str(cache_path)])

    def _row_to_stac_item(self, row: dict[str, Any]) -> StacItem:
        """Convert a GeoParquet row to the local StacItem representation."""
        bbox_dict = row.get("bbox")
        bbox = None
        if bbox_dict is not None:
            bbox = (
                bbox_dict["xmin"],
                bbox_dict["ymin"],
                bbox_dict["xmax"],
                bbox_dict["ymax"],
            )

        geometry_wkb = row.get("geometry")
        if geometry_wkb is not None:
            geometry = shapely.geometry.mapping(shapely.from_wkb(geometry_wkb))
        elif bbox is not None:
            geometry = shapely.geometry.mapping(shapely.box(*bbox))
        else:
            geometry = None

        raw_datetime = row["datetime"]
        if isinstance(raw_datetime, str):
            item_datetime = datetime.fromisoformat(raw_datetime.replace("Z", "+00:00"))
        else:
            item_datetime = raw_datetime
        item_datetime = _datetime_to_utc(item_datetime)

        assets = {}
        for asset_key, asset_dict in row.get("assets", {}).items():
            if asset_dict is None or asset_dict.get("href") is None:
                continue
            assets[asset_key] = StacAsset(
                href=asset_dict["href"],
                title=asset_dict.get("title"),
                type=asset_dict.get("type"),
                roles=asset_dict.get("roles"),
            )

        properties = {"datetime": item_datetime.isoformat()}
        for key, value in row.items():
            if key in {"id", "bbox", "datetime", "collection", "assets", "geometry"}:
                continue
            properties[key] = value

        return StacItem(
            id=row["id"],
            properties=properties,
            collection=row.get("collection"),
            bbox=bbox,
            geometry=geometry,
            assets=assets,
            time_range=(item_datetime, item_datetime),
        )


class PlanetaryComputerStacClient(StacClient):
    """A StacClient subclass that handles Planetary Computer's pagination limits.

    Planetary Computer STAC API does not support standard pagination and has a max
    limit of 1000. If the initial query returns 1000 items, this client paginates
    by sorting by ID and using gt (greater than) queries to fetch subsequent pages.
    """

    @override
    def search(
        self,
        collections: list[str] | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        intersects: dict[str, Any] | None = None,
        date_time: datetime | tuple[datetime, datetime] | None = None,
        ids: list[str] | None = None,
        limit: int | None = None,
        query: dict[str, Any] | None = None,
        sortby: list[dict[str, str]] | None = None,
    ) -> list[StacItem]:
        # We will use sortby for pagination, so the caller must not set it.
        if sortby is not None:
            raise ValueError("sortby must not be set for PlanetaryComputerStacClient")

        # First, try a simple query with the PC limit to detect if pagination is needed.
        # We always use PLANETARY_COMPUTER_LIMIT for the request because PC doesn't
        # support standard pagination, and we need to detect when we hit the limit
        # to switch to ID-based pagination.
        # We could just start sorting by ID here and do pagination, but we treate it as
        # a special case to avoid sorting since that seems to speed up the query.
        stac_items = super().search(
            collections=collections,
            bbox=bbox,
            intersects=intersects,
            date_time=date_time,
            ids=ids,
            limit=PLANETARY_COMPUTER_LIMIT,
            query=query,
        )

        # If we got fewer than the PC limit, we have all the results.
        if len(stac_items) < PLANETARY_COMPUTER_LIMIT:
            return stac_items

        # We hit the limit, so we need to paginate by ID.
        # Re-fetch with sorting by ID to ensure consistent ordering for pagination.
        logger.debug(
            "Initial request returned %d items (at limit), switching to ID pagination",
            len(stac_items),
        )

        all_items: list[StacItem] = []
        last_id: str | None = None

        while True:
            # Build query with id > last_id if we're paginating.
            combined_query: dict[str, Any] = dict(query) if query else {}
            if last_id is not None:
                combined_query["id"] = {"gt": last_id}

            stac_items = super().search(
                collections=collections,
                bbox=bbox,
                intersects=intersects,
                date_time=date_time,
                ids=ids,
                limit=PLANETARY_COMPUTER_LIMIT,
                query=combined_query if combined_query else None,
                sortby=[{"field": "id", "direction": "asc"}],
            )

            all_items.extend(stac_items)

            # If we got fewer than the limit, we've fetched everything.
            if len(stac_items) < PLANETARY_COMPUTER_LIMIT:
                break

            # Otherwise, paginate using the last item's ID.
            last_id = stac_items[-1].id
            logger.debug(
                "Got %d items, paginating with id > %s",
                len(stac_items),
                last_id,
            )

        logger.debug("Total items fetched: %d", len(all_items))
        return all_items


class PlanetaryComputer(DirectMaterializeDataSource[SourceItem], StacDataSource):
    """Modality-agnostic data source for data on Microsoft Planetary Computer.

    If there is a subclass available for a modality, it is recommended to use the
    subclass since it provides additional functionality.

    Otherwise, PlanetaryComputer can be configured with the collection name and a
    dictionary of assets and bands to ingest.

    See https://planetarycomputer.microsoft.com/ for details.

    The PC_SDK_SUBSCRIPTION_KEY environment variable can be set for higher rate limits
    but is not needed.
    """

    STAC_ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1"
    client: StacClient | PlanetaryComputerGeoParquetClient

    def __init__(
        self,
        collection_name: str,
        asset_bands: dict[str, list[str]],
        query: dict[str, Any] | None = None,
        sort_by: str | None = None,
        sort_ascending: bool = True,
        timeout: timedelta = timedelta(seconds=10),
        skip_items_missing_assets: bool = False,
        cache_dir: str | None = None,
        metadata_backend: PlanetaryComputerMetadataBackend = PlanetaryComputerMetadataBackend.STAC,
        metadata_cache_dir: str | None = None,
        geoparquet_href: str | None = None,
        context: DataSourceContext = DataSourceContext(),
    ):
        """Initialize a new PlanetaryComputer instance.

        Args:
            collection_name: the STAC collection name on Planetary Computer.
            asset_bands: assets to ingest, mapping from asset name to the list of bands
                in that asset.
            query: optional query argument to STAC searches.
            sort_by: sort by this property in the STAC items.
            sort_ascending: whether to sort ascending (or descending).
            timeout: timeout for API requests.
            skip_items_missing_assets: skip STAC items that are missing any of the
                assets in asset_bands during get_items.
            cache_dir: deprecated, no longer used. Item data is now passed to
                materialization functions so caching in the data source is unnecessary.
            metadata_backend: item metadata backend, either "stac" or "geoparquet".
                The GeoParquet backend performs one bulk query per get_items call and
                avoids repeated Planetary Computer STAC API search requests.
            metadata_cache_dir: optional directory for GeoParquet partition-list and
                query-result caches. Relative paths are resolved against the dataset
                path when available.
            geoparquet_href: optional explicit GeoParquet asset href. If omitted, the
                GeoParquet backend resolves the collection's "geoparquet-items" asset.
            context: the data source context.
        """
        if cache_dir is not None:
            warnings.warn(
                "cache_dir is deprecated and no longer used. "
                "Item data is now passed directly during materialization.",
                FutureWarning,
                stacklevel=2,
            )

        self.metadata_backend = PlanetaryComputerMetadataBackend(metadata_backend)

        # Initialize the DirectMaterializeDataSource with asset_bands
        DirectMaterializeDataSource.__init__(self, asset_bands=asset_bands)

        # We pass required_assets to StacDataSource if skip_items_missing_assets is set.
        required_assets: list[str] | None = None
        if skip_items_missing_assets:
            required_assets = list(asset_bands.keys())

        StacDataSource.__init__(
            self,
            endpoint=self.STAC_ENDPOINT,
            collection_name=collection_name,
            query=query,
            sort_by=sort_by,
            sort_ascending=sort_ascending,
            required_assets=required_assets,
        )

        self.collection_name = (
            collection_name if isinstance(collection_name, str) else None
        )

        if self.metadata_backend == PlanetaryComputerMetadataBackend.STAC:
            # Replace the client with PlanetaryComputerStacClient to handle PC's
            # pagination limits.
            self.client = PlanetaryComputerStacClient(self.STAC_ENDPOINT)
        elif self.metadata_backend == PlanetaryComputerMetadataBackend.GEOPARQUET:
            if not isinstance(collection_name, str):
                raise ValueError("GeoParquet metadata backend requires one collection")
            query_properties: list[str] = []
            if query is not None:
                query_properties.extend(query.keys())
            if sort_by is not None and sort_by not in query_properties:
                query_properties.append(sort_by)
            metadata_cache_path = None
            if metadata_cache_dir is not None:
                if context.ds_path is not None:
                    metadata_cache_path = join_upath(
                        context.ds_path, metadata_cache_dir
                    )
                else:
                    metadata_cache_path = UPath(metadata_cache_dir)
            self.client = PlanetaryComputerGeoParquetClient(
                endpoint=self.STAC_ENDPOINT,
                collection_name=collection_name,
                required_assets=required_assets,
                query_properties=query_properties,
                timeout=timeout,
                metadata_cache_dir=metadata_cache_path,
                geoparquet_href=geoparquet_href,
            )
        else:
            raise ValueError(f"unsupported metadata_backend {metadata_backend!r}")

        self.timeout = timeout
        self.skip_items_missing_assets = skip_items_missing_assets

    # --- DirectMaterializeDataSource implementation ---

    def get_asset_url(self, item: SourceItem, asset_key: str) -> str:
        """Get the signed URL to read the asset for the given item and asset key.

        Args:
            item: the item.
            asset_key: the key identifying which asset to get.

        Returns:
            the signed URL to read the asset from.
        """
        return planetary_computer.sign(item.asset_urls[asset_key])

    def get_raster_bands(self, layer_name: str, item: Item) -> list[list[str]]:
        """Get the sets of bands that have been stored for the specified item.

        Args:
            layer_name: the layer name or alias.
            item: the item.

        Returns:
            a list of lists of bands that are in the tile store (with one raster
                stored corresponding to each inner list). If no rasters are ready for
                this item, returns empty list.
        """
        if not isinstance(item, SourceItem):
            raise TypeError(f"expected SourceItem, got {type(item)}")
        all_bands = []
        for asset_key, band_names in self.asset_bands.items():
            if asset_key not in item.asset_urls:
                continue
            all_bands.append(band_names)
        return all_bands

    # --- DataSource implementation ---

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[SourceItem],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        Args:
            tile_store: the tile store to ingest into
            items: the items to ingest
            geometries: a list of geometries needed for each item
        """
        for item in items:
            for asset_key, band_names in self.asset_bands.items():
                if asset_key not in item.asset_urls:
                    continue
                if tile_store.is_raster_ready(item, band_names):
                    continue

                asset_url = planetary_computer.sign(item.asset_urls[asset_key])

                with tempfile.TemporaryDirectory() as tmp_dir:
                    local_fname = os.path.join(tmp_dir, f"{asset_key}.tif")
                    logger.debug(
                        "PlanetaryComputer download item %s asset %s to %s",
                        item.name,
                        asset_key,
                        local_fname,
                    )
                    with requests.get(
                        asset_url, stream=True, timeout=self.timeout.total_seconds()
                    ) as r:
                        r.raise_for_status()
                        with open(local_fname, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)

                    logger.debug(
                        "PlanetaryComputer ingest item %s asset %s",
                        item.name,
                        asset_key,
                    )
                    tile_store.write_raster_file(
                        item,
                        band_names,
                        UPath(local_fname),
                        time_range=item.geometry.time_range,
                    )

                logger.debug(
                    "PlanetaryComputer done ingesting item %s asset %s",
                    item.name,
                    asset_key,
                )

    def _get_aggregate_search_time_range(
        self, geometries: list[STGeometry]
    ) -> tuple[datetime, datetime] | None:
        """Return the aggregate search time range for a batch of windows."""
        min_time = None
        max_time = None
        for geometry in geometries:
            search_time_range = self._get_search_time_range(geometry)
            if search_time_range is None:
                continue
            if isinstance(search_time_range, tuple):
                start, end = search_time_range
            else:
                start = end = search_time_range
            min_time = start if min_time is None else min(min_time, start)
            max_time = end if max_time is None else max(max_time, end)
        if min_time is None or max_time is None:
            return None
        return (min_time, max_time)

    def _get_aggregate_search_bbox(
        self, geometries: list[STGeometry]
    ) -> tuple[float, float, float, float] | None:
        """Return the aggregate WGS84 bbox for a batch of windows."""
        min_west = None
        min_south = None
        max_east = None
        max_north = None
        for geometry in geometries:
            west, south, east, north = geometry.shp.bounds
            min_west = west if min_west is None else min(min_west, west)
            min_south = south if min_south is None else min(min_south, south)
            max_east = east if max_east is None else max(max_east, east)
            max_north = north if max_north is None else max(max_north, north)
        if (
            min_west is None
            or min_south is None
            or max_east is None
            or max_north is None
        ):
            return None
        return (min_west, min_south, max_east, max_north)

    def _get_items_geoparquet(
        self, geometries: list[STGeometry], query_config: Any
    ) -> list[list[MatchedItemGroup[SourceItem]]]:
        """Get items using one batched GeoParquet metadata query."""
        wgs84_geometries = [geometry.to_wgs84() for geometry in geometries]
        aggregate_bbox = self._get_aggregate_search_bbox(wgs84_geometries)
        aggregate_time_range = self._get_aggregate_search_time_range(wgs84_geometries)

        stac_items = self.client.search(
            collections=self.collection_names,
            bbox=aggregate_bbox,
            date_time=aggregate_time_range,
            query=self.query,
            limit=self.limit,
        )
        logger.debug("GeoParquet search yielded %d items", len(stac_items))

        if self.required_assets is not None:
            good_stac_items = []
            for stac_item in stac_items:
                if stac_item.assets is None:
                    raise ValueError(f"got STAC item {stac_item.id} with no assets")

                good = True
                for asset_key in self.required_assets:
                    if asset_key in stac_item.assets:
                        continue
                    good = False
                    break
                if good:
                    good_stac_items.append(stac_item)
            logger.debug(
                "required_assets filter from %d to %d items",
                len(stac_items),
                len(good_stac_items),
            )
            stac_items = good_stac_items

        if self.sort_by is not None:
            stac_items.sort(
                key=self._get_sort_key,
                reverse=not self.sort_ascending,
            )

        candidate_items = []
        for stac_item in stac_items:
            candidate_item = self._stac_item_to_item(stac_item)
            if not self._should_include_item(candidate_item):
                continue
            candidate_items.append(candidate_item)

        groups = []
        for geometry, wgs84_geometry in zip(geometries, wgs84_geometries):
            cur_candidate_items = []
            for item in candidate_items:
                if (
                    wgs84_geometry.time_range is not None
                    and not wgs84_geometry.intersects_time_range(
                        item.geometry.time_range
                    )
                ):
                    continue
                if not wgs84_geometry.shp.intersects(item.geometry.shp):
                    continue
                cur_candidate_items.append(item)

            cur_groups = match_candidate_items_to_window(
                geometry, cur_candidate_items, query_config
            )
            groups.append(cur_groups)

        return groups

    @override
    def get_items(
        self, geometries: list[STGeometry], query_config: Any
    ) -> list[list[MatchedItemGroup[SourceItem]]]:
        """Get items intersecting the given geometries."""
        if self.metadata_backend == PlanetaryComputerMetadataBackend.GEOPARQUET:
            return self._get_items_geoparquet(geometries, query_config)
        return super().get_items(geometries, query_config)


class Sentinel2(PlanetaryComputer):
    """A data source for Sentinel-2 L2A data on Microsoft Planetary Computer.

    See https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a.
    """

    COLLECTION_NAME = "sentinel-2-l2a"

    BANDS = {
        "B01": ["B01"],
        "B02": ["B02"],
        "B03": ["B03"],
        "B04": ["B04"],
        "B05": ["B05"],
        "B06": ["B06"],
        "B07": ["B07"],
        "B08": ["B08"],
        "B09": ["B09"],
        "B11": ["B11"],
        "B12": ["B12"],
        "B8A": ["B8A"],
        "SCL": ["SCL"],
        "visual": ["R", "G", "B"],
    }
    NON_REFLECTANCE_ASSETS = frozenset({"SCL", "visual"})

    def __init__(
        self,
        harmonize: bool = False,
        assets: list[str] | None = None,
        context: DataSourceContext = DataSourceContext(),
        **kwargs: Any,
    ):
        """Initialize a new Sentinel2 instance.

        Args:
            harmonize: harmonize pixel values across different processing baselines,
                see https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
            assets: list of asset names to ingest, or None to ingest all assets. This
                is only used if the layer config is missing from the context.
            context: the data source context.
            kwargs: other arguments to pass to PlanetaryComputer.
        """
        self.harmonize = harmonize

        # Determine which assets we need based on the bands in the layer config.
        if context.layer_config is not None:
            asset_bands: dict[str, list[str]] = {}
            for asset_key, band_names in self.BANDS.items():
                # See if the bands provided by this asset intersect with the bands in
                # at least one configured band set.
                for band_set in context.layer_config.band_sets:
                    if not set(band_set.bands).intersection(set(band_names)):
                        continue
                    asset_bands[asset_key] = band_names
                    break
        elif assets is not None:
            asset_bands = {asset_key: self.BANDS[asset_key] for asset_key in assets}
        else:
            asset_bands = self.BANDS

        super().__init__(
            collection_name=self.COLLECTION_NAME,
            asset_bands=asset_bands,
            # Skip since all of the items should have the same assets.
            skip_items_missing_assets=True,
            context=context,
            **kwargs,
        )

    def _get_product_xml(self, item: SourceItem) -> ET.Element:
        asset_url = planetary_computer.sign(item.asset_urls["product-metadata"])
        response = requests.get(asset_url, timeout=self.timeout.total_seconds())
        response.raise_for_status()
        return ET.fromstring(response.content)

    def _should_harmonize_asset(self, asset_key: str) -> bool:
        """Return whether harmonization should be applied for this Sentinel-2 asset."""
        return self.harmonize and asset_key not in self.NON_REFLECTANCE_ASSETS

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[SourceItem],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        Args:
            tile_store: the tile store to ingest into
            items: the items to ingest
            geometries: a list of geometries needed for each item
        """
        for item in items:
            for asset_key, band_names in self.asset_bands.items():
                if tile_store.is_raster_ready(item, band_names):
                    continue

                asset_url = planetary_computer.sign(item.asset_urls[asset_key])

                with tempfile.TemporaryDirectory() as tmp_dir:
                    local_fname = os.path.join(tmp_dir, f"{asset_key}.tif")
                    logger.debug(
                        "PlanetaryComputer download item %s asset %s to %s",
                        item.name,
                        asset_key,
                        local_fname,
                    )
                    with requests.get(
                        asset_url, stream=True, timeout=self.timeout.total_seconds()
                    ) as r:
                        r.raise_for_status()
                        with open(local_fname, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)

                    logger.debug(
                        "PlanetaryComputer ingest item %s asset %s",
                        item.name,
                        asset_key,
                    )

                    # Harmonize reflectance assets if needed.
                    harmonize_callback = None
                    if self._should_harmonize_asset(asset_key):
                        harmonize_callback = get_harmonize_callback(
                            self._get_product_xml(item)
                        )

                    if harmonize_callback is not None:
                        # In this case we need to read the array, convert the pixel
                        # values, and pass modified array directly to the TileStore.
                        with rasterio.open(local_fname) as src:
                            array = src.read()
                            src_nodata = src.nodata
                            projection, bounds = get_raster_projection_and_bounds(src)
                        array = harmonize_callback(array)
                        raster_metadata = RasterMetadata(nodata_value=src_nodata)
                        tile_store.write_raster(
                            item,
                            band_names,
                            projection,
                            bounds,
                            RasterArray(
                                chw_array=array,
                                time_range=item.geometry.time_range,
                                metadata=raster_metadata,
                            ),
                        )

                    else:
                        tile_store.write_raster_file(
                            item,
                            band_names,
                            UPath(local_fname),
                            time_range=item.geometry.time_range,
                        )

                logger.debug(
                    "PlanetaryComputer done ingesting item %s asset %s",
                    item.name,
                    asset_key,
                )

    def get_read_callback(
        self, item: SourceItem, asset_key: str
    ) -> Callable[[npt.NDArray[Any]], npt.NDArray[Any]] | None:
        """Return a callback to harmonize Sentinel-2 data if needed.

        Args:
            item: the item being read.
            asset_key: the key identifying which asset is being read.

        Returns:
            A callback function for harmonization, or None if not needed.
        """
        if not self._should_harmonize_asset(asset_key):
            return None

        return get_harmonize_callback(self._get_product_xml(item))


class Hls2S30(PlanetaryComputer):
    """A data source for HLS v2 Sentinel-2 (S30) data on Planetary Computer."""

    COLLECTION_NAME = "hls2-s30"
    DEFAULT_PLATFORMS = ["sentinel-2a", "sentinel-2b", "sentinel-2c"]
    # Asset keys exposed by the collection.
    ASSET_KEY_TO_COMMON_NAME = {
        "B01": "coastal",
        "B02": "blue",
        "B03": "green",
        "B04": "red",
        "B08": "nir",
        "B10": "cirrus",
        "B11": "swir16",
        "B12": "swir22",
    }
    COMMON_NAME_TO_ASSET_KEY = {
        common: asset for asset, common in ASSET_KEY_TO_COMMON_NAME.items()
    }
    DEFAULT_BANDS = list(ASSET_KEY_TO_COMMON_NAME.keys())

    @classmethod
    def _normalize_band_name(cls, band: str) -> str:
        if band in cls.ASSET_KEY_TO_COMMON_NAME:
            return band
        if band in cls.COMMON_NAME_TO_ASSET_KEY:
            return cls.COMMON_NAME_TO_ASSET_KEY[band]
        raise ValueError(
            f"unsupported HLS2 S30 band '{band}'. Use one of {sorted(cls.ASSET_KEY_TO_COMMON_NAME.keys())} "
            f"(asset keys) or {sorted(cls.COMMON_NAME_TO_ASSET_KEY.keys())} (common names)."
        )

    def __init__(
        self,
        band_names: list[str] | None = None,
        platforms: list[str] | None = None,
        query: dict[str, Any] | None = None,
        context: DataSourceContext = DataSourceContext(),
        **kwargs: Any,
    ):
        """Initialize a new Hls2S30 instance.

        Args:
            band_names: optional list of bands to expose. If not provided and a layer
                config is present, bands are inferred from the band sets. Otherwise
                defaults to the HLS S30 reflectance bands.
            platforms: optional list of Sentinel-2 platform identifiers to include.
                Defaults to ["sentinel-2a", "sentinel-2b", "sentinel-2c"].
            query: optional STAC query filter to use. If not set, this defaults to a
                platform filter for the configured platforms.
            context: the data source context.
            kwargs: additional arguments to pass to PlanetaryComputer.
        """
        if context.layer_config is not None:
            requested_bands = {
                band
                for band_set in context.layer_config.band_sets
                for band in band_set.bands
            }
            band_names = [self._normalize_band_name(band) for band in requested_bands]
        elif band_names is None:
            band_names = self.DEFAULT_BANDS
        else:
            band_names = [self._normalize_band_name(band) for band in band_names]

        if platforms is None:
            platforms = self.DEFAULT_PLATFORMS

        if query is None:
            query = {"platform": {"in": platforms}}

        # Assets are keyed by band name; each asset is a single band.
        asset_bands = {band: [band] for band in band_names}

        super().__init__(
            collection_name=self.COLLECTION_NAME,
            asset_bands=asset_bands,
            query=query,
            # Skip per-item asset checks; required assets are derived from asset_bands.
            skip_items_missing_assets=True,
            context=context,
            **kwargs,
        )


class Hls2L30(PlanetaryComputer):
    """A data source for HLS v2 Landsat (L30) data on Planetary Computer."""

    COLLECTION_NAME = "hls2-l30"
    DEFAULT_PLATFORMS = ["landsat-8", "landsat-9"]
    ASSET_KEY_TO_COMMON_NAME = {
        "B01": "coastal",
        "B02": "blue",
        "B03": "green",
        "B04": "red",
        "B05": "nir",
        "B06": "swir16",
        "B07": "swir22",
        "B09": "cirrus",
        "B10": "lwir11",
        "B11": "lwir12",
    }
    COMMON_NAME_TO_ASSET_KEY = {
        common: asset for asset, common in ASSET_KEY_TO_COMMON_NAME.items()
    }
    DEFAULT_BANDS = list(ASSET_KEY_TO_COMMON_NAME.keys())

    @classmethod
    def _normalize_band_name(cls, band: str) -> str:
        if band in cls.ASSET_KEY_TO_COMMON_NAME:
            return band
        if band in cls.COMMON_NAME_TO_ASSET_KEY:
            return cls.COMMON_NAME_TO_ASSET_KEY[band]
        raise ValueError(
            f"unknown HLS2 L30 band '{band}'. Use one of {sorted(cls.ASSET_KEY_TO_COMMON_NAME.keys())} "
            f"(asset keys) or {sorted(cls.COMMON_NAME_TO_ASSET_KEY.keys())} (common names)."
        )

    def __init__(
        self,
        band_names: list[str] | None = None,
        platforms: list[str] | None = None,
        query: dict[str, Any] | None = None,
        context: DataSourceContext = DataSourceContext(),
        **kwargs: Any,
    ):
        """Initialize a new Hls2L30 instance.

        Args:
            band_names: optional list of bands to expose. If not provided and a layer
                config is present, bands are inferred from the band sets. Otherwise
                defaults to the HLS L30 reflectance bands.
            platforms: optional list of Landsat platform identifiers to include.
                Defaults to ["landsat-8", "landsat-9"].
            query: optional STAC query filter to use. If not set, this defaults to a
                platform filter for the configured platforms.
            context: the data source context.
            kwargs: additional arguments to pass to PlanetaryComputer.
        """
        if context.layer_config is not None:
            requested_bands = {
                band
                for band_set in context.layer_config.band_sets
                for band in band_set.bands
            }
            band_names = [self._normalize_band_name(band) for band in requested_bands]
        elif band_names is None:
            band_names = self.DEFAULT_BANDS
        else:
            band_names = [self._normalize_band_name(band) for band in band_names]

        if platforms is None:
            platforms = self.DEFAULT_PLATFORMS

        if query is None:
            query = {"platform": {"in": platforms}}

        asset_bands = {band: [band] for band in band_names}

        super().__init__(
            collection_name=self.COLLECTION_NAME,
            asset_bands=asset_bands,
            query=query,
            # Skip per-item asset checks; required assets are derived from asset_bands.
            skip_items_missing_assets=True,
            context=context,
            **kwargs,
        )


class LandsatC2L2(PlanetaryComputer):
    """A data source for Landsat Collection 2 Level-2 data on Planetary Computer.

    This data source targets Landsat 8/9 items in the `landsat-c2-l2` collection.
    Band names exposed by this data source are Landsat-style band identifiers
    (e.g. "B4", "B5", "B10") for maximum compatibility with
    `rslearn.data_sources.aws_landsat.LandsatOliTirs`.

    For convenience, configuration also accepts STAC `common_name` values (e.g. "red",
    "nir08") and STAC `eo:bands[].name` aliases (e.g. "OLI_B4", "TIRS_B10"), which are
    normalized to the Landsat-style band identifiers above.

    Note: this is Level-2 data, not Level-1. If you need Level-1-specific bands
    (e.g. panchromatic/cirrus or thermal band 11), use
    `rslearn.data_sources.aws_landsat.LandsatOliTirs`.
    """

    COLLECTION_NAME = "landsat-c2-l2"

    # Map STAC asset keys (common_name) to the Landsat band identifiers we expose.
    # Planetary Computer assets for `landsat-c2-l2` are keyed by common_name.
    ASSET_COMMON_NAME_TO_BAND = {
        "coastal": "B1",
        "blue": "B2",
        "green": "B3",
        "red": "B4",
        "nir08": "B5",
        "swir16": "B6",
        "swir22": "B7",
        "lwir11": "B10",
    }

    BAND_TO_ASSET_COMMON_NAME = {v: k for k, v in ASSET_COMMON_NAME_TO_BAND.items()}

    # STAC eo:bands name -> Landsat-style band identifiers.
    STAC_BAND_NAME_ALIASES = {
        "OLI_B1": "B1",
        "OLI_B2": "B2",
        "OLI_B3": "B3",
        "OLI_B4": "B4",
        "OLI_B5": "B5",
        "OLI_B6": "B6",
        "OLI_B7": "B7",
        "TIRS_B10": "B10",
    }

    DEFAULT_PLATFORM_QUERY = {"platform": {"in": ["landsat-8", "landsat-9"]}}

    @classmethod
    def _normalize_band_name(cls, band: str) -> str:
        if band in cls.BAND_TO_ASSET_COMMON_NAME:
            return band
        if band in cls.ASSET_COMMON_NAME_TO_BAND:
            return cls.ASSET_COMMON_NAME_TO_BAND[band]
        if band in cls.STAC_BAND_NAME_ALIASES:
            return cls.STAC_BAND_NAME_ALIASES[band]
        if band in {"B8", "B9", "B11"}:
            raise ValueError(
                f"LandsatC2L2 does not provide {band} in the Planetary Computer "
                "landsat-c2-l2 collection. Use rslearn.data_sources.aws_landsat.LandsatOliTirs "
                "for Level-1 bands like panchromatic (B8), cirrus (B9), or thermal band 11 (B11)."
            )
        raise ValueError(
            f"unknown Landsat band '{band}'. Use one of {sorted(cls.BAND_TO_ASSET_COMMON_NAME.keys())} "
            f"(Landsat band names), {sorted(cls.ASSET_COMMON_NAME_TO_BAND.keys())} (STAC common names), "
            f"or {sorted(cls.STAC_BAND_NAME_ALIASES.keys())} (STAC band names)."
        )

    def __init__(
        self,
        band_names: list[str] | None = None,
        query: dict[str, Any] | None = None,
        context: DataSourceContext = DataSourceContext(),
        **kwargs: Any,
    ):
        """Initialize a new LandsatC2L2 instance.

        Args:
            band_names: optional list of band names to expose. Values can be either
                STAC common names (preferred) or STAC `eo:bands[].name` aliases.
                If not provided, defaults to the reflectance bands listed in BANDS.
            query: optional STAC query filter to use. If not set, this defaults to a
                platform filter for Landsat 8/9. If set, the provided query is used
                as-is (no implicit platform filtering is added).
            context: the data source context.
            kwargs: additional arguments to pass to PlanetaryComputer.
        """
        # Prefer determining bands from the configured layer config (if present).
        if context.layer_config is not None:
            requested_bands = {
                band
                for band_set in context.layer_config.band_sets
                for band in band_set.bands
            }
            band_names = [self._normalize_band_name(band) for band in requested_bands]
        elif band_names is not None:
            band_names = [self._normalize_band_name(band) for band in band_names]
        else:
            band_names = list(self.BAND_TO_ASSET_COMMON_NAME.keys())

        # Landsat C2 L2 assets are keyed by common name; each asset is a single band.
        # We expose Landsat-style band identifiers (B1, B2, ...).
        asset_bands = {
            self.BAND_TO_ASSET_COMMON_NAME[band]: [band] for band in band_names
        }

        if query is None:
            query = self.DEFAULT_PLATFORM_QUERY

        super().__init__(
            collection_name=self.COLLECTION_NAME,
            asset_bands=asset_bands,
            query=query,
            # Skip per-item asset checks; required assets are derived from asset_bands.
            skip_items_missing_assets=True,
            context=context,
            **kwargs,
        )


class Sentinel1(PlanetaryComputer):
    """A data source for Sentinel-1 data on Microsoft Planetary Computer.

    This uses the radiometrically corrected data.

    See https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc.
    """

    COLLECTION_NAME = "sentinel-1-rtc"

    def __init__(
        self,
        band_names: list[str] | None = None,
        context: DataSourceContext = DataSourceContext(),
        **kwargs: Any,
    ):
        """Initialize a new Sentinel1 instance.

        Args:
            band_names: list of bands to try to ingest, if the layer config is missing
                from the context.
            context: the data source context.
            kwargs: additional arguments to pass to PlanetaryComputer.
        """
        # Get band names from the config if possible. If it isn't in the context, then
        # we have to use the provided band names.
        if context.layer_config is not None:
            band_names = list(
                {
                    band
                    for band_set in context.layer_config.band_sets
                    for band in band_set.bands
                }
            )
        if band_names is None:
            raise ValueError(
                "band_names must be set if layer config is not in the context"
            )
        # For Sentinel-1, the asset key should be the same as the band name (and all
        # assets have one band).
        asset_bands = {band: [band] for band in band_names}
        super().__init__(
            collection_name=self.COLLECTION_NAME,
            asset_bands=asset_bands,
            context=context,
            **kwargs,
        )


class Naip(PlanetaryComputer):
    """A data source for NAIP data on Microsoft Planetary Computer.

    See https://planetarycomputer.microsoft.com/dataset/naip.
    """

    COLLECTION_NAME = "naip"
    ASSET_BANDS = {"image": ["R", "G", "B", "NIR"]}

    def __init__(
        self,
        context: DataSourceContext = DataSourceContext(),
        **kwargs: Any,
    ):
        """Initialize a new Naip instance.

        Args:
            context: the data source context.
            kwargs: additional arguments to pass to PlanetaryComputer.
        """
        super().__init__(
            collection_name=self.COLLECTION_NAME,
            asset_bands=self.ASSET_BANDS,
            context=context,
            **kwargs,
        )


class CopDemGlo30(PlanetaryComputer):
    """A data source for Copernicus DEM GLO-30 (30m) on Microsoft Planetary Computer.

    See https://planetarycomputer.microsoft.com/dataset/cop-dem-glo-30.
    """

    COLLECTION_NAME = "cop-dem-glo-30"
    DATA_ASSET = "data"

    def __init__(
        self,
        band_name: str = "DEM",
        context: DataSourceContext = DataSourceContext(),
        **kwargs: Any,
    ):
        """Initialize a new CopDemGlo30 instance.

        Args:
            band_name: band name to use if the layer config is missing from the
                context.
            context: the data source context.
            kwargs: additional arguments to pass to PlanetaryComputer.
        """
        if context.layer_config is not None:
            if len(context.layer_config.band_sets) != 1:
                raise ValueError("expected a single band set")
            if len(context.layer_config.band_sets[0].bands) != 1:
                raise ValueError("expected band set to have a single band")
            band_name = context.layer_config.band_sets[0].bands[0]

        super().__init__(
            collection_name=self.COLLECTION_NAME,
            asset_bands={self.DATA_ASSET: [band_name]},
            # Skip since all items should have the same asset(s).
            skip_items_missing_assets=True,
            context=context,
            **kwargs,
        )

    def _stac_item_to_item(self, stac_item: Any) -> SourceItem:
        # Copernicus DEM is static; ignore item timestamps so it matches any window.
        item = super()._stac_item_to_item(stac_item)
        item.geometry = STGeometry(item.geometry.projection, item.geometry.shp, None)
        return item

    def _get_search_time_range(self, geometry: STGeometry) -> None:
        # Copernicus DEM is static; do not filter STAC searches by time.
        return None


class Sentinel3SlstrLST(PlanetaryComputer):
    """Sentinel-3 SLSTR L2 Land Surface Temperature data on Planetary Computer.

    This collection provides netCDF swaths with geolocation arrays. We interpolate
    the swath onto a regular lat/lon grid using linear interpolation during ingestion.
    Direct materialization is not supported; keep ingest enabled.

    Requires the optional netCDF/xarray dependencies (netCDF4/h5netcdf/h5py).
    """

    COLLECTION_NAME = "sentinel-3-slstr-lst-l2-netcdf"
    LST_ASSET_KEY = "lst-in"
    GEODETIC_ASSET_KEY = "slstr-geodetic-in"
    DEFAULT_BANDS = ["LST"]

    def __init__(
        self,
        sample_step: int = 20,
        nodata_value: float = 0.0,
        grid_resolution: float | None = None,
        context: DataSourceContext = DataSourceContext(),
        **kwargs: Any,
    ) -> None:
        """Initialize a new Sentinel3SlstrLST instance.

        Args:
            sample_step: stride (in pixels) for sampling the geodetic arrays when
                estimating grid resolution.
            nodata_value: value to use for missing data in the output GeoTIFF.
            grid_resolution: optional output grid resolution (degrees). If not set,
                it is estimated from the geodetic arrays.
            context: the data source context.
            kwargs: additional arguments to pass to PlanetaryComputer.
        """
        self.sample_step = max(1, sample_step)
        self.nodata_value = nodata_value
        self.grid_resolution = grid_resolution

        if context.layer_config is not None:
            requested_bands = {
                band
                for band_set in context.layer_config.band_sets
                for band in band_set.bands
            }
            if requested_bands != set(self.DEFAULT_BANDS):
                raise ValueError(
                    "Sentinel3SlstrLST only supports the LST band. "
                    f"Requested: {sorted(requested_bands)}"
                )

        self.band_names = self.DEFAULT_BANDS

        super().__init__(
            collection_name=self.COLLECTION_NAME,
            asset_bands={self.LST_ASSET_KEY: self.band_names},
            skip_items_missing_assets=True,
            context=context,
            **kwargs,
        )

    def _estimate_grid_resolution(
        self, lons: npt.NDArray[np.floating], lats: npt.NDArray[np.floating]
    ) -> float:
        """Estimate grid resolution in degrees from geodetic arrays."""
        if lons.shape != lats.shape:
            raise ValueError(
                f"expected lon/lat arrays to have same shape, got {lons.shape} and {lats.shape}"
            )
        step = max(1, self.sample_step)
        lons_s = lons[::step, ::step]
        lats_s = lats[::step, ::step]

        lon_diff = np.abs(np.diff(lons_s, axis=1)).ravel()
        lat_diff = np.abs(np.diff(lats_s, axis=0)).ravel()
        diffs = np.concatenate([lon_diff, lat_diff])
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size == 0:
            return 0.01
        return float(np.median(diffs))

    def _mask_geodetic_by_valid_data(
        self,
        lons: npt.NDArray[np.floating],
        lats: npt.NDArray[np.floating],
        data: npt.NDArray[np.floating],
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Mask lon/lat arrays where the data is invalid."""
        valid_mask = np.isfinite(data[0])
        lons = np.where(valid_mask, lons, np.nan)
        lats = np.where(valid_mask, lats, np.nan)
        return lons, lats

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[SourceItem],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store."""
        for item in items:
            if tile_store.is_raster_ready(item, self.band_names):
                continue

            if self.LST_ASSET_KEY not in item.asset_urls:
                logger.warning(
                    "Sentinel3SlstrLST item %s missing asset %s, skipping",
                    item.name,
                    self.LST_ASSET_KEY,
                )
                continue
            if self.GEODETIC_ASSET_KEY not in item.asset_urls:
                logger.warning(
                    "Sentinel3SlstrLST item %s missing asset %s, skipping",
                    item.name,
                    self.GEODETIC_ASSET_KEY,
                )
                continue

            lst_url = planetary_computer.sign(item.asset_urls[self.LST_ASSET_KEY])
            geodetic_url = planetary_computer.sign(
                item.asset_urls[self.GEODETIC_ASSET_KEY]
            )

            with tempfile.TemporaryDirectory() as tmp_dir:
                lst_path = os.path.join(tmp_dir, "lst-in.nc")
                geodetic_path = os.path.join(tmp_dir, "geodetic-in.nc")
                for url, path in ((lst_url, lst_path), (geodetic_url, geodetic_path)):
                    with requests.get(
                        url, stream=True, timeout=self.timeout.total_seconds()
                    ) as r:
                        r.raise_for_status()
                        with open(path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)

                with (
                    xr.open_dataset(lst_path, mask_and_scale=True) as lst_ds,
                    xr.open_dataset(geodetic_path, mask_and_scale=True) as geo_ds,
                ):
                    lons = np.asarray(geo_ds["longitude_in"].values, dtype=np.float64)
                    lats = np.asarray(geo_ds["latitude_in"].values, dtype=np.float64)

                    band_arrays = []
                    for band in self.band_names:
                        if band not in lst_ds:
                            raise ValueError(
                                f"Sentinel3SlstrLST band '{band}' not found in {self.LST_ASSET_KEY}"
                            )
                        band_arrays.append(
                            np.asarray(lst_ds[band].values, dtype=np.float32)
                        )

                    stack = np.stack(band_arrays, axis=0)
                    lons, lats = self._mask_geodetic_by_valid_data(lons, lats, stack)

                    grid_resolution = (
                        self.grid_resolution
                        if self.grid_resolution is not None
                        else self._estimate_grid_resolution(lons, lats)
                    )
                    logger.debug(
                        "SLSTR LST grid resolution (deg): %s",
                        grid_resolution,
                    )
                    gridded_array, projection, bounds = interpolate_to_grid(
                        data=stack,
                        lon=lons,
                        lat=lats,
                        grid_resolution=grid_resolution,
                    )

                    if self.nodata_value != NODATA_VALUE:
                        gridded_array = np.where(
                            gridded_array == NODATA_VALUE,
                            self.nodata_value,
                            gridded_array,
                        )

                raster_metadata = RasterMetadata(nodata_value=self.nodata_value)
                tile_store.write_raster(
                    item,
                    self.band_names,
                    projection,
                    bounds,
                    RasterArray(
                        chw_array=gridded_array,
                        time_range=item.geometry.time_range,
                        metadata=raster_metadata,
                    ),
                )

    def read_raster(
        self,
        layer_name: str,
        item: Item,
        bands: list[str],
        projection: Any,
        bounds: Any,
        resampling: Any = rasterio.enums.Resampling.bilinear,
    ) -> RasterArray:
        """Direct materialization is not supported for this data source."""
        raise NotImplementedError(
            "Sentinel3SlstrLST does not support direct materialization; set ingest=true."
        )
