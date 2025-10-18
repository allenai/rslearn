"""Lightweight STAC API client utilities.

Portions adapted from pystac_client (https://github.com/stac-utils/pystac-client)
Copyright (c) pystac_client contributors
Licensed under Apache License 2.0

Modified for rslearn:
- Removed object caching and deserialization to pystac objects
- Simplified for direct JSON handling
- Optimized for lower memory footprint
"""

import json
import logging
import re
from collections.abc import Iterator
from copy import deepcopy
from datetime import UTC
from datetime import datetime as datetime_
from typing import Any

import requests
from dateutil.relativedelta import relativedelta
from dateutil.tz import tzutc
from requests import Request, Session
from requests.adapters import HTTPAdapter
from urllib3 import Retry

logger = logging.getLogger(__name__)


# Datetime regex from pystac_client
DATETIME_REGEX = re.compile(
    r"^(?P<year>\d{4})(-(?P<month>\d{2})(-(?P<day>\d{2})"
    r"(?P<remainder>([Tt])\d{2}:\d{2}:\d{2}(\.\d+)?"
    r"(?P<tz_info>[Zz]|([-+])(\d{2}):(\d{2}))?)?)?)?$"
)


class APIError(Exception):
    """Exception raised when a STAC API returns an error response.

    Adapted from pystac_client.exceptions.APIError
    """

    def __init__(self, msg: str):
        """Initialize an APIError.

        Args:
            msg: The error message.
        """
        super().__init__(msg)

    @staticmethod
    def from_response(response: requests.Response) -> "APIError":
        """Create an APIError from a requests Response object."""
        try:
            error_data = response.json()
            if "description" in error_data:
                msg = error_data["description"]
            elif "message" in error_data:
                msg = error_data["message"]
            else:
                msg = json.dumps(error_data)
        except Exception:
            msg = response.text

        return APIError(f"HTTP {response.status_code} error from {response.url}: {msg}")


class StacApiClient:
    """Lightweight STAC API client without object caching.

    This client makes HTTP requests to a STAC API endpoint and returns raw JSON
    dictionaries instead of deserializing to pystac objects. This avoids the
    unbounded memory growth from pystac's resolved object cache.

    Args:
        endpoint: The STAC API search endpoint URL
        timeout: Timeout in seconds for HTTP requests
        max_retries: Maximum number of retries for failed requests
    """

    def __init__(
        self,
        endpoint: str,
        timeout: float = 10.0,
        max_retries: int = 5,
    ):
        """Initialize a StacApiClient.

        Args:
            endpoint: The STAC API search endpoint URL
            timeout: Timeout in seconds for HTTP requests
            max_retries: Maximum number of retries for failed requests
        """
        self.endpoint = endpoint
        self.timeout = timeout

        # Setup session with retry logic (adapted from StacApiIO)
        self.session = Session()
        if max_retries:
            retry = Retry(
                total=max_retries,
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)

    def search(
        self,
        collections: list[str] | None = None,
        ids: list[str] | None = None,
        bbox: tuple[float, ...] | None = None,
        intersects: dict[str, Any] | str | None = None,
        datetime: tuple[datetime_, datetime_] | datetime_ | str | None = None,
        query: dict[str, Any] | None = None,
        limit: int | None = None,
        max_items: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Search the STAC API and yield item dictionaries.

        Args:
            collections: List of collection IDs to search
            ids: List of item IDs to search for
            bbox: Bounding box as (minx, miny, maxx, maxy)
            intersects: GeoJSON geometry dict or string
            datetime: Datetime or datetime range
            query: STAC query extension parameters
            limit: Items per page (recommendation to server)
            max_items: Maximum total items to return across all pages

        Yields:
            dict: STAC Item as a dictionary
        """
        # Build request parameters
        params: dict[str, Any] = {}

        if collections is not None:
            params["collections"] = collections

        if ids is not None:
            params["ids"] = ids

        if bbox is not None:
            params["bbox"] = list(bbox)

        if intersects is not None:
            params["intersects"] = self._format_intersects(intersects)

        if datetime is not None:
            params["datetime"] = self._format_datetime(datetime)

        if query is not None:
            params["query"] = query

        if limit is not None:
            params["limit"] = limit

        # Iterate through pages and yield items
        num_items = 0
        for page in self._get_pages(self.endpoint, "POST", params):
            for item in page.get("features", []):
                yield item
                num_items += 1
                if max_items is not None and num_items >= max_items:
                    return

    def get_item(
        self, item_id: str, collection_id: str | None = None
    ) -> dict[str, Any] | None:
        """Get a single item by ID.

        Args:
            item_id: The item ID
            collection_id: Optional collection ID to narrow search

        Returns:
            The item dictionary, or None if not found
        """
        for item in self.search(
            ids=[item_id],
            collections=[collection_id] if collection_id else None,
        ):
            return item
        return None

    def _request(
        self,
        url: str,
        method: str = "GET",
        parameters: dict[str, Any] | None = None,
    ) -> str:
        """Make an HTTP request to the STAC API.

        Adapted from StacApiIO.request()

        Args:
            url: The URL to request
            method: HTTP method (GET or POST)
            parameters: Query parameters (GET) or body (POST)

        Returns:
            Response text

        Raises:
            APIError: If the request fails
        """
        if method == "POST":
            request = Request(method=method, url=url, json=parameters)
        else:
            params = deepcopy(parameters) or {}
            request = Request(method="GET", url=url, params=params)

        try:
            prepped = self.session.prepare_request(request)
            msg = f"{prepped.method} {prepped.url}"
            if method == "POST" and parameters:
                msg += f" Body: {json.dumps(parameters)}"
            logger.debug(msg)

            send_kwargs = self.session.merge_environment_settings(
                prepped.url, proxies={}, stream=None, verify=True, cert=None
            )
            resp = self.session.send(prepped, timeout=self.timeout, **send_kwargs)
        except Exception as err:
            logger.debug(err)
            raise APIError(str(err))

        if resp.status_code != 200:
            raise APIError.from_response(resp)

        try:
            return resp.content.decode("utf-8")
        except Exception as err:
            raise APIError(str(err))

    def _get_pages(
        self,
        url: str,
        method: str,
        parameters: dict[str, Any] | None,
    ) -> Iterator[dict[str, Any]]:
        """Iterator that yields dictionaries for each page.

        Adapted from StacApiIO.get_pages()

        Args:
            url: The endpoint URL
            method: HTTP method
            parameters: Request parameters

        Yields:
            dict: Page of results (FeatureCollection-like dict)
        """
        # Get first page
        page_text = self._request(url, method=method, parameters=parameters)
        page = json.loads(page_text)

        if not page.get("features"):
            return

        yield page

        # Follow next links
        next_link = self._find_next_link(page)
        while next_link:
            # Handle different link formats
            link_href = next_link.get("href")
            if not link_href:
                break

            link_method = next_link.get("method", "GET")
            link_body = next_link.get("body")

            if link_method == "POST":
                # POST pagination with body
                page_text = self._request(
                    link_href,
                    method="POST",
                    parameters=link_body or parameters,
                )
            else:
                # GET pagination
                page_text = self._request(link_href, method="GET")

            page = json.loads(page_text)

            if not page.get("features"):
                return

            yield page

            next_link = self._find_next_link(page)

    @staticmethod
    def _find_next_link(page: dict[str, Any]) -> dict[str, Any] | None:
        """Find the 'next' link in a page response."""
        return next(
            (link for link in page.get("links", []) if link.get("rel") == "next"),
            None,
        )

    @staticmethod
    def _format_intersects(value: dict[str, Any] | str) -> dict[str, Any]:
        """Format intersects parameter.

        Adapted from ItemSearch._format_intersects()

        Args:
            value: GeoJSON geometry dict or string

        Returns:
            GeoJSON geometry dict
        """
        if isinstance(value, dict):
            # If it's a Feature, extract geometry
            if value.get("type") == "Feature":
                return deepcopy(value.get("geometry", {}))
            else:
                return deepcopy(value)
        elif isinstance(value, str):
            return dict(json.loads(value))
        elif hasattr(value, "__geo_interface__"):
            return dict(deepcopy(getattr(value, "__geo_interface__")))
        else:
            raise ValueError(
                "intersects must be a dict, str, or implement __geo_interface__"
            )

    @staticmethod
    def _to_utc_isoformat(dt: datetime_) -> str:
        """Convert datetime to UTC ISO format.

        Adapted from ItemSearch._to_utc_isoformat()
        """
        if dt.tzinfo is not None:
            dt = dt.astimezone(UTC)
        dt = dt.replace(tzinfo=None)
        return f"{dt.isoformat('T')}Z"

    def _to_isoformat_range(
        self,
        component: datetime_ | str | None,
    ) -> tuple[str, str | None]:
        """Convert a datetime component to an ISO format range.

        Adapted from ItemSearch._to_isoformat_range()

        This handles expansion of partial dates (e.g., "2023" -> full year range)
        and returns a tuple of (start, end) where end may be None for exact times.

        Args:
            component: A datetime, string timestamp, or None

        Returns:
            Tuple of (start_string, optional_end_string)
        """
        if component is None:
            return "..", None
        elif isinstance(component, str):
            if component == "..":
                return component, None
            elif component == "":
                return "..", None

            match = DATETIME_REGEX.match(component)
            if not match:
                raise ValueError(f"invalid datetime component: {component}")
            elif match.group("remainder"):
                # Full timestamp provided
                if match.group("tz_info"):
                    return component, None
                else:
                    return f"{component}Z", None
            else:
                # Partial date - expand to range
                year = int(match.group("year"))
                optional_month = match.group("month")
                optional_day = match.group("day")

            if optional_day is not None:
                # Day specified - expand to full day
                start = datetime_(
                    year,
                    int(optional_month),
                    int(optional_day),
                    0,
                    0,
                    0,
                    tzinfo=tzutc(),
                )
                end = start + relativedelta(days=1, seconds=-1)
            elif optional_month is not None:
                # Month specified - expand to full month
                start = datetime_(year, int(optional_month), 1, 0, 0, 0, tzinfo=tzutc())
                end = start + relativedelta(months=1, seconds=-1)
            else:
                # Year only - expand to full year
                start = datetime_(year, 1, 1, 0, 0, 0, tzinfo=tzutc())
                end = start + relativedelta(years=1, seconds=-1)

            return self._to_utc_isoformat(start), self._to_utc_isoformat(end)
        else:
            # datetime object
            return self._to_utc_isoformat(component), None

    def _format_datetime(
        self, value: datetime_ | str | tuple[datetime_ | str | None, ...]
    ) -> str:
        """Format datetime parameter for STAC API.

        Adapted from ItemSearch._format_datetime()

        Args:
            value: Single datetime, string, or tuple of (start, end)

        Returns:
            RFC3339 datetime string or range
        """
        components: list[Any]

        if isinstance(value, datetime_):
            return self._to_utc_isoformat(value)
        elif isinstance(value, str):
            components = value.split("/")
        else:
            # Assume tuple/list - convert to list for indexing
            components = list(value)

        if not components:
            raise ValueError("datetime cannot be empty")
        elif len(components) == 1:
            # Single datetime
            if components[0] is None:
                raise ValueError("cannot create a datetime query with None")
            start, end = self._to_isoformat_range(components[0])
            if end is not None:
                return f"{start}/{end}"
            else:
                return start
        elif len(components) == 2:
            # Datetime range
            if all(c is None for c in components):
                raise ValueError("cannot create a double open-ended interval")
            start, _ = self._to_isoformat_range(components[0])
            backup_end, end = self._to_isoformat_range(components[1])
            return f"{start}/{end or backup_end}"
        else:
            raise ValueError(
                f"too many datetime components (max=2, actual={len(components)}): {value}"
            )
