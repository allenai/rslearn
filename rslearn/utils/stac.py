"""STAC API client."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)

Bbox = tuple[float, float, float, float]


@dataclass(frozen=True)
class StacAsset:
    """A STAC asset."""

    href: str
    title: str | None
    type: str | None
    roles: list[str] | None


@dataclass(frozen=True)
class StacItem:
    """A STAC item."""

    id: str
    properties: dict[str, Any]
    collection: str | None
    bbox: Bbox | None
    geometry: dict[str, Any] | None
    assets: dict[str, StacAsset] | None
    time_range: tuple[datetime, datetime] | None

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "StacItem":
        """Create a STAC item from the item dict returned from API."""
        properties = item.get("properties", {})

        # Parse bbox.
        bbox: Bbox | None = None
        if "bbox" in item:
            if len(item["bbox"]) != 4:
                raise NotImplementedError(
                    f"got bbox with {len(item['bbox'])} coordinates but only 4 coordinates is implemented"
                )
            bbox = tuple(item["bbox"])

        # Parse assets.
        assets: dict[str, StacAsset] = {}
        for name, asset in item.get("assets", {}).items():
            assets[name] = StacAsset(
                href=asset["href"],
                title=asset.get("title"),
                type=asset.get("type"),
                roles=asset.get("roles"),
            )

        # Parse time range.
        time_range: tuple[datetime, datetime] | None = None
        if "start_datetime" in properties and "end_datetime" in properties:
            time_range = (
                datetime.fromisoformat(properties["start_datetime"]),
                datetime.fromisoformat(properties["end_datetime"]),
            )
        elif "datetime" in properties:
            ts = datetime.fromisoformat(properties["datetime"])
            time_range = (ts, ts)

        return cls(
            id=item["id"],
            properties=properties,
            collection=item.get("collection"),
            bbox=bbox,
            geometry=item.get("geometry"),
            assets=assets,
            time_range=time_range,
        )


class StacClient:
    """Limited functionality client for STAC APIs."""

    def __init__(self, endpoint: str):
        """Create a new StacClient.

        Args:
            endpoint: the STAC endpoint (base URL)
        """
        self.endpoint = endpoint
        self.session = requests.Session()

    def search(
        self,
        collections: list[str] | None = None,
        bbox: Bbox | None = None,
        intersects: dict[str, Any] | None = None,
        date_time: datetime | tuple[datetime, datetime] | None = None,
        ids: list[str] | None = None,
        limit: int | None = None,
        query: dict[str, Any] | None = None,
        sortby: list[dict[str, str]] | None = None,
    ) -> list[StacItem]:
        """Execute a STAC item search.

        We use the JSON POST API. Pagination is handled so the returned items are
        concatenated across all available pages.

        Args:
            collections: only search within the provided collection(s).
            bbox: only return features intersecting the provided bounding box.
            intersects: only return features intersecting this GeoJSON geometry.
            date_time: only return features that have a temporal property intersecting
                the provided time range or timestamp.
            ids: only return the provided item IDs.
            limit: number of items per page. We will read all the pages.
            query: query dict, if STAC query extension is supported by this API. See
                https://github.com/stac-api-extensions/query.
            sortby: list of sort specifications, e.g. [{"field": "id", "direction": "asc"}].

        Returns:
            list of matching STAC items.
        """
        # Build JSON request data.
        request_data: dict[str, Any] = {}
        if collections is not None:
            request_data["collections"] = collections
        if bbox is not None:
            request_data["bbox"] = bbox
        if intersects is not None:
            request_data["intersects"] = intersects
        if date_time is not None:
            if isinstance(date_time, tuple):
                start_time = date_time[0].isoformat().replace("+00:00", "Z")
                end_time = date_time[1].isoformat().replace("+00:00", "Z")
                request_data["datetime"] = f"{start_time}/{end_time}"
            else:
                request_data["datetime"] = date_time.isoformat().replace("+00:00", "Z")
        if ids is not None:
            request_data["ids"] = ids
        if limit is not None:
            request_data["limit"] = limit
        if query is not None:
            request_data["query"] = query
        if sortby is not None:
            request_data["sortby"] = sortby

        # Handle pagination.
        cur_url = self.endpoint + "/search"
        items: list[StacItem] = []
        while True:
            logger.debug("Reading STAC items from %s", cur_url)
            response = self.session.post(url=cur_url, json=request_data)
            response.raise_for_status()
            data = response.json()
            for item_dict in data["features"]:
                items.append(StacItem.from_dict(item_dict))

            next_link = None
            next_request_data: dict[str, Any] = {}
            for link in data.get("links", []):
                if "rel" not in link or link["rel"] != "next":
                    continue
                assert link["method"] == "POST"
                next_link = link["href"]
                next_request_data = link["body"]
                break

            if next_link is None:
                break

            cur_url = next_link
            request_data = next_request_data

        return items
