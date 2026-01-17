"""A partial data source implementation providing get_items using a STAC API."""

import json
from datetime import datetime
from typing import Any

import shapely
from upath import UPath

from rslearn.config import QueryConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.data_source import Item, ItemLookupDataSource
from rslearn.data_sources.utils import match_candidate_items_to_window
from rslearn.log_utils import get_logger
from rslearn.utils.geometry import STGeometry
from rslearn.utils.stac import StacClient, StacItem

logger = get_logger(__name__)


class SourceItem(Item):
    """An item in the StacDataSource data source."""

    def __init__(
        self,
        name: str,
        geometry: STGeometry,
        asset_urls: dict[str, str],
        properties: dict[str, str],
    ):
        """Creates a new SourceItem.

        Args:
            name: unique name of the item
            geometry: the spatial and temporal extent of the item
            asset_urls: map from asset key to the unsigned asset URL.
            properties: properties requested by the data source implementation.
        """
        super().__init__(name, geometry)
        self.asset_urls = asset_urls
        self.properties = properties

    def serialize(self) -> dict[str, Any]:
        """Serializes the item to a JSON-encodable dictionary."""
        d = super().serialize()
        d["asset_urls"] = self.asset_urls
        d["properties"] = self.properties
        return d

    @staticmethod
    def deserialize(d: dict[str, Any]) -> "SourceItem":
        """Deserializes an item from a JSON-decoded dictionary."""
        item = super(SourceItem, SourceItem).deserialize(d)
        return SourceItem(
            name=item.name,
            geometry=item.geometry,
            asset_urls=d["asset_urls"],
            properties=d["properties"],
        )


class StacDataSource(ItemLookupDataSource[SourceItem]):
    """A partial data source implementing get_items using a STAC API.

    This is a helper class that full implementations can extend to not have to worry
    about the get_items and get_item_by_name implementation.
    """

    def __init__(
        self,
        endpoint: str,
        collection_name: str,
        query: dict[str, Any] | None = None,
        sort_by: str | None = None,
        sort_ascending: bool = True,
        required_assets: list[str] | None = None,
        cache_dir: UPath | None = None,
        limit: int = 100,
        properties_to_record: list[str] = [],
    ):
        """Create a new StacDataSource.

        Args:
            endpoint: the STAC endpoint to use.
            collection_name: the STAC collection name.
            query: optional STAC query dict to include in searches, e.g. {"eo:cloud_cover": {"lt": 50}}.
            sort_by: sort results by this STAC property.
            sort_ascending: if sort_by is set, sort in ascending order (default).
                Otherwise sort in descending order.
            required_assets: if set, we ignore items that do not have all of these
                asset keys.
            cache_dir: optional cache directory to cache items. This is recommended if
                allowing direct materialization from the data source, since it will
                likely be necessary to make lots of get_item_by_name calls during
                materialization. TODO: give direct materialization access to the Item
                object.
            limit: limit to pass to search queries.
            properties_to_record: if these properties on the STAC item exist, they are
                are retained in the SourceItem when we initialize it.
        """
        self.client = StacClient(endpoint)
        self.collection_name = collection_name
        self.query = query
        self.sort_by = sort_by
        self.sort_ascending = sort_ascending
        self.required_assets = required_assets
        self.cache_dir = cache_dir
        self.limit = limit
        self.properties_to_record = properties_to_record

    def _stac_item_to_item(self, stac_item: StacItem) -> SourceItem:
        # Make sure geometry, time range, and assets are set.
        if stac_item.geometry is None:
            raise ValueError("got unexpected item with no geometry")
        if stac_item.time_range is None:
            raise ValueError("got unexpected item with no time range")
        if stac_item.assets is None:
            raise ValueError("got unexpected item with no assets")

        shp = shapely.geometry.shape(stac_item.geometry)
        geom = STGeometry(WGS84_PROJECTION, shp, stac_item.time_range)
        asset_urls = {
            asset_key: asset_obj.href
            for asset_key, asset_obj in stac_item.assets.items()
        }

        # Keep any properties requested by the data source implementation.
        properties = {}
        for prop_name in self.properties_to_record:
            if prop_name not in stac_item.properties:
                continue
            properties[prop_name] = stac_item.properties[prop_name]

        return SourceItem(stac_item.id, geom, asset_urls, properties)

    def _get_search_time_range(
        self, geometry: STGeometry
    ) -> datetime | tuple[datetime, datetime] | None:
        """Get time range to include in STAC API search.

        By default, we filter STAC searches to the window's time range. Subclasses can
        override this to disable time filtering for "static" datasets.

        Args:
            geometry: the geometry we are searching for.

        Returns:
            the time range (or timestamp) to pass to the STAC search, or None to avoid
                temporal filtering in the search request.
        """
        # Note: StacClient.search accepts either a datetime or a (start, end) tuple.
        return geometry.time_range

    def get_item_by_name(self, name: str) -> SourceItem:
        """Gets an item by name.

        Args:
            name: the name of the item to get

        Returns:
            the item object
        """
        # If cache_dir is set, we cache the item. First here we check if it is already
        # in the cache.
        cache_fname: UPath | None = None
        if self.cache_dir:
            cache_fname = self.cache_dir / f"{name}.json"
        if cache_fname is not None and cache_fname.exists():
            with cache_fname.open() as f:
                return SourceItem.deserialize(json.load(f))

        # No cache or not in cache, so we need to make the STAC request.
        logger.debug(f"Getting STAC item {name}")
        stac_items = self.client.search(ids=[name], collections=[self.collection_name])

        if len(stac_items) == 0:
            raise ValueError(
                f"Item {name} not found in collection {self.collection_name}"
            )
        if len(stac_items) > 1:
            raise ValueError(
                f"Multiple items found for ID {name} in collection {self.collection_name}"
            )

        stac_item = stac_items[0]
        item = self._stac_item_to_item(stac_item)

        # Finally we cache it if cache_dir is set.
        if cache_fname is not None:
            with cache_fname.open("w") as f:
                json.dump(item.serialize(), f)

        return item

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[SourceItem]]]:
        """Get a list of items in the data source intersecting the given geometries.

        Args:
            geometries: the spatiotemporal geometries
            query_config: the query configuration

        Returns:
            List of groups of items that should be retrieved for each geometry.
        """
        groups = []
        for geometry in geometries:
            # Get potentially relevant items from the collection by performing one search
            # for each requested geometry.
            wgs84_geometry = geometry.to_projection(WGS84_PROJECTION)
            logger.debug("performing STAC search for geometry %s", wgs84_geometry)
            search_time_range = self._get_search_time_range(wgs84_geometry)
            stac_items = self.client.search(
                collections=[self.collection_name],
                intersects=json.loads(shapely.to_geojson(wgs84_geometry.shp)),
                date_time=search_time_range,
                query=self.query,
                limit=self.limit,
            )
            logger.debug("STAC search yielded %d items", len(stac_items))

            if self.required_assets is not None:
                # Filter out items that are missing any of the assets in self.asset_bands.
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
                sort_by = self.sort_by
                stac_items.sort(
                    key=lambda stac_item: stac_item.properties[sort_by],
                    reverse=not self.sort_ascending,
                )

            candidate_items = [
                self._stac_item_to_item(stac_item) for stac_item in stac_items
            ]

            # Since we made the STAC request, might as well save these to the cache.
            if self.cache_dir is not None:
                for item in candidate_items:
                    cache_fname = self.cache_dir / f"{item.name}.json"
                    if cache_fname.exists():
                        continue
                    with cache_fname.open("w") as f:
                        json.dump(item.serialize(), f)

            cur_groups = match_candidate_items_to_window(
                geometry, candidate_items, query_config
            )
            groups.append(cur_groups)

        return groups

    def deserialize_item(self, serialized_item: Any) -> SourceItem:
        """Deserializes an item from JSON-decoded data."""
        assert isinstance(serialized_item, dict)
        return SourceItem.deserialize(serialized_item)
