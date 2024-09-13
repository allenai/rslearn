"""Data source for Planet Labs API."""

import asyncio
import io
import json
import shutil
from collections.abc import Generator
from datetime import datetime
from typing import Any, BinaryIO

import planet
import rasterio
import requests
import shapely
from upath import UPath

from rslearn.config import LayerConfig, QueryConfig, RasterLayerConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import DataSource, Item
from rslearn.data_sources.utils import match_candidate_items_to_window
from rslearn.tile_stores import PrefixedTileStore, TileStore
from rslearn.utils import STGeometry

from .raster_source import get_needed_projections, ingest_raster


class Planet(DataSource):
    """A data source for Planet Labs API.

    The API key should be set via environment variable (PL_API_KEY).
    """

    def __init__(
        self,
        config: LayerConfig,
        item_type_id: str,
        product_bundle: str = "analytic_udm2",
        range_filters: dict[str, dict[str, Any]] = {},
        use_permission_filter: bool = True,
        sort_by: str | None = None,
    ):
        """Initialize a new Planet instance.

        Args:
            config: the LayerConfig of the layer containing this data source
            item_type_id: the item type ID, like "PSScene" or "SkySatCollect".
            product_bundle: the product bundle to download.
            range_filters: specifications for range filters to apply, such as
                {"cloud_cover": {"lte": 0.5}} to search for scenes with less than 50%
                cloud cover. It is map from the property name to a kwargs dict to apply
                when creating the range filter object.
            use_permission_filter: when querying the Planet Data API, use permission
                filter to only return scenes that we have access to.
            sort_by: name of attribute returned by Planet API to sort by like
                "-clear_percent" or "cloud_cover" (if it starts with minus sign then we
                sort descending.)
        """
        self.config = config
        self.item_type_id = item_type_id
        self.product_bundle = product_bundle
        self.range_filters = range_filters
        self.use_permission_filter = use_permission_filter
        self.sort_by = sort_by

    @staticmethod
    def from_config(config: LayerConfig, ds_path: UPath) -> "Planet":
        """Creates a new Planet instance from a configuration dictionary."""
        assert isinstance(config, RasterLayerConfig)
        d = config.data_source.config_dict
        kwargs = dict(
            config=config,
            item_type_id=d["item_type_id"],
        )
        if "range_filters" in d:
            kwargs["range_filters"] = d["range_filters"]
        return Planet(**kwargs)

    async def _search_items(self, geometry: STGeometry):
        wgs84_geometry = geometry.to_projection(WGS84_PROJECTION)
        geojson_data = json.loads(shapely.to_geojson(wgs84_geometry.shp))

        async with planet.Session() as session:
            client = session.client("data")

            filter_list = [
                planet.data_filter.date_range_filter(
                    "acquired", gte=geometry.time_range[0], lte=geometry.time_range[1]
                ),
                planet.data_filter.geometry_filter(geojson_data),
            ]
            if self.use_permission_filter:
                filter_list.append(planet.data_filter.permission_filter())
            for name, kwargs in self.range_filters.items():
                range_filter = planet.data_filter.range_filter(name, **kwargs)
                filter_list.append(range_filter)
            combined_filter = planet.data_filter.and_filter(filter_list)

            return [
                item
                async for item in client.search([self.item_type_id, combined_filter])
            ]

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[Item]]]:
        """Get a list of items in the data source intersecting the given geometries.

        Args:
            geometries: the spatiotemporal geometries
            query_config: the query configuration

        Returns:
            List of groups of items that should be retrieved for each geometry.
        """
        groups = []
        for geometry in geometries:
            planet_items = asyncio.run(self._search_items(geometry, query_config))

            if self.sort_by:
                if self.sort_by.startswith("-"):
                    multiplier = -1
                    sort_by = self.sort_by[1:]
                else:
                    multiplier = 1
                    sort_by = self.sort_by

                planet_items.sort(
                    key=lambda planet_item: multiplier * item["properties"][sort_by]
                )

            items = []
            for planet_item in planet_items:
                shp = shapely.geometry.shape(planet_item["geometry"])
                ts = datetime.fromisoformat(planet_item["properties"]["acquired"])
                item_geom = STGeometry(WGS84_PROJECTION, shp, (ts, ts))
                item = Item(planet_item["id"], item_geom)
                items.append(item)

            cur_groups = match_candidate_items_to_window(geometry, items, query_config)
            groups.append(cur_groups)

        return groups

    def get_item_by_name(self, name: str) -> Item:
        """Gets an item by name.

        Args:
            name: the item name.

        Returns:
            the item
        """
        raise NotImplementedError

    def deserialize_item(self, serialized_item: Any) -> Item:
        """Deserializes an item from JSON-decoded data."""
        assert isinstance(serialized_item, dict)
        return Item.deserialize(serialized_item)

    async def _wait_for_order(self, item: Item) -> None:
        """Make order and wait for download to be ready.

        Args:
            item: the item to order.
        """
        async with planet.Session() as session:
            product = planet.order_request.product(
                item_ids=[item.name],
                product_bundle=self.product_bundle,
                item_type=self.item_type_id,
            )
            request = planet.order_request.build_request(
                name=f"rslearn_order_{item.name}", products=[product]
            )
            client = session.client("orders")
            order = await client.create_order(request)
            await client.wait(order["id"])
            # await client.download_order(order["id"])
            return order

    def retrieve_item(self, item: Item) -> Generator[tuple[str, BinaryIO], None, None]:
        """Retrieves the rasters corresponding to an item as file streams.

        Args:
            item: the item to retrieve.

        Returns:
            generator that yields the item name along with binary buffer of GeoTIFF.
        """
        raise NotImplementedError

    def ingest(
        self,
        tile_store: TileStore,
        items: list[Item],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        Args:
            tile_store: the tile store to ingest into
            items: the items to ingest
            geometries: a list of geometries needed for each item
        """
        for item, cur_geometries in zip(items, geometries):
            download_urls = self._get_download_urls(item)
            for band in self.bands:
                band_names = [band]
                cur_tile_store = PrefixedTileStore(
                    tile_store, (item.name, "_".join(band_names))
                )
                needed_projections = get_needed_projections(
                    cur_tile_store, band_names, self.config.band_sets, cur_geometries
                )
                if not needed_projections:
                    continue

                buf = io.BytesIO()
                with requests.get(download_urls[band][1], stream=True) as r:
                    r.raise_for_status()
                    shutil.copyfileobj(r.raw, buf)
                buf.seek(0)
                with rasterio.open(buf) as raster:
                    for projection in needed_projections:
                        ingest_raster(
                            tile_store=cur_tile_store,
                            raster=raster,
                            projection=projection,
                            time_range=item.geometry.time_range,
                            layer_config=self.config,
                        )
