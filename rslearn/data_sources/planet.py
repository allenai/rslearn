"""Data source for Planet Labs API."""

import asyncio
import json
import pathlib
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import planet
import rasterio
import shapely
from fsspec.implementations.local import LocalFileSystem
from upath import UPath

from rslearn.config import QueryConfig, RasterLayerConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import DataSource, Item
from rslearn.data_sources.utils import match_candidate_items_to_window
from rslearn.tile_stores import PrefixedTileStore, TileStore
from rslearn.utils import STGeometry
from rslearn.utils.fsspec import join_upath

from .raster_source import get_needed_projections, ingest_raster


class Planet(DataSource):
    """A data source for Planet Labs API.

    The API key should be set via environment variable (PL_API_KEY).
    """

    def __init__(
        self,
        config: RasterLayerConfig,
        item_type_id: str,
        cache_dir: UPath | None = None,
        asset_type_id: str = "ortho_analytic_sr",
        range_filters: dict[str, dict[str, Any]] = {},
        use_permission_filter: bool = True,
        sort_by: str | None = None,
        bands: list[str] = ["b01", "b02", "b03", "b04"],
    ):
        """Initialize a new Planet instance.

        Args:
            config: the LayerConfig of the layer containing this data source
            item_type_id: the item type ID, like "PSScene" or "SkySatCollect".
            cache_dir: where to store downloaded assets, or None to just store it in
                temporary directory before putting into tile store.
            asset_type_id: the asset type ID, see e.g.
                https://developers.planet.com/docs/data/skysatcollect/
                for a list of SkySatCollect assets.
            range_filters: specifications for range filters to apply, such as
                {"cloud_cover": {"lte": 0.5}} to search for scenes with less than 50%
                cloud cover. It is map from the property name to a kwargs dict to apply
                when creating the range filter object.
            use_permission_filter: when querying the Planet Data API, use permission
                filter to only return scenes that we have access to.
            sort_by: name of attribute returned by Planet API to sort by like
                "-clear_percent" or "cloud_cover" (if it starts with minus sign then we
                sort descending.)
            bands: what to call the bands in the asset.
        """
        self.config = config
        self.item_type_id = item_type_id
        self.cache_dir = cache_dir
        self.asset_type_id = asset_type_id
        self.range_filters = range_filters
        self.use_permission_filter = use_permission_filter
        self.sort_by = sort_by
        self.bands = bands

    @staticmethod
    def from_config(config: RasterLayerConfig, ds_path: UPath) -> "Planet":
        """Creates a new Planet instance from a configuration dictionary."""
        if config.data_source is None:
            raise ValueError("data_source is required")
        d = config.data_source.config_dict
        kwargs = dict(
            config=config,
            item_type_id=d["item_type_id"],
        )
        optional_keys = [
            "asset_type_id",
            "range_filters",
            "use_permission_filter",
            "sort_by",
            "bands",
        ]
        for optional_key in optional_keys:
            if optional_key in d:
                kwargs[optional_key] = d[optional_key]
        if "cache_dir" in d:
            kwargs["cache_dir"] = join_upath(ds_path, d["cache_dir"])
        return Planet(**kwargs)

    async def _search_items(self, geometry: STGeometry) -> list[dict[str, Any]]:
        wgs84_geometry = geometry.to_projection(WGS84_PROJECTION)
        geojson_data = json.loads(shapely.to_geojson(wgs84_geometry.shp))

        async with planet.Session() as session:
            client = session.client("data")
            gte = geometry.time_range[0] if geometry.time_range is not None else None
            lte = geometry.time_range[1] if geometry.time_range is not None else None
            filter_list = [
                planet.data_filter.date_range_filter("acquired", gte=gte, lte=lte),
                planet.data_filter.geometry_filter(geojson_data),
                planet.data_filter.asset_filter([self.asset_type_id]),
            ]
            if self.use_permission_filter:
                filter_list.append(planet.data_filter.permission_filter())
            for name, kwargs in self.range_filters.items():
                range_filter = planet.data_filter.range_filter(name, **kwargs)
                filter_list.append(range_filter)
            combined_filter = planet.data_filter.and_filter(filter_list)

            return [
                item
                async for item in client.search(
                    [self.item_type_id], search_filter=combined_filter
                )
            ]

    def _wrap_planet_item(self, planet_item: dict[str, Any]) -> Item:
        """Convert a decoded Planet API item into an Item object."""
        shp = shapely.geometry.shape(planet_item["geometry"])
        ts = datetime.fromisoformat(planet_item["properties"]["acquired"])
        item_geom = STGeometry(WGS84_PROJECTION, shp, (ts, ts))
        return Item(planet_item["id"], item_geom)

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
            planet_items = asyncio.run(self._search_items(geometry))

            if self.sort_by:
                if self.sort_by.startswith("-"):
                    multiplier = -1
                    sort_by = self.sort_by[1:]
                else:
                    multiplier = 1
                    sort_by = self.sort_by

                planet_items.sort(
                    key=lambda planet_item: multiplier
                    * planet_item["properties"][sort_by]
                )

            items = []
            for planet_item in planet_items:
                items.append(self._wrap_planet_item(planet_item))

            cur_groups = match_candidate_items_to_window(geometry, items, query_config)
            groups.append(cur_groups)

        return groups

    async def _get_item_by_name(self, name: str) -> dict[str, Any]:
        async with planet.Session() as session:
            client = session.client("data")
            filter = planet.data_filter.string_in_filter("id", [name])
            results = [
                item
                async for item in client.search(
                    [self.item_type_id], search_filter=filter
                )
            ]
            assert len(results) == 1
            return results[0]

    def get_item_by_name(self, name: str) -> Item:
        """Gets an item by name.

        Args:
            name: the item name.

        Returns:
            the item
        """
        planet_item = asyncio.run(self._get_item_by_name(name))
        return self._wrap_planet_item(planet_item)

    def deserialize_item(self, serialized_item: Any) -> Item:
        """Deserializes an item from JSON-decoded data."""
        assert isinstance(serialized_item, dict)
        return Item.deserialize(serialized_item)

    async def _download_asset(self, item: Item, tmp_dir: pathlib.Path) -> UPath:
        """Activate asset and download it.

        Args:
            item: the item to download.
            tmp_dir: temporary directory to download the asset to in case cache_dir is
                either not set or remote.

        Returns:
            the path where the asset is downloaded.
        """
        async with planet.Session() as session:
            client = session.client("data")
            assets = await client.list_item_assets(self.item_type_id, item.name)
            asset = assets[self.asset_type_id]
            await client.activate_asset(asset)
            # Wait up to two hours for asset to be ready.
            await client.wait_asset(asset, max_attempts=1600, delay=5)

            # Need to refresh the asset so it has location attribute.
            asset = await client.get_asset(
                self.item_type_id, item.name, self.asset_type_id
            )

            if self.cache_dir is None:
                output_path = await client.download_asset(asset, directory=tmp_dir)
                return UPath(output_path)

            elif isinstance(self.cache_dir.fs, LocalFileSystem):
                output_path = await client.download_asset(
                    asset, directory=self.cache_dir.path
                )
                return UPath(output_path)

            else:
                output_path = await client.download_asset(asset, directory=tmp_dir)
                wanted_path = self.cache_dir / output_path.name
                with open(output_path, "rb") as src:
                    with wanted_path.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                return wanted_path

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
            with tempfile.TemporaryDirectory() as tmp_dir:
                band_names = self.bands
                cur_tile_store = PrefixedTileStore(
                    tile_store, (item.name, "_".join(band_names))
                )
                needed_projections = get_needed_projections(
                    cur_tile_store, band_names, self.config.band_sets, cur_geometries
                )
                if not needed_projections:
                    continue

                asset_path = asyncio.run(self._download_asset(item, Path(tmp_dir)))
                with asset_path.open("rb") as f:
                    with rasterio.open(f) as raster:
                        for projection in needed_projections:
                            ingest_raster(
                                tile_store=cur_tile_store,
                                raster=raster,
                                projection=projection,
                                time_range=item.geometry.time_range,
                                layer_config=self.config,
                            )
