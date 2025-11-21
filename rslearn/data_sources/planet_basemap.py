"""Data source for Planet Labs Basemaps API."""

import os
import tempfile
from datetime import datetime
from typing import Any

import requests
import shapely
from upath import UPath

from rslearn.config import QueryConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import DataSource, DataSourceContext, Item
from rslearn.data_sources.utils import match_candidate_items_to_window
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils import STGeometry

logger = get_logger(__name__)


class PlanetItem(Item):
    """An item referencing a particular mosaic and quad in Basemaps API."""

    def __init__(self, name: str, geometry: STGeometry, mosaic_id: str, quad_id: str):
        """Create a new PlanetItem.

        Args:
            name: the item name (combination of mosaic and quad ID).
            geometry: the geometry associated with this quad.
            mosaic_id: the mosaic ID in API
            quad_id: the quad ID in API
        """
        super().__init__(name, geometry)
        self.mosaic_id = mosaic_id
        self.quad_id = quad_id

    def serialize(self) -> dict:
        """Serializes the item to a JSON-encodable dictionary."""
        d = super().serialize()
        d["mosaic_id"] = self.mosaic_id
        d["quad_id"] = self.quad_id
        return d

    @staticmethod
    def deserialize(d: dict) -> Item:
        """Deserializes an item from a JSON-decoded dictionary."""
        item = super(PlanetItem, PlanetItem).deserialize(d)
        return PlanetItem(
            name=item.name,
            geometry=item.geometry,
            mosaic_id=d["mosaic_id"],
            quad_id=d["quad_id"],
        )


class ApiError(Exception):
    """An error from Planet Labs API."""

    pass


class PlanetBasemap(DataSource):
    """A data source for Planet Labs Basemaps API."""

    api_url = "https://api.planet.com/basemaps/v1/"

    def __init__(
        self,
        series_id: str,
        bands: list[str],
        api_key: str | None = None,
        context: DataSourceContext = DataSourceContext(),
    ):
        """Initialize a new Planet instance.

        Args:
            series_id: the series of mosaics to use.
            bands: list of band names to use.
            api_key: optional Planet API key (it can also be provided via PL_API_KEY
                environmnet variable).
            context: the data source context
        """
        self.series_id = series_id
        self.bands = bands

        self.session = requests.Session()
        if api_key is None:
            api_key = os.environ["PL_API_KEY"]
        self.session.auth = (api_key, "")

        # Lazily load mosaics.
        self.mosaics: dict | None = None

    def _load_mosaics(self) -> dict[str, STGeometry]:
        """Lazily load mosaics in the configured series_id from Planet API.

        We don't load it when creating the data source because it takes time and caller
        may not be calling get_items. Additionally, loading it during the get_items
        call enables leveraging the retry loop functionality in
        prepare_dataset_windows.
        """
        if self.mosaics is not None:
            return self.mosaics

        self.mosaics = {}
        for mosaic_dict in self._api_get_paginate(
            path=f"series/{self.series_id}/mosaics", list_key="mosaics"
        ):
            shp = shapely.box(*mosaic_dict["bbox"])
            time_range = (
                datetime.fromisoformat(mosaic_dict["first_acquired"]),
                datetime.fromisoformat(mosaic_dict["last_acquired"]),
            )
            geom = STGeometry(WGS84_PROJECTION, shp, time_range)
            self.mosaics[mosaic_dict["id"]] = geom

        return self.mosaics

    def _api_get(
        self,
        path: str | None = None,
        url: str | None = None,
        query_args: dict[str, str] | None = None,
    ) -> list[Any] | dict[str, Any]:
        """Perform a GET request on the API.

        Args:
            path: the path to GET, like "series".
            url: the full URL to GET. Only one of path or url should be set.
            query_args: optional params to include with the request.

        Returns:
            the JSON response data.

        Raises:
            ApiError: if the API returned an error response.
        """
        if path is None and url is None:
            raise ValueError("Only one of path or url should be set")
        if query_args:
            kwargs = dict(params=query_args)
        else:
            kwargs = {}

        if path:
            url = self.api_url + path
        if url is None:
            raise ValueError("url is required")
        response = self.session.get(url, **kwargs)  # type: ignore

        if response.status_code != 200:
            raise ApiError(
                f"{url}: got status code {response.status_code}: {response.text}"
            )

        return response.json()

    def _api_get_paginate(
        self, path: str, list_key: str, query_args: dict[str, str] | None = None
    ) -> list:
        """Get all items in a paginated response.

        Args:
            path: the path to GET.
            list_key: the key in the response containing the list that should be
                concatenated across all available pages.
            query_args: optional params to include with the requests.

        Returns:
            the concatenated list of items.

        Raises:
            ApiError if the API returned an error response.
        """
        next_url = self.api_url + path
        items = []
        while True:
            json_data = self._api_get(url=next_url, query_args=query_args)
            if not isinstance(json_data, dict):
                logger.warning(f"Expected dict, got {type(json_data)}")
                continue
            items += json_data[list_key]

            if "_next" in json_data["_links"]:
                next_url = json_data["_links"]["_next"]
            else:
                return items

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[PlanetItem]]]:
        """Get a list of items in the data source intersecting the given geometries.

        Args:
            geometries: the spatiotemporal geometries
            query_config: the query configuration

        Returns:
            List of groups of items that should be retrieved for each geometry.
        """
        mosaics = self._load_mosaics()

        groups = []
        for geometry in geometries:
            geom_bbox = geometry.to_projection(WGS84_PROJECTION).shp.bounds
            geom_bbox_str = ",".join([str(value) for value in geom_bbox])

            # Find the relevant mosaics that the geometry intersects.
            # For each relevant mosaic, identify the intersecting quads.
            items = []
            for mosaic_id, mosaic_geom in mosaics.items():
                if not geometry.intersects(mosaic_geom):
                    continue
                logger.info(f"found mosaic {mosaic_geom} for geom {geometry}")
                # List all quads that intersect the current geometry's
                # longitude/latitude bbox in this mosaic.
                for quad_dict in self._api_get_paginate(
                    path=f"mosaics/{mosaic_id}/quads",
                    list_key="items",
                    query_args={"bbox": geom_bbox_str},
                ):
                    logger.info(f"found quad {quad_dict}")
                    shp = shapely.box(*quad_dict["bbox"])
                    geom = STGeometry(WGS84_PROJECTION, shp, mosaic_geom.time_range)
                    quad_id = quad_dict["id"]
                    items.append(
                        PlanetItem(f"{mosaic_id}_{quad_id}", geom, mosaic_id, quad_id)
                    )
            logger.info(f"found {len(items)} items for geom {geometry}")
            cur_groups = match_candidate_items_to_window(geometry, items, query_config)
            groups.append(cur_groups)

        return groups

    def deserialize_item(self, serialized_item: Any) -> Item:
        """Deserializes an item from JSON-decoded data."""
        assert isinstance(serialized_item, dict)
        return PlanetItem.deserialize(serialized_item)

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[Item],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        Args:
            tile_store: the tile store to ingest into
            items: the items to ingest
            geometries: a list of geometries needed for each item
        """
        for item in items:
            if tile_store.is_raster_ready(item.name, self.bands):
                continue

            assert isinstance(item, PlanetItem)
            download_url = (
                self.api_url + f"mosaics/{item.mosaic_id}/quads/{item.quad_id}/full"
            )
            response = self.session.get(download_url, allow_redirects=True, stream=True)
            if response.status_code != 200:
                raise ApiError(
                    f"{download_url}: got status code {response.status_code}: {response.text}"
                )

            with tempfile.TemporaryDirectory() as tmp_dir:
                local_fname = os.path.join(tmp_dir, "temp.tif")
                with open(local_fname, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                tile_store.write_raster_file(item.name, self.bands, UPath(local_fname))
