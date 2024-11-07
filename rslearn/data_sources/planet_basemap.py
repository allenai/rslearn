"""Data source for Planet Labs Basemaps API."""

import os
import tempfile
from datetime import datetime
from typing import Any

import rasterio
import requests
import shapely
from upath import UPath

from rslearn.config import LayerConfig, QueryConfig, RasterLayerConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import DataSource, Item
from rslearn.data_sources.utils import match_candidate_items_to_window
from rslearn.log_utils import get_logger
from rslearn.tile_stores import PrefixedTileStore, TileStore
from rslearn.utils import STGeometry

from .raster_source import get_needed_projections, ingest_raster

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
        config: RasterLayerConfig,
        series_id: str,
        bands: list[str],
        api_key: str | None = None,
    ):
        """Initialize a new Planet instance.

        Args:
            config: the LayerConfig of the layer containing this data source
            series_id: the series of mosaics to use.
            bands: list of band names to use.
            api_key: optional Planet API key (it can also be provided via PL_API_KEY
                environmnet variable).
        """
        self.config = config
        self.bands = bands

        self.session = requests.Session()
        if api_key is None:
            api_key = os.environ["PL_API_KEY"]
        self.session.auth = (api_key, "")

        # List mosaics.
        self.mosaics = {}
        for mosaic_dict in self._api_get_paginate(
            path=f"series/{series_id}/mosaics", list_key="mosaics"
        ):
            shp = shapely.box(*mosaic_dict["bbox"])
            time_range = (
                datetime.fromisoformat(mosaic_dict["first_acquired"]),
                datetime.fromisoformat(mosaic_dict["last_acquired"]),
            )
            geom = STGeometry(WGS84_PROJECTION, shp, time_range)
            self.mosaics[mosaic_dict["id"]] = geom

    @staticmethod
    def from_config(config: LayerConfig, ds_path: UPath) -> "PlanetBasemap":
        """Creates a new PlanetBasemap instance from a configuration dictionary."""
        assert isinstance(config, RasterLayerConfig)
        if config.data_source is None:
            raise ValueError("data_source is required")
        d = config.data_source.config_dict
        kwargs = dict(
            config=config,
            series_id=d["series_id"],
            bands=d["bands"],
        )
        optional_keys = [
            "api_key",
        ]
        for optional_key in optional_keys:
            if optional_key in d:
                kwargs[optional_key] = d[optional_key]
        return PlanetBasemap(**kwargs)

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
        groups = []
        for geometry in geometries:
            geom_bbox = geometry.to_projection(WGS84_PROJECTION).shp.bounds
            geom_bbox_str = ",".join([str(value) for value in geom_bbox])

            # Find the relevant mosaics that the geometry intersects.
            # For each relevant mosaic, identify the intersecting quads.
            items = []
            for mosaic_id, mosaic_geom in self.mosaics.items():
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

                assert isinstance(item, PlanetItem)
                download_url = (
                    self.api_url + f"mosaics/{item.mosaic_id}/quads/{item.quad_id}/full"
                )
                response = self.session.get(
                    download_url, allow_redirects=True, stream=True
                )
                if response.status_code != 200:
                    # # temporary skip for now
                    # logger.error(
                    #     f"{download_url}: got status code {response.status_code}: {response.text}"
                    # )
                    # continue
                    raise ApiError(
                        f"{download_url}: got status code {response.status_code}: {response.text}"
                    )

                with tempfile.TemporaryDirectory() as tmp_dir:
                    local_fname = os.path.join(tmp_dir, "temp.tif")
                    with open(local_fname, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    with rasterio.open(local_fname) as raster:
                        for projection in needed_projections:
                            ingest_raster(
                                tile_store=cur_tile_store,
                                raster=raster,
                                projection=projection,
                                time_range=item.geometry.time_range,
                                layer_config=self.config,
                            )
