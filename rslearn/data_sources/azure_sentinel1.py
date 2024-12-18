"""Sentinel-1 on Planetary Computer."""

import os
import tempfile
from typing import Any

import planetary_computer
import pystac
import pystac_client
import requests
import shapely
from upath import UPath

from rslearn.config import QueryConfig, RasterLayerConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import DataSource, Item
from rslearn.data_sources.raster_source import is_raster_needed
from rslearn.data_sources.utils import match_candidate_items_to_window
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.geometry import STGeometry

logger = get_logger(__name__)


class Sentinel1(DataSource):
    """A data source for Sentinel-1 data on Microsoft Planetary Computer.

    This uses the radiometrically corrected data.

    See https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc.

    The PC_SDK_SUBSCRIPTION_KEY environment variable can be set but is not needed.
    """

    STAC_ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1"

    COLLECTION_NAME = "sentinel-1-rtc"

    def __init__(
        self,
        config: RasterLayerConfig,
        query: dict[str, Any] | None = None,
        sort_by: str | None = None,
        sort_ascending: bool = True,
        timeout: int = 10,
    ):
        """Initialize a new Sentinel1 instance.

        Args:
            config: the LayerConfig of the layer containing this data source.
            query: optional query argument to STAC searches.
            sort_by: sort by this property in the STAC items.
            sort_ascending: whether to sort ascending (or descending).
            timeout: timeout for API requests in seconds.
        """
        self.config = config
        self.query = query
        self.sort_by = sort_by
        self.sort_ascending = sort_ascending
        self.timeout = timeout

        self.client = pystac_client.Client.open(
            self.STAC_ENDPOINT, modifier=planetary_computer.sign_inplace
        )
        self.collection = self.client.get_collection(self.COLLECTION_NAME)

    @staticmethod
    def from_config(config: RasterLayerConfig, ds_path: UPath) -> "Sentinel1":
        """Creates a new  Sentinel1instance from a configuration dictionary."""
        if config.data_source is None:
            raise ValueError("config.data_source is required")
        d = config.data_source.config_dict
        kwargs: dict[str, Any] = dict(
            config=config,
        )

        simple_optionals = ["query", "sort_by", "sort_ascending", "timeout"]
        for k in simple_optionals:
            if k in d:
                kwargs[k] = d[k]

        return Sentinel1(**kwargs)

    def _stac_item_to_item(self, stac_item: pystac.Item) -> Item:
        shp = shapely.geometry.shape(stac_item.geometry)

        # Get time range.
        metadata = stac_item.common_metadata
        if metadata.start_datetime is not None and metadata.end_datetime is not None:
            time_range = (
                metadata.start_datetime,
                metadata.end_datetime,
            )
        elif stac_item.datetime is not None:
            time_range = (stac_item.datetime, stac_item.datetime)
        else:
            raise ValueError(
                f"item {stac_item.id} unexpectedly missing start_datetime, end_datetime, and datetime"
            )

        geom = STGeometry(WGS84_PROJECTION, shp, time_range)
        return Item(stac_item.id, geom)

    def get_item_by_name(self, name: str) -> Item:
        """Gets an item by name.

        Args:
            name: the name of the item to get

        Returns:
            the item object
        """
        stac_item = self.collection.get_item(name)
        return self._stac_item_to_item(stac_item)

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
            # Get potentially relevant items from the collection by performing one search
            # for each requested geometry.
            wgs84_geometry = geometry.to_projection(WGS84_PROJECTION)
            logger.debug("performing STAC search for geometry %s", wgs84_geometry)
            result = self.client.search(
                collections=[self.COLLECTION_NAME],
                intersects=shapely.to_geojson(wgs84_geometry.shp),
                datetime=wgs84_geometry.time_range,
                query=self.query,
            )
            stac_items = [item for item in result.item_collection()]
            logger.debug("STAC search yielded %d items", len(stac_items))

            if self.sort_by is not None:
                stac_items.sort(
                    key=lambda stac_item: stac_item.properties[self.sort_by],
                    reverse=not self.sort_ascending,
                )

            candidate_items = [
                self._stac_item_to_item(stac_item) for stac_item in stac_items
            ]
            cur_groups = match_candidate_items_to_window(
                geometry, candidate_items, query_config
            )
            groups.append(cur_groups)

        return groups

    def deserialize_item(self, serialized_item: Any) -> Item:
        """Deserializes an item from JSON-decoded data."""
        assert isinstance(serialized_item, dict)
        return Item.deserialize(serialized_item)

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
            stac_item = self.collection.get_item(item.name)

            for band_name, asset in stac_item.assets.items():
                if not is_raster_needed([band_name], self.config.band_sets):
                    continue
                if tile_store.is_raster_ready(item.name, [band_name]):
                    continue

                asset_url = asset.href
                with tempfile.TemporaryDirectory() as tmp_dir:
                    local_fname = os.path.join(tmp_dir, "geotiff.tif")
                    logger.debug(
                        "azure_sentinel1 download item %s asset %s to %s",
                        item.name,
                        band_name,
                        local_fname,
                    )
                    with requests.get(
                        asset_url, stream=True, timeout=self.timeout
                    ) as r:
                        r.raise_for_status()
                        with open(local_fname, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)

                    logger.debug(
                        "azure_sentinel1 ingest item %s asset %s", item.name, band_name
                    )
                    tile_store.write_raster_file(
                        item.name, [band_name], UPath(local_fname)
                    )
                    logger.debug(
                        "azure_sentinel1 done ingesting item %s asset %s",
                        item.name,
                        band_name,
                    )
