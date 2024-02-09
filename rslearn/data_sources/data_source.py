"""Base classes for rslearn data sources."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

import shapely
from shapely import Geometry

from rslearn.tile_stores import TileStore
from rslearn.utils import STGeometry


class Item:
    """An item in a data source.

    Items correspond to distinct objects in the data source, such as a raster file
    (e.g., Sentinel-2 scene) or a vector file (e.g., a single shapefile).
    """

    def __init__(self, name: str, shp: Geometry, time: Optional[datetime]):
        """Creates a new item.

        Args:
            name: unique name of the item
            shp: the geometry of the item
            time: the time of the item
        """
        self.name = name
        self.shp = shp
        self.time = time

    def serialize(self) -> dict:
        """Serializes the item to a JSON-encodable dictionary."""
        return {
            "name": self.name,
            "shp": self.shp.wkt,
            "time": self.time.isoformat() if self.time else None,
        }

    @staticmethod
    def deserialize(d: dict) -> "Item":
        """Deserializes an item from a JSON-decoded dictionary."""
        return Item(
            name=d["name"],
            shp=shapely.from_wkt(d["shp"]),
            time=datetime.fromisoformat(d["time"]) if d["time"] else None,
        )


class SpaceMode(Enum):
    CONTAINS = 1
    """Items must contain the entire window."""

    INTERSECTS = 2
    """Items must overlap any portion of the window."""

    MOSAIC = 3
    """Groups of items should be computed that cover the entire window.

    During materialization, items in each group are merged to form a mosaic in the
    dataset.
    """


class TimeMode(Enum):
    WITHIN = 1
    """Items must be within the window time range."""

    NEAREST = 2
    """Select items closest to the window time range, up to max_matches."""

    BEFORE = 3
    """Select items before the start of the window time range, up to max_matches."""

    AFTER = 4
    """Select items after the end of the window time range, up to max_matches."""


class QueryConfig:
    """A configuration for querying items in a data source."""

    def __init__(
        self,
        space_mode: SpaceMode = SpaceMode.MOSAIC,
        time_mode: TimeMode = TimeMode.WITHIN,
        max_matches: int = 1,
    ):
        """Creates a new query configuration.

        The provided options determine how a DataSource should lookup items that match a
        spatiotemporal window.

        Args:
            space_mode: specifies how items should be matched with windows spatially
            time_mode: specifies how items should be matched with windows temporally
            max_matches: the maximum number of items (or groups of items, if space_mode
                is MOSAIC) to match
        """
        self.space_mode = space_mode
        self.time_mode = time_mode
        self.max_matches = max_matches


class DataSource:
    """A set of raster or vector files that can be retrieved."""

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
        raise NotImplementedError()

    def deserialize_item(self, serialized_item: Any) -> Item:
        """Deserializes an item from JSON-decoded data."""
        raise NotImplementedError()

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
        raise NotImplementedError()
