"""Base classes for rslearn data sources."""

from typing import Any

from rslearn.config import QueryConfig
from rslearn.tile_stores import TileStore
from rslearn.utils import STGeometry


class Item:
    """An item in a data source.

    Items correspond to distinct objects in the data source, such as a raster file
    (e.g., Sentinel-2 scene) or a vector file (e.g., a single shapefile).
    """

    def __init__(self, name: str, geometry: STGeometry):
        """Creates a new item.

        Args:
            name: unique name of the item
            geometry: the spatial and temporal extent of the item
        """
        self.name = name
        self.geometry = geometry

    def serialize(self) -> dict:
        """Serializes the item to a JSON-encodable dictionary."""
        return {
            "name": self.name,
            "geometry": self.geometry.serialize(),
        }

    @staticmethod
    def deserialize(d: dict) -> "Item":
        """Deserializes an item from a JSON-decoded dictionary."""
        return Item(
            name=d["name"],
            geometry=STGeometry.deserialize(d["geometry"]),
        )


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
        raise NotImplementedError

    def deserialize_item(self, serialized_item: Any) -> Item:
        """Deserializes an item from JSON-decoded data."""
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
        raise NotImplementedError
