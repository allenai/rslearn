"""Base classes for rslearn data sources."""

from collections.abc import Generator
from typing import Any, BinaryIO

from rslearn.config import LayerConfig, QueryConfig
from rslearn.dataset import Window
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

    def __eq__(self, other: Any) -> bool:
        """Check equality.

        Args:
            other: the other Item

        Returns:
            whether this Item is the same as the other Item.
        """
        return isinstance(other, Item) and self.name == other.name

    def __hash__(self) -> int:
        """Returns a hash of this item."""
        return hash(self.name)


class DataSource:
    """A set of raster or vector files that can be retrieved.

    Data sources should support at least one of ingest and materialize.
    """

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

    def materialize(
        self,
        window: Window,
        item_groups: list[list[Item]],
        layer_name: str,
        layer_cfg: LayerConfig,
    ) -> None:
        """Materialize data for the window.

        Args:
            window: the window to materialize
            item_groups: the items from get_items
            layer_name: the name of this layer
            layer_cfg: the config of this layer
        """
        raise NotImplementedError


class ItemLookupDataSource(DataSource):
    """A data source that can look up items by name."""

    def get_item_by_name(self, name: str) -> Item:
        """Gets an item by name."""
        raise NotImplementedError


class RetrieveItemDataSource(DataSource):
    """A data source that can retrieve items in their raw format."""

    def retrieve_item(self, item: Item) -> Generator[tuple[str, BinaryIO], None, None]:
        """Retrieves the rasters corresponding to an item as file streams."""
        raise NotImplementedError
