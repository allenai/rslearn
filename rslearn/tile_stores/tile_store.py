"""Base class for tile stores."""

from datetime import datetime
from typing import Any, Optional

from rasterio.crs import CRS


class LayerMetadata:
    """Stores metadata about a TileStoreLayer."""

    def __init__(
        self,
        crs: CRS,
        resolution: float,
        time_range: Optional[tuple[datetime, datetime]],
        properties: dict[str, Any],
    ) -> None:
        """Create a new LayerMetadata instance."""
        self.crs = crs
        self.resolution = resolution
        self.time_range = time_range
        self.properties = properties

    def serialize(self) -> dict:
        """Serializes the metadata to a JSON-encodable dictionary."""
        return {
            "crs": self.crs.to_string(),
            "resolution": self.resolution,
            "time_range": (
                [self.time_range[0].isoformat(), self.time_range[1].isoformat()]
                if self.time_range
                else None
            ),
            "properties": self.properties,
        }

    @staticmethod
    def deserialize(d: dict) -> "LayerMetadata":
        """Deserializes metadata from a JSON-decoded dictionary."""
        return LayerMetadata(
            crs=CRS.from_string(d["crs"]),
            resolution=d["resolution"],
            time_range=(
                (
                    datetime.fromisoformat(d["time_range"][0]),
                    datetime.fromisoformat(d["time_range"][1]),
                )
                if d["time_range"]
                else None
            ),
            properties=d["properties"],
        )


class TileStoreLayer:
    def get_tile(self, x: int, y: int) -> Any:
        """Get a tile from the store.

        Args:
            x: the x coordinate of the tile
            y: the y coordinate of the tile

        Returns:
            the tile data, typically either raster or vector content
        """
        raise NotImplementedError

    def save_tiles(self, data: list[tuple[int, int, Any]]) -> None:
        """Save tiles to the store.

        Args:
            data: a list of (x, y, data) tuples to save
        """
        raise NotImplementedError

    def get_metadata(self) -> LayerMetadata:
        """Get the LayerMetadata associated with this layer."""
        raise NotImplementedError

    def set_property(self, key: str, value: Any) -> None:
        """Set a property in the metadata for this layer.

        Args:
            key: the property key
            value: the property value
        """
        raise NotImplementedError


class TileStore:
    def create_layer(
        self, layer_id: tuple[str, ...], metadata: LayerMetadata
    ) -> TileStoreLayer:
        """Create a layer in the tile store (or get matching existing layer).

        Args:
            layer_id: the id of the layer to create
            metadata: metadata about the layer

        Returns:
            a TileStoreLayer corresponding to the new or pre-existing layer
        """
        raise NotImplementedError

    def get_layer(self, layer_id: tuple[str, ...]) -> Optional[TileStoreLayer]:
        """Get a layer in the tile store.

        Args:
            layer_id: the id of the layer to get

        Returns:
            the layer, or None if it does not exist yet.
        """
        raise NotImplementedError
