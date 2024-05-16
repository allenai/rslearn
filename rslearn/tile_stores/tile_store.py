"""Base class for tile stores."""

from datetime import datetime
from typing import Any, Optional

import numpy.typing as npt

from rslearn.utils import Feature, PixelBounds, Projection


class LayerMetadata:
    """Stores metadata about a TileStoreLayer."""

    def __init__(
        self,
        projection: Projection,
        time_range: Optional[tuple[datetime, datetime]],
        properties: dict[str, Any],
    ) -> None:
        """Create a new LayerMetadata instance."""
        self.projection = projection
        self.time_range = time_range
        self.properties = properties

    def serialize(self) -> dict:
        """Serializes the metadata to a JSON-encodable dictionary."""
        return {
            "projection": self.projection.serialize(),
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
            projection=Projection.deserialize(d["projection"]),
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
    """An abstract class for a layer in a tile store.

    The layer can store one or more raster and vector datas.
    """

    def read_raster(self, bounds: PixelBounds) -> Optional[npt.NDArray[Any]]:
        """Read raster data from the store.

        Args:
            bounds: the bounds within which to read

        Returns:
            the raster data
        """
        raise NotImplementedError

    def write_raster(
        self,
        bounds: PixelBounds,
        array: npt.NDArray[Any],
    ) -> None:
        """Write raster data to the store.

        Args:
            bounds: the bounds of the raster
            array: the raster data
        """
        raise NotImplementedError

    def read_vector(self, bounds: PixelBounds) -> list[Feature]:
        """Read vector data from the store.

        Args:
            bounds: the bounds within which to read

        Returns:
            the vector data
        """
        raise NotImplementedError

    def write_vector(self, data: list[Feature]) -> None:
        """Save vector tiles to the store.

        Args:
            data: the vector data
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
    """An abstract class for a tile store.

    A tile store supports operations to read and write raster and vector data.
    """

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

    def list_layers(self, prefix: tuple[str, ...] = tuple()) -> list[str]:
        """List options for next part of layer ID with the specified prefix.

        Args:
            prefix: the prefix to match

        Returns:
            available options for next part of the layer ID
        """
        raise NotImplementedError


class PrefixedTileStore(TileStore):
    """Wraps another tile store by adding prefix to all layer IDs."""

    def __init__(self, tile_store: TileStore, prefix: tuple[str, ...]):
        """Initialize a new PrefixedTileStore.

        Args:
            tile_store: the tile store to wrap
            prefix: the prefix to add to the layer IDs
        """
        self.tile_store = tile_store
        self.prefix = prefix

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
        return self.tile_store.create_layer(self.prefix + layer_id, metadata)

    def get_layer(self, layer_id: tuple[str, ...]) -> Optional[TileStoreLayer]:
        """Get a layer in the tile store.

        Args:
            layer_id: the id of the layer to get

        Returns:
            the layer, or None if it does not exist yet.
        """
        return self.tile_store.get_layer(self.prefix + layer_id)

    def list_layers(self, prefix: tuple[str, ...] = tuple()) -> list[str]:
        """List options for next part of layer ID with the specified prefix.

        Args:
            prefix: the prefix to match

        Returns:
            available options for next part of the layer ID
        """
        return self.tile_store.list_layers(self.prefix + prefix)
