import json
import os
from typing import Any, Optional

from .tile_store import LayerMetadata, TileStore, TileStoreLayer


class FileTileStoreLayer(TileStoreLayer):
    def __init__(self, root_dir):
        """Creates a new FileTileStoreLayer.

        The root directory is a subfolder of the FileTileStore's root directory.

        Args:
            root_dir: root directory for this layer
        """
        self.root_dir = root_dir

    def get_tile(self, x: int, y: int) -> Any:
        """Get a tile from the store.

        Args:
            x: the x coordinate of the tile
            y: the y coordinate of the tile

        Returns:
            the tile data, typically either raster or vector content
        """
        return open(f"{self.root_dir}/{x}_{y}.png", "rb")

    def save_tiles(self, data: list[tuple[int, int, Any]]) -> None:
        """Save tiles to the store.

        Args:
            data: a list of (x, y, data) tuples to save
        """
        for x, y, bytes in data:
            with open(f"{self.root_dir}/{x}_{y}.png", "wb") as f:
                f.write(bytes)

    def get_metadata(self) -> LayerMetadata:
        """Get the LayerMetadata associated with this layer."""
        with open(os.path.join(self.root_dir, "metadata.json"), "r") as f:
            return LayerMetadata.deserialize(json.load(f))

    def set_property(self, key: str, value: Any) -> None:
        """Set a property in the metadata for this layer.

        Args:
            key: the property key
            value: the property value
        """
        metadata = self.get_metadata()
        metadata.properties[key] = value
        self.save_metadata(metadata)

    def save_metadata(self, metadata: LayerMetadata) -> None:
        """Save the LayerMetadata associated with this layer."""
        with open(os.path.join(self.root_dir, "metadata.json"), "w") as f:
            json.dump(metadata.serialize(), f)


class FileTileStore(TileStore):
    def __init__(self, root_dir):
        self.root_dir = root_dir

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
        layer_dir = os.path.join(self.root_dir, "_".join(layer_id))
        layer = FileTileStoreLayer(layer_dir)
        if not os.path.exists(layer_dir):
            os.makedirs(layer_dir)
            layer.save_metadata(metadata)
        return layer

    def get_layer(self, layer_id: tuple[str, ...]) -> Optional[TileStoreLayer]:
        """Get a layer in the tile store.

        Args:
            layer_id: the id of the layer to get

        Returns:
            the layer, or None if it does not exist yet.
        """
        layer_dir = os.path.join(self.root_dir, "_".join(layer_id))
        if not os.path.exists(layer_dir):
            return None
        return FileTileStoreLayer(layer_dir)
