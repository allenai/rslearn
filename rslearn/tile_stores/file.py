import json
import os
from typing import Any, Optional

import numpy.typing as npt

from rslearn.config import RasterFormatConfig, TileStoreConfig
from rslearn.const import TILE_SIZE
from rslearn.utils import Projection
from rslearn.utils.raster_format import (
    GeotiffRasterFormat,
    RasterFormat,
    load_raster_format,
)

from .tile_store import LayerMetadata, TileStore, TileStoreLayer


class FileTileStoreLayer(TileStoreLayer):
    def __init__(
        self,
        root_dir: str,
        projection: Optional[Projection] = None,
        default_raster_format: RasterFormat = GeotiffRasterFormat(),
    ):
        """Creates a new FileTileStoreLayer.

        The root directory is a subfolder of the FileTileStore's root directory.

        Args:
            root_dir: root directory for this layer
        """
        self.root_dir = root_dir
        self.default_raster_format = default_raster_format
        self.projection = projection

        if not self.projection:
            self.projection = self.get_metadata().projection

    def get_raster(
        self, x: int, y: int, format: Optional[RasterFormat] = None
    ) -> Optional[npt.NDArray[Any]]:
        """Get a raster tile from the store.

        Args:
            x: the x coordinate of the tile
            y: the y coordinate of the tile
            format: the raster format to use

        Returns:
            the raster data
        """
        if format is None:
            format = self.default_raster_format
        extension = format.get_extension()
        fname = f"{self.root_dir}/{x}_{y}.{extension}"
        if not os.path.exists(fname):
            return None
        with open(fname, "rb") as f:
            return format.decode_raster(f)

    def save_rasters(
        self,
        data: list[tuple[int, int, npt.NDArray[Any]]],
        format: Optional[RasterFormat] = None,
    ) -> None:
        """Save tiles to the store.

        Args:
            data: a list of (x, y, image) tuples to save
            format: the raster format to use
        """
        if format is None:
            format = self.default_raster_format
        extension = format.get_extension()
        for x, y, image in data:
            bounds = (
                x * TILE_SIZE,
                y * TILE_SIZE,
                (x + 1) * TILE_SIZE,
                (y + 1) * TILE_SIZE,
            )
            with open(f"{self.root_dir}/{x}_{y}.{extension}", "wb") as f:
                format.encode_raster(f, self.projection, bounds, image)

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
    def __init__(
        self, root_dir, default_raster_format: RasterFormat = GeotiffRasterFormat()
    ):
        self.root_dir = root_dir
        self.default_raster_format = default_raster_format

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
        layer = FileTileStoreLayer(
            layer_dir,
            projection=metadata.projection,
            default_raster_format=self.default_raster_format,
        )
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

    @staticmethod
    def from_config(config: TileStoreConfig) -> "FileTileStore":
        d = config.config_dict
        if "default_raster_format" in d:
            default_raster_format = load_raster_format(
                RasterFormatConfig.from_config(d["default_raster_format"])
            )
        else:
            default_raster_format = GeotiffRasterFormat()
        return FileTileStore(
            root_dir=d["root_dir"],
            default_raster_format=default_raster_format,
        )
