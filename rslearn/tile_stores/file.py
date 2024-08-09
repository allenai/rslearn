"""FileTileStore TileStore implementation."""

import json
from typing import Any

import numpy.typing as npt

from rslearn.config import RasterFormatConfig, TileStoreConfig, VectorFormatConfig
from rslearn.utils import Feature, FileAPI, PixelBounds, Projection
from rslearn.utils.raster_format import (
    GeotiffRasterFormat,
    RasterFormat,
    load_raster_format,
)
from rslearn.utils.vector_format import (
    GeojsonVectorFormat,
    VectorFormat,
    load_vector_format,
)

from .tile_store import LayerMetadata, TileStore, TileStoreLayer


class FileTileStoreLayer(TileStoreLayer):
    """A layer in a FileTileStore."""

    def __init__(
        self,
        file_api: FileAPI,
        projection: Projection | None = None,
        raster_format: RasterFormat = GeotiffRasterFormat(),
        vector_format: VectorFormat = GeojsonVectorFormat(),
    ):
        """Creates a new FileTileStoreLayer.

        The root directory is a subfolder of the FileTileStore's root directory.

        Args:
            file_api: FileAPI for this layer
            projection: the projection of this layer
            raster_format: the RasterFormat to use for reading/writing raster data
            vector_format: the VectorFormat to use for reading/writing vector data
        """
        self.file_api = file_api
        self.raster_format = raster_format
        self.vector_format = vector_format
        self.projection = projection

        if not self.projection:
            self.projection = self.get_metadata().projection

    def read_raster(self, bounds: PixelBounds) -> npt.NDArray[Any] | None:
        """Read raster data from the store.

        Args:
            bounds: the bounds within which to read

        Returns:
            the raster data
        """
        return self.raster_format.decode_raster(self.file_api, bounds)

    def write_raster(self, bounds: PixelBounds, array: npt.NDArray[Any]) -> None:
        """Write raster data to the store.

        Args:
            bounds: the bounds of the raster
            array: the raster data
        """
        self.raster_format.encode_raster(self.file_api, self.projection, bounds, array)

    def get_raster_bounds(self) -> PixelBounds:
        """Gets the bounds of the raster data in the store."""
        return self.raster_format.get_raster_bounds(self.file_api)

    def read_vector(self, bounds: PixelBounds) -> list[Feature]:
        """Read vector data from the store.

        Args:
            bounds: the bounds within which to read

        Returns:
            the vector data
        """
        return self.vector_format.decode_vector(self.file_api, bounds)

    def write_vector(self, data: list[Feature]) -> None:
        """Save vector tiles to the store.

        Args:
            data: the vector data
        """
        self.vector_format.encode_vector(self.file_api, self.projection, data)

    def get_metadata(self) -> LayerMetadata:
        """Get the LayerMetadata associated with this layer."""
        with self.file_api.open("metadata.json", "r") as f:
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
        with self.file_api.open_atomic("metadata.json", "w") as f:
            json.dump(metadata.serialize(), f)


class FileTileStore(TileStore):
    """A TileStore that stores data on the local filesystem."""

    def __init__(
        self,
        file_api: FileAPI,
        raster_format: RasterFormat = GeotiffRasterFormat(),
        vector_format: VectorFormat = GeojsonVectorFormat(),
    ):
        """Initialize a new FileTileStore.

        Args:
            file_api: the FileAPI to store data
            raster_format: the RasterFormat (defaults to Geotiff)
            vector_format: the VectorFormat (defaults to GeoJSON)
        """
        self.file_api = file_api
        self.raster_format = raster_format
        self.vector_format = vector_format

    def _get_layer_dir(self, layer_id: tuple[str, ...]) -> FileAPI:
        """Get the FileAPI of the specified layer ID."""
        for part in layer_id:
            if "/" in part or part.startswith("."):
                raise ValueError(f"Invalid layer_id part {part}")
        return self.file_api.get_folder(self.file_api.join(*layer_id))

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
        layer_dir = self._get_layer_dir(layer_id)
        layer = FileTileStoreLayer(
            layer_dir,
            projection=metadata.projection,
            raster_format=self.raster_format,
            vector_format=self.vector_format,
        )
        if not layer_dir.exists("metadata.json"):
            layer.save_metadata(metadata)
        return layer

    def get_layer(self, layer_id: tuple[str, ...]) -> TileStoreLayer | None:
        """Get a layer in the tile store.

        Args:
            layer_id: the id of the layer to get

        Returns:
            the layer, or None if it does not exist yet.
        """
        layer_dir = self._get_layer_dir(layer_id)
        if not layer_dir.exists("metadata.json"):
            return None
        return FileTileStoreLayer(
            layer_dir,
            raster_format=self.raster_format,
            vector_format=self.vector_format,
        )

    def list_layers(self, prefix: tuple[str, ...] = tuple()) -> list[str]:
        """List options for next part of layer ID with the specified prefix.

        Args:
            prefix: the prefix to match

        Returns:
            available options for next part of the layer ID
        """
        layer_dir = self._get_layer_dir(prefix)
        return layer_dir.listdir()

    @staticmethod
    def from_config(config: TileStoreConfig, ds_file_api: FileAPI) -> "FileTileStore":
        """Initialize a FileTileStore from configuration.

        Args:
            config: the TileStoreConfig
            ds_file_api: the dataset FileAPI

        Returns:
            the FileTileStore
        """
        d = config.config_dict
        kwargs = {"file_api": ds_file_api.get_folder(d["root_dir"])}
        if "raster_format" in d:
            kwargs["raster_format"] = load_raster_format(
                RasterFormatConfig.from_config(d["raster_format"])
            )
        if "vector_format" in d:
            kwargs["vector_format"] = load_vector_format(
                VectorFormatConfig.from_config(d["vector_format"])
            )
        return FileTileStore(**kwargs)
