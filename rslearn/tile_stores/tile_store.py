"""Base class for tile stores."""

from typing import Any

import numpy.typing as npt
from rasterio.enums import Resampling
from upath import UPath

from rslearn.utils import Feature, PixelBounds, Projection


class TileStore:
    """An abstract class for a tile store.

    A tile store supports operations to read and write raster and vector data.
    """

    def set_dataset_path(self, ds_path: UPath) -> None:
        """Set the dataset path.

        This is in case the TileStore wants to use the ds_path to help determine where
        to store data.

        Args:
            ds_path: the dataset path that this TileStore is a part of.
        """
        pass

    def is_raster_ready(
        self, layer_name: str, item_name: str, bands: list[str]
    ) -> bool:
        """Checks if this raster has been written to the store.

        Args:
            layer_name: the layer name or alias.
            item_name: the item.
            bands: the list of bands identifying which specific raster to read.

        Returns:
            whether there is a raster in the store matching the source, item, and
                bands.
        """
        raise NotImplementedError

    def get_raster_bands(self, layer_name: str, item_name: str) -> list[list[str]]:
        """Get the sets of bands that have been stored for the specified item.

        Args:
            layer_name: the layer name or alias.
            item_name: the item.

        Returns:
            a list of lists of bands that are in the tile store (with one raster
                stored corresponding to each inner list). If no rasters are ready for
                this item, returns empty list.
        """
        raise NotImplementedError

    def get_raster_bounds(
        self, layer_name: str, item_name: str, bands: list[str], projection: Projection
    ) -> PixelBounds:
        """Get the bounds of the raster in the specified projection.

        Args:
            layer_name: the layer name or alias.
            item_name: the item to check.
            bands: the list of bands identifying which specific raster to read. These
                bands must match the bands of a stored raster.
            projection: the projection to get the raster's bounds in.

        Returns:
            the bounds of the raster in the projection.
        """
        raise NotImplementedError

    def read_raster(
        self,
        layer_name: str,
        item_name: str,
        bands: list[str],
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
    ) -> npt.NDArray[Any]:
        """Read raster data from the store.

        Args:
            layer_name: the layer name or alias.
            item_name: the item to read.
            bands: the list of bands identifying which specific raster to read. These
                bands must match the bands of a stored raster.
            projection: the projection to read in.
            bounds: the bounds to read.
            resampling: the resampling method to use in case reprojection is needed.

        Returns:
            the raster data
        """
        raise NotImplementedError

    def write_raster(
        self,
        layer_name: str,
        item_name: str,
        bands: list[str],
        projection: Projection,
        bounds: PixelBounds,
        array: npt.NDArray[Any],
    ) -> None:
        """Write raster data to the store.

        Args:
            layer_name: the layer name or alias.
            item_name: the item to write.
            bands: the list of bands in the array.
            projection: the projection of the array.
            bounds: the bounds of the array.
            array: the raster data.
        """
        raise NotImplementedError

    def write_raster_file(
        self, layer_name: str, item_name: str, bands: list[str], fname: UPath
    ) -> None:
        """Write raster data to the store.

        Args:
            layer_name: the layer name or alias.
            item_name: the item to write.
            bands: the list of bands in the array.
            fname: the raster file.
        """
        raise NotImplementedError

    def is_vector_ready(self, layer_name: str, item_name: str) -> bool:
        """Checks if this vector item has been written to the store.

        Args:
            layer_name: the layer name or alias.
            item_name: the item.

        Returns:
            whether the vector data from the item has been stored.
        """
        raise NotImplementedError

    def read_vector(
        self,
        layer_name: str,
        item_name: str,
        projection: Projection,
        bounds: PixelBounds,
    ) -> list[Feature]:
        """Read vector data from the store.

        Args:
            layer_name: the layer name or alias.
            item_name: the item to read.
            projection: the projection to read in.
            bounds: the bounds within which to read.

        Returns:
            the vector data
        """
        raise NotImplementedError

    def write_vector(
        self, layer_name: str, item_name: str, features: list[Feature]
    ) -> None:
        """Write vector data to the store.

        Args:
            layer_name: the layer name or alias.
            item_name: the item to write.
            features: the vector data.
        """
        raise NotImplementedError


class TileStoreWithLayer:
    """Convenience class to access TileStore in the context of a layer."""

    def __init__(self, tile_store: TileStore, layer_name: str):
        """Create a new TileStoreWithLayer.

        Args:
            tile_store: underlying TileStore.
            layer_name: the layer name.
        """
        self.tile_store = tile_store
        self.layer_name = layer_name

    def is_raster_ready(self, item_name: str, bands: list[str]) -> bool:
        """Checks if this raster has been written to the store.

        Args:
            item_name: the item.
            bands: the list of bands identifying which specific raster to read.

        Returns:
            whether there is a raster in the store matching the source, item, and
                bands.
        """
        return self.tile_store.is_raster_ready(self.layer_name, item_name, bands)

    def get_raster_bands(self, item_name: str) -> list[list[str]]:
        """Get the sets of bands that have been stored for the specified item.

        Args:
            item_name: the item.

        Returns:
            a list of lists of bands that are in the tile store (with one raster
                stored corresponding to each inner list). If no rasters are ready for
                this item, returns empty list.
        """
        return self.tile_store.get_raster_bands(self.layer_name, item_name)

    def get_raster_bounds(
        self, item_name: str, bands: list[str], projection: Projection
    ) -> PixelBounds:
        """Get the bounds of the raster in the specified projection.

        Args:
            item_name: the item to check.
            bands: the list of bands identifying which specific raster to read. These
                bands must match the bands of a stored raster.
            projection: the projection to get the raster's bounds in.

        Returns:
            the bounds of the raster in the projection.
        """
        return self.tile_store.get_raster_bounds(
            self.layer_name, item_name, bands, projection
        )

    def read_raster(
        self,
        item_name: str,
        bands: list[str],
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
    ) -> npt.NDArray[Any]:
        """Read raster data from the store.

        Args:
            item_name: the item to read.
            bands: the list of bands identifying which specific raster to read. These
                bands must match the bands of a stored raster.
            projection: the projection to read in.
            bounds: the bounds to read.
            resampling: the resampling method to use in case reprojection is needed.

        Returns:
            the raster data
        """
        return self.tile_store.read_raster(
            self.layer_name, item_name, bands, projection, bounds, resampling
        )

    def write_raster(
        self,
        item_name: str,
        bands: list[str],
        projection: Projection,
        bounds: PixelBounds,
        array: npt.NDArray[Any],
    ) -> None:
        """Write raster data to the store.

        Args:
            item_name: the item to write.
            bands: the list of bands in the array.
            projection: the projection of the array.
            bounds: the bounds of the array.
            array: the raster data.
        """
        self.tile_store.write_raster(
            self.layer_name, item_name, bands, projection, bounds, array
        )

    def write_raster_file(self, item_name: str, bands: list[str], fname: UPath) -> None:
        """Write raster data to the store.

        Args:
            item_name: the item to write.
            bands: the list of bands in the array.
            fname: the raster file.
        """
        self.tile_store.write_raster_file(self.layer_name, item_name, bands, fname)

    def is_vector_ready(self, item_name: str) -> bool:
        """Checks if this vector item has been written to the store.

        Args:
            item_name: the item.

        Returns:
            whether the vector data from the item has been stored.
        """
        return self.tile_store.is_vector_ready(self.layer_name, item_name)

    def read_vector(
        self, item_name: str, projection: Projection, bounds: PixelBounds
    ) -> list[Feature]:
        """Read vector data from the store.

        Args:
            item_name: the item to read.
            projection: the projection to read in.
            bounds: the bounds within which to read.

        Returns:
            the vector data
        """
        return self.tile_store.read_vector(
            self.layer_name, item_name, projection, bounds
        )

    def write_vector(self, item_name: str, features: list[Feature]) -> None:
        """Write vector data to the store.

        Args:
            item_name: the item to write.
            features: the vector data.
        """
        self.tile_store.write_vector(self.layer_name, item_name, features)
