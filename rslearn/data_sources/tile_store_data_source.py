"""Base class for data sources that support direct materialization via TileStore."""

from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Generic

import affine
import numpy.typing as npt
import rasterio
import rasterio.vrt
from rasterio.enums import Resampling

from rslearn.config import LayerConfig
from rslearn.data_sources.data_source import DataSource, Item, ItemType
from rslearn.dataset import Window
from rslearn.dataset.materialize import RasterMaterializer
from rslearn.tile_stores import TileStore, TileStoreWithLayer
from rslearn.utils.geometry import PixelBounds, Projection


class TileStoreDataSource(DataSource[ItemType], TileStore, Generic[ItemType]):
    """Base class for data sources that support direct materialization via TileStore.

    This class provides common TileStore functionality for data sources that can read
    raster data on-demand from remote sources (like cloud buckets or APIs) without
    first ingesting into a local tile store.

    Subclasses must implement:
        - get_asset_url(): Get the URL for an asset given item name and bands
        - get_item_by_name(): Get an item by its name

    Subclasses may optionally override:
        - get_raster_bands(): By default, we assume that items have all assets. If
            items may have a subset of assets, override get_raster_bands to return
            the sets of bands available for that item.
        - get_read_callback(): Returns a callback to transform the raster array,
            for post-processing like Sentinel-2 harmonization.
    """

    def __init__(self, asset_bands: dict[str, list[str]]):
        """Initialize the TileStoreDataSource.

        Args:
            asset_bands: mapping from asset key to the list of band names in that asset.
        """
        self.asset_bands = asset_bands

    def _get_asset_key_by_bands(self, bands: list[str]) -> str:
        """Get the asset key based on the band names.

        Args:
            bands: list of band names to look up.

        Returns:
            the asset key that provides those bands.

        Raises:
            ValueError: if no asset provides those bands.
        """
        for asset_key, asset_bands in self.asset_bands.items():
            if bands == asset_bands:
                return asset_key
        raise ValueError(f"no known asset with bands {bands}")

    # --- Abstract methods that subclasses must implement ---

    @abstractmethod
    def get_asset_url(self, item_name: str, asset_key: str) -> str:
        """Get the URL to read the asset for the given item and asset key.

        Args:
            item_name: the name of the item.
            asset_key: the key identifying which asset to get.

        Returns:
            the URL to read the asset from (must be readable by rasterio).
        """
        raise NotImplementedError

    @abstractmethod
    def get_item_by_name(self, name: str) -> ItemType:
        """Get an item by its name.

        Args:
            name: the name of the item to get.

        Returns:
            the item object.
        """
        raise NotImplementedError

    # --- Optional hooks for subclasses ---

    def get_read_callback(
        self, item_name: str, asset_key: str
    ) -> Callable[[npt.NDArray[Any]], npt.NDArray[Any]] | None:
        """Return a callback to post-process raster data (e.g., harmonization).

        Subclasses can override this to apply transformations to the raw raster data
        after reading, such as harmonization for Sentinel-2 data.

        Args:
            item_name: the name of the item being read.
            asset_key: the key identifying which asset is being read.

        Returns:
            A callback function that takes an array and returns a modified array,
            or None if no post-processing is needed.
        """
        return None

    # --- TileStore implementation ---

    def is_raster_ready(
        self, layer_name: str, item_name: str, bands: list[str]
    ) -> bool:
        """Checks if this raster has been written to the store.

        For remote-backed tile stores, this always returns True since data is
        read on-demand from the remote source.

        Args:
            layer_name: the layer name or alias.
            item_name: the item.
            bands: the list of bands identifying which specific raster to read.

        Returns:
            True, since data is always available from the remote source.
        """
        return True

    def get_raster_bands(self, layer_name: str, item_name: str) -> list[list[str]]:
        """Get the sets of bands that have been stored for the specified item.

        By default, returns all band sets from the asset_bands configuration.
        Subclasses can override this if not all items have all assets.

        Args:
            layer_name: the layer name or alias.
            item_name: the item.

        Returns:
            a list of lists of bands available for this item.
        """
        return list(self.asset_bands.values())

    def get_raster_bounds(
        self, layer_name: str, item_name: str, bands: list[str], projection: Projection
    ) -> PixelBounds:
        """Get the bounds of the raster in the specified projection.

        Args:
            layer_name: the layer name or alias.
            item_name: the item to check.
            bands: the list of bands identifying which specific raster to read.
            projection: the projection to get the raster's bounds in.

        Returns:
            the bounds of the raster in the projection.
        """
        item = self.get_item_by_name(item_name)
        geom = item.geometry.to_projection(projection)
        return (
            int(geom.shp.bounds[0]),
            int(geom.shp.bounds[1]),
            int(geom.shp.bounds[2]),
            int(geom.shp.bounds[3]),
        )

    def _read_raster_from_url(
        self,
        url: str,
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling,
    ) -> npt.NDArray[Any]:
        """Read raster data from a URL with reprojection.

        This is the common logic for reading raster data from a URL and reprojecting
        it to the target projection and bounds using rasterio's WarpedVRT.

        Args:
            url: the URL to read from (must be readable by rasterio).
            projection: the projection to read in.
            bounds: the bounds to read.
            resampling: the resampling method to use.

        Returns:
            the raster data as a numpy array.
        """
        # Construct the transform to use for the warped dataset.
        wanted_transform = affine.Affine(
            projection.x_resolution,
            0,
            bounds[0] * projection.x_resolution,
            0,
            projection.y_resolution,
            bounds[1] * projection.y_resolution,
        )

        with rasterio.open(url) as src:
            with rasterio.vrt.WarpedVRT(
                src,
                crs=projection.crs,
                transform=wanted_transform,
                width=bounds[2] - bounds[0],
                height=bounds[3] - bounds[1],
                resampling=resampling,
            ) as vrt:
                return vrt.read()

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
            bands: the list of bands identifying which specific raster to read.
            projection: the projection to read in.
            bounds: the bounds to read.
            resampling: the resampling method to use in case reprojection is needed.

        Returns:
            the raster data as a numpy array.
        """
        # Get the asset key for the requested bands
        asset_key = self._get_asset_key_by_bands(bands)

        # Get the asset URL from the subclass
        asset_url = self.get_asset_url(item_name, asset_key)

        # Read the raster data
        raw_data = self._read_raster_from_url(asset_url, projection, bounds, resampling)

        # Apply any post-processing callback
        callback = self.get_read_callback(item_name, asset_key)
        if callback is not None:
            raw_data = callback(raw_data)

        return raw_data

    def materialize(
        self,
        window: Window,
        item_groups: list[list[ItemType]],
        layer_name: str,
        layer_cfg: LayerConfig,
    ) -> None:
        """Materialize data for the window.

        Args:
            window: the window to materialize.
            item_groups: the items from get_items.
            layer_name: the name of this layer.
            layer_cfg: the config of this layer.
        """
        RasterMaterializer().materialize(
            TileStoreWithLayer(self, layer_name),
            window,
            layer_name,
            layer_cfg,
            item_groups,
        )

    # --- TileStore methods that are not supported ---

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

        This is not supported for remote-backed tile stores.
        """
        raise NotImplementedError(
            "TileStoreDataSource does not support writing raster data"
        )

    def write_raster_file(
        self, layer_name: str, item_name: str, bands: list[str], fname: Any
    ) -> None:
        """Write raster data to the store.

        This is not supported for remote-backed tile stores.
        """
        raise NotImplementedError(
            "TileStoreDataSource does not support writing raster files"
        )

    def is_vector_ready(self, layer_name: str, item_name: str) -> bool:
        """Checks if this vector item has been written to the store.

        This is not supported for remote-backed tile stores.
        """
        raise NotImplementedError(
            "TileStoreDataSource does not support vector operations"
        )

    def read_vector(
        self,
        layer_name: str,
        item_name: str,
        projection: Projection,
        bounds: PixelBounds,
    ) -> Any:
        """Read vector data from the store.

        This is not supported for remote-backed tile stores.
        """
        raise NotImplementedError(
            "TileStoreDataSource does not support vector operations"
        )

    def write_vector(
        self, layer_name: str, item_name: str, features: Any
    ) -> None:
        """Write vector data to the store.

        This is not supported for remote-backed tile stores.
        """
        raise NotImplementedError(
            "TileStoreDataSource does not support vector operations"
        )
