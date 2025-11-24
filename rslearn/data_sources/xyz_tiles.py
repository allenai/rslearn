"""Data source for xyz tiles."""

import math
import urllib.request
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np
import numpy.typing as npt
import rasterio.transform
import rasterio.warp
import shapely
from PIL import Image
from rasterio.crs import CRS
from rasterio.enums import Resampling

from rslearn.config import LayerConfig, QueryConfig
from rslearn.dataset import Window
from rslearn.dataset.materialize import RasterMaterializer
from rslearn.tile_stores import TileStore, TileStoreWithLayer
from rslearn.utils import PixelBounds, Projection, STGeometry
from rslearn.utils.array import copy_spatial_array
from rslearn.utils.raster_format import get_transform_from_projection_and_bounds

from .data_source import DataSource, DataSourceContext, Item
from .utils import match_candidate_items_to_window

WEB_MERCATOR_EPSG = 3857
WEB_MERCATOR_UNITS = 2 * math.pi * 6378137


def read_from_tile_callback(
    bounds: PixelBounds,
    callback: Callable[[int, int], npt.NDArray[Any] | None],
    tile_size: int = 256,
) -> npt.NDArray[Any]:
    """Read raster data from tiles.

    We assume tile (0, 0) covers pixels from (0, 0) to (tile_size, tile_size), while
    tile (-5, 5) covers pixels from (-5*tile_size, 5*tile_size) to
    (-4*tile_size, 6*tile_size).

    Args:
        bounds: the bounds to read
        callback: a callback to read the CHW tile at a given (column, row).
        tile_size: the tile size (grid size)

    Returns:
        raster data corresponding to bounds
    """
    data = None
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]

    start_tile = (bounds[0] // tile_size, bounds[1] // tile_size)
    end_tile = ((bounds[2] - 1) // tile_size, (bounds[3] - 1) // tile_size)
    for tile_col in range(start_tile[0], end_tile[0] + 1):
        for tile_row in range(start_tile[1], end_tile[1] + 1):
            cur_im = callback(tile_col, tile_row)
            if cur_im is None:
                # Callback can return None if no image is available here.
                continue

            if len(cur_im.shape) == 2:
                # Add channel dimension for greyscale images.
                cur_im = cur_im[None, :, :]

            if data is None:
                # Initialize data now that we know how many bands there are.
                data = np.zeros((cur_im.shape[0], height, width), dtype=cur_im.dtype)

            cur_col_off = tile_size * tile_col
            cur_row_off = tile_size * tile_row

            copy_spatial_array(
                src=cur_im,
                dst=data,
                src_offset=(cur_col_off, cur_row_off),
                dst_offset=(bounds[0], bounds[1]),
            )

    return data


class XyzTiles(DataSource, TileStore):
    """A data source for web xyz image tiles.

    These tiles are usually in WebMercator projection, but different CRS can be
    configured here.
    """

    def __init__(
        self,
        url_templates: list[str],
        time_ranges: list[tuple[datetime, datetime]],
        zoom: int,
        crs: str | CRS = CRS.from_epsg(WEB_MERCATOR_EPSG),
        total_units: float = WEB_MERCATOR_UNITS,
        offset: float = WEB_MERCATOR_UNITS / 2,
        tile_size: int = 256,
        band_names: list[str] = ["R", "G", "B"],
        context: DataSourceContext = DataSourceContext(),
    ):
        """Initialize an XyzTiles instance.

        It is configured with a list of URL templates and corresponding time ranges.
        The URL template should have placeholders that allow accessing an arbitrary
        grid cell of a global mosaic. Sources that have a single layer of the world can
        be configured with a single URL template and arbitrary time range, but multiple
        templates / time ranges is supported for sources that expose image time series.

        Args:
            url_templates: the image tile URLs with "{x}" (column), "{y}" (row), and
                "{z}" (zoom) placeholders.
            time_ranges: corresponding list of time ranges for each URL template.
            zoom: the zoom level. Currently a single zoom level must be used.
            crs: the CRS, defaults to WebMercator.
            total_units: the total projection units along each axis. Used to determine
                the pixel size to map from projection coordinates to pixel coordinates.
            offset: offset added to projection units when converting to tile positions.
            tile_size: size in pixels of each tile. Tiles must be square.
            band_names: what to name the bands that we read.
            context: the data source context.
        """
        self.url_templates = url_templates
        self.time_ranges = time_ranges
        self.zoom = zoom
        self.total_units = total_units
        self.offset = offset
        self.tile_size = tile_size
        self.band_names = band_names

        # Convert to CRS if needed.
        if isinstance(crs, str):
            self.crs = CRS.from_string(crs)
        else:
            self.crs = crs

        # Compute total number of pixels (a function of the zoom level and tile size).
        self.total_pixels = tile_size * (2**zoom)
        # Compute pixel size (resolution).
        self.pixel_size = self.total_units / self.total_pixels
        # Compute offset in pixels.
        self.pixel_offset = int(self.offset / self.pixel_size)
        # Compute the extent in pixel coordinates as an STGeometry.
        # Note that pixel coordinates are prior to applying the offset.
        self.shp = shapely.box(
            -self.total_pixels // 2,
            -self.total_pixels // 2,
            self.total_pixels // 2,
            self.total_pixels // 2,
        )
        self.projection = Projection(self.crs, self.pixel_size, -self.pixel_size)

        self.items = []
        for url_template, time_range in zip(self.url_templates, self.time_ranges):
            geometry = STGeometry(self.projection, self.shp, time_range)
            item = Item(url_template, geometry)
            self.items.append(item)

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[Item]]]:
        """Get a list of items in the data source intersecting the given geometries.

        In XyzTiles we treat the data source as containing a single item, i.e., the
        entire image at the configured zoom level. So we always return a single group
        containing the single same item, for each geometry.

        Args:
            geometries: the spatiotemporal geometries
            query_config: the query configuration

        Returns:
            List of groups of items that should be retrieved for each geometry.
        """
        groups = []
        for geometry in geometries:
            geometry = geometry.to_projection(self.projection)
            cur_groups = match_candidate_items_to_window(
                geometry, self.items, query_config
            )
            groups.append(cur_groups)
        return groups

    def deserialize_item(self, serialized_item: Any) -> Item:
        """Deserializes an item from JSON-decoded data."""
        return Item.deserialize(serialized_item)

    def read_tile(self, url_template: str, col: int, row: int) -> npt.NDArray[Any]:
        """Read the tile at specified column and row.

        Args:
            url_template: the URL template to use
            col: the tile column
            row: the tile row

        Returns:
            the raster data of this tile
        """
        url = url_template
        url = url.replace("{x}", str(col))
        url = url.replace("{y}", str(row))
        url = url.replace("{z}", str(self.zoom))
        image = np.array(Image.open(urllib.request.urlopen(url)))
        # Handle grayscale images (add single-band channel dimension).
        if len(image.shape) == 2:
            image = image[:, :, None]
        return image.transpose(2, 0, 1)

    def read_bounds(self, url_template: str, bounds: PixelBounds) -> npt.NDArray[Any]:
        """Reads the portion of the raster in the specified bounds.

        Args:
            url_template: the URL template to read from
            bounds: the bounds to read

        Returns:
            CHW numpy array containing raster data corresponding to the bounds.
        """
        # Add the tile/grid offset to the bounds before reading.
        bounds = (
            bounds[0] + self.pixel_offset,
            bounds[1] + self.pixel_offset,
            bounds[2] + self.pixel_offset,
            bounds[3] + self.pixel_offset,
        )
        return read_from_tile_callback(
            bounds,
            lambda col, row: self.read_tile(url_template, col, row),
            self.tile_size,
        )

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
        # Always ready since we wrap accesses to the XYZ tile URL.
        return True

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
        return [self.band_names]

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
        geom = STGeometry(self.projection, self.shp, None).to_projection(projection)
        return (
            int(geom.shp.bounds[0]),
            int(geom.shp.bounds[1]),
            int(geom.shp.bounds[2]),
            int(geom.shp.bounds[3]),
        )

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
        # Validate bands.
        if bands != self.band_names:
            raise ValueError(
                f"expected request for bands {self.band_names} but requested {bands}"
            )

        # Read a raster matching the given bounds but projected onto the projection of
        # the xyz tiles.
        request_geometry = STGeometry(projection, shapely.box(*bounds), None)
        projected_geometry = request_geometry.to_projection(self.projection)
        projected_bounds = (
            math.floor(projected_geometry.shp.bounds[0]),
            math.floor(projected_geometry.shp.bounds[1]),
            math.ceil(projected_geometry.shp.bounds[2]),
            math.ceil(projected_geometry.shp.bounds[3]),
        )
        # The item name is the URL template.
        url_template = item_name
        array = self.read_bounds(url_template, projected_bounds)
        # Now project it back to the requested geometry.
        src_transform = get_transform_from_projection_and_bounds(
            self.projection, projected_bounds
        )
        dst_transform = get_transform_from_projection_and_bounds(projection, bounds)
        dst_array = np.zeros(
            (array.shape[0], bounds[3] - bounds[1], bounds[2] - bounds[0]),
            dtype=array.dtype,
        )
        rasterio.warp.reproject(
            source=array,
            src_crs=self.projection.crs,
            src_transform=src_transform,
            destination=dst_array,
            dst_crs=projection.crs,
            dst_transform=dst_transform,
            resampling=resampling,
        )
        return dst_array

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
        RasterMaterializer().materialize(
            TileStoreWithLayer(self, layer_name),
            window,
            layer_name,
            layer_cfg,
            item_groups,
        )
