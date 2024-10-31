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
from upath import UPath

from rslearn.config import LayerConfig, QueryConfig, RasterLayerConfig
from rslearn.dataset import Window
from rslearn.utils import PixelBounds, Projection, STGeometry
from rslearn.utils.array import copy_spatial_array

from .data_source import DataSource, Item
from .raster_source import ArrayWithTransform, materialize_raster
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


class XyzItem(Item):
    """An item in the XyzTiles data source.

    Each item represents one layer of tiles. Often there is only one itm in the data
    source, but if there are multiple then they should correspond to different time
    ranges.
    """

    def __init__(self, name: str, geometry: STGeometry, url_template: str):
        """Creates a new XyzItem.

        Args:
            name: unique name of the item
            geometry: the spatial and temporal extent of the item
            url_template: the URL template for an xyz tile.
        """
        super().__init__(name, geometry)
        self.url_template = url_template

    def serialize(self) -> dict:
        """Serializes the item to a JSON-encodable dictionary."""
        d = super().serialize()
        d["url_template"] = self.url_template
        return d

    @staticmethod
    def deserialize(d: dict) -> Item:
        """Deserializes an item from a JSON-decoded dictionary."""
        item = super(XyzItem, XyzItem).deserialize(d)
        return XyzItem(
            name=item.name, geometry=item.geometry, url_template=d["url_template"]
        )


class XyzTiles(DataSource):
    """A data source for web xyz image tiles.

    These tiles are usually in WebMercator projection, but different CRS can be
    configured here.
    """

    item_name = "xyz_tiles"

    def __init__(
        self,
        url_templates: list[str],
        time_ranges: list[tuple[datetime, datetime]],
        zoom: int,
        crs: CRS = CRS.from_epsg(WEB_MERCATOR_EPSG),
        total_units: float = WEB_MERCATOR_UNITS,
        offset: float = WEB_MERCATOR_UNITS / 2,
        tile_size: int = 256,
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
        """
        self.url_templates = url_templates
        self.time_ranges = time_ranges
        self.zoom = zoom
        self.crs = crs
        self.total_units = total_units
        self.offset = offset
        self.tile_size = tile_size

        # Compute total number of pixels (a function of the zoom level and tile size).
        self.total_pixels = tile_size * (2**zoom)
        # Compute pixel size (resolution).
        self.pixel_size = self.total_units / self.total_pixels
        # Compute offset in pixels.
        self.pixel_offset = int(self.offset / self.pixel_size)
        # Compute the extent in pixel coordinates as an STGeometry.
        # Note that pixel coordinates are prior to applying the offset.
        shp = shapely.box(
            -self.total_pixels // 2,
            -self.total_pixels // 2,
            self.total_pixels // 2,
            self.total_pixels // 2,
        )
        self.projection = Projection(self.crs, self.pixel_size, -self.pixel_size)

        self.items = []
        for url_template, time_range in zip(self.url_templates, self.time_ranges):
            geometry = STGeometry(self.projection, shp, time_range)
            item = XyzItem(self.item_name, geometry, url_template)
            self.items.append(item)

    @staticmethod
    def from_config(config: LayerConfig, ds_path: UPath) -> "XyzTiles":
        """Creates a new XyzTiles instance from a configuration dictionary."""
        if config.data_source is None:
            raise ValueError("data_source is required")
        d = config.data_source.config_dict
        time_ranges = []
        for str1, str2 in d["time_ranges"]:
            time1 = datetime.fromisoformat(str1)
            time2 = datetime.fromisoformat(str2)
            time_ranges.append((time1, time2))
        kwargs = dict(
            url_templates=d["url_templates"], zoom=d["zoom"], time_ranges=time_ranges
        )
        if "crs" in d:
            kwargs["crs"] = CRS.from_string(d["crs"])
        if "total_units" in d:
            kwargs["total_units"] = d["total_units"]
        if "offset" in d:
            kwargs["offset"] = d["offset"]
        if "tile_size" in d:
            kwargs["tile_size"] = d["tile_size"]
        return XyzTiles(**kwargs)

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[XyzItem]]]:
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
        return XyzItem.deserialize(serialized_item)

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
        image = Image.open(urllib.request.urlopen(url))
        return np.array(image).transpose(2, 0, 1)

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

    def materialize(
        self,
        window: Window,
        item_groups: list[list[XyzItem]],
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
        assert len(item_groups) == 1 and len(item_groups[0]) == 1
        item = item_groups[0][0]
        assert isinstance(item, XyzItem)

        # Read a raster matching the bounds of the window's bounds projected onto the
        # projection of the xyz tiles.
        assert isinstance(layer_cfg, RasterLayerConfig)
        band_cfg = layer_cfg.band_sets[0]
        window_projection, window_bounds = band_cfg.get_final_projection_and_bounds(
            window.projection, window.bounds
        )
        window_geometry = STGeometry(
            window_projection, shapely.box(*window_bounds), None
        )
        projected_geometry = window_geometry.to_projection(self.projection)
        projected_bounds = tuple(
            math.floor(projected_geometry.shp.bounds[i]) for i in range(4)
        )
        projected_raster = self.read_bounds(item.url_template, projected_bounds)  # type: ignore

        # Attach the transform to the raster.
        src_transform = rasterio.transform.Affine(
            self.projection.x_resolution,
            0,
            projected_bounds[0] * self.projection.x_resolution,
            0,
            self.projection.y_resolution,
            projected_bounds[1] * self.projection.y_resolution,
        )
        array_with_transform = ArrayWithTransform(
            projected_raster, self.projection.crs, src_transform
        )

        materialize_raster(array_with_transform, window, layer_name, band_cfg)
