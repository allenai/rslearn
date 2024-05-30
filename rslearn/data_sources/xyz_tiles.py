"""Data source for xyz tiles."""

import math
import os
import urllib.request
from typing import Any, Callable, Optional

import numpy as np
import numpy.typing as npt
import rasterio.transform
import rasterio.warp
import shapely
from PIL import Image
from rasterio.crs import CRS

from rslearn.config import (
    LayerConfig,
    QueryConfig,
    RasterFormatConfig,
    RasterLayerConfig,
)
from rslearn.dataset import Window
from rslearn.utils import LocalFileAPI, PixelBounds, Projection, STGeometry
from rslearn.utils.raster_format import load_raster_format

from .data_source import DataSource, Item

WEB_MERCATOR_EPSG = 3857
WEB_MERCATOR_UNITS = 2 * math.pi * 6378137


def read_from_tile_callback(bounds: PixelBounds, callback: Callable[[int, int], Optional[npt.NDArray[Any]]], tile_size: int = 256) -> npt.NDArray[Any]:
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

    start_tile = (bounds[0]//tile_size, bounds[1]//tile_size)
    end_tile = ((bounds[2]-1)//tile_size, (bounds[3]-1)//tile_size)
    for tile_col in range(start_tile[0], end_tile[0]+1):
        for tile_row in range(start_tile[1], end_tile[1]+1):
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

            src_col_offset = max(bounds[0] - cur_col_off, 0)
            src_row_offset = max(bounds[1] - cur_row_off, 0)
            dst_col_offset = max(cur_col_off - bounds[0], 0)
            dst_row_offset = max(cur_row_off - bounds[1], 0)
            col_overlap = min(cur_im.shape[2] - src_col_offset, width - dst_col_offset)
            row_overlap = min(cur_im.shape[1] - src_row_offset, height - dst_row_offset)
            data[:, dst_row_offset:dst_row_offset+row_overlap, dst_col_offset:dst_col_offset+col_overlap] = cur_im[:, src_row_offset:src_row_offset+row_overlap, src_col_offset:src_col_offset+col_overlap]

    return data


class XyzTiles(DataSource):
    """A data source for web xyz image tiles.

    These tiles are usually in WebMercator projection, but different CRS can be
    configured here.
    """

    item_name = "xyz_tiles"

    def __init__(self, url_template: str, zoom: int, crs: CRS = CRS.from_epsg(WEB_MERCATOR_EPSG), total_units: float = WEB_MERCATOR_UNITS, offset: float = WEB_MERCATOR_UNITS/2, tile_size: int = 256):
        """Initialize an XyzTiles instance.

        Args:
            url_template: the image tile URL with "{x}" (column), "{y}" (row), and
                "{z}" (zoom) placeholders.
            zoom: the zoom level. Currently a single zoom level must be used.
            crs: the CRS, defaults to WebMercator.
            total_units: the total projection units along each axis. Used to determine
                the pixel size to map from projection coordinates to pixel coordinates.
            offset: offset added to projection units when converting to tile positions.
            tile_size: size in pixels of each tile. Tiles must be square.
        """
        self.url_template = url_template
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
        shp = shapely.box(-self.total_pixels//2, -self.total_pixels//2, self.total_pixels//2, self.total_pixels//2)
        self.projection = Projection(self.crs, self.pixel_size, -self.pixel_size)
        self.geometry = STGeometry(self.projection, shp, None)
        self.item = Item(self.item_name, self.geometry)

    @staticmethod
    def from_config(config: LayerConfig, root_dir: str = ".") -> "XyzTiles":
        """Creates a new XyzTiles instance from a configuration dictionary."""
        d = config.data_source.config_dict
        kwargs = dict(
            url_template=d["url_template"],
            zoom=d["zoom"],
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
        return [[[self.item]]] * len(geometries)

    def deserialize_item(self, serialized_item: Any) -> Item:
        """Deserializes an item from JSON-decoded data."""
        return Item.deserialize(serialized_item)

    def read_tile(self, col: int, row: int) -> npt.NDArray[Any]:
        """Read the tile at specified column and row.

        Args:
            col: the tile column
            row: the tile row

        Returns:
            the raster data of this tile
        """
        url = self.url_template
        url = url.replace("{x}", str(col))
        url = url.replace("{y}", str(row))
        url = url.replace("{z}", str(self.zoom))
        image = Image.open(urllib.request.urlopen(url))
        return np.array(image).transpose(2, 0, 1)

    def read_bounds(self, bounds: PixelBounds) -> npt.NDArray[Any]:
        """Reads the portion of the raster in the specified bounds.

        Args:
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
        return read_from_tile_callback(bounds, self.read_tile, self.tile_size)

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
        # Read a raster matching the bounds of the window's bounds projected onto the
        # projection of the xyz tiles.
        assert isinstance(layer_cfg, RasterLayerConfig)
        band_cfg = layer_cfg.band_sets[0]
        window_projection, window_bounds = band_cfg.get_final_projection_and_bounds(
            window.projection, window.bounds
        )
        window_geometry = STGeometry(
            window_projection,
            shapely.box(*window_bounds),
            None,
        )
        projected_geometry = window_geometry.to_projection(self.projection)
        projected_bounds = [
            math.floor(projected_geometry.shp.bounds[0]),
            math.floor(projected_geometry.shp.bounds[1]),
            math.ceil(projected_geometry.shp.bounds[2]),
            math.ceil(projected_geometry.shp.bounds[3]),
        ]
        projected_raster = self.read_bounds(projected_bounds)

        # Now we need to project it back to the desired projection.
        window_width = window_bounds[2] - window_bounds[0]
        window_height = window_bounds[3] - window_bounds[1]
        src_transform = rasterio.transform.Affine(
            self.projection.x_resolution,
            0,
            projected_bounds[0] * self.projection.x_resolution,
            0,
            self.projection.y_resolution,
            projected_bounds[1] * self.projection.y_resolution,
        )
        dst_transform = rasterio.transform.Affine(
            window_projection.x_resolution,
            0,
            window_bounds[0] * window_projection.x_resolution,
            0,
            window_projection.y_resolution,
            window_bounds[1] * window_projection.y_resolution,
        )
        raster = np.zeros((projected_raster.shape[0], window_height, window_width), dtype=projected_raster.dtype)
        rasterio.warp.reproject(
            source=projected_raster,
            src_crs=self.projection.crs,
            src_transform=src_transform,
            destination=raster,
            dst_crs=window_projection.crs,
            dst_transform=dst_transform,
            resampling=rasterio.enums.Resampling.bilinear,
        )

        # And then write the data to the window directory.
        # TODO: this part shouldn't really go in the data source.
        assert isinstance(layer_cfg, RasterLayerConfig)
        out_dir = os.path.join(
            window.window_root,
            "layers",
            layer_name,
            "R_G_B",
        )
        if os.path.exists(out_dir):
            return
        tmp_out_dir = out_dir + ".tmp"
        os.makedirs(tmp_out_dir, exist_ok=True)
        band_cfg = layer_cfg.band_sets[0]
        raster_format = load_raster_format(
            RasterFormatConfig(band_cfg.format["name"], band_cfg.format)
        )
        raster_format.encode_raster(
            LocalFileAPI(tmp_out_dir), window_projection, window_bounds, raster
        )
        os.rename(tmp_out_dir, out_dir)
