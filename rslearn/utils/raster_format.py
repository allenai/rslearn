from typing import Any, BinaryIO, Optional

import affine
import numpy as np
import numpy.typing as npt
import rasterio
from class_registry import ClassRegistry
from PIL import Image

from rslearn.config import RasterFormatConfig
from rslearn.const import TILE_SIZE

from .file_api import FileAPI
from .geometry import PixelBounds, Projection

RasterFormats = ClassRegistry()


class RasterFormat:
    def encode_raster(
        self,
        file_api: FileAPI,
        projection: Projection,
        bounds: PixelBounds,
        array: npt.NDArray[Any],
    ) -> None:
        """Encodes raster data.

        Args:
            file_api: the file API to write to
            projection: the projection of the raster data
            bounds: the bounds of the raster data in the projection
            array: the raster data
        """
        raise NotImplementedError

    def decode_raster(
        self, file_api: FileAPI, bounds: PixelBounds
    ) -> Optional[npt.NDArray[Any]]:
        """Decodes raster data.

        Args:
            file_api: the file API to read from
            bounds: the bounds of the raster to read

        Returns:
            the raster data, or None if no image content is found
        """
        raise NotImplementedError


@RasterFormats.register("image_tile")
class ImageTileRasterFormat(RasterFormat):
    def __init__(self, format: str, tile_size: int = 512):
        """Initialize a new ImageTileRasterFormat instance.

        Args:
            format: one of "geotiff", "png", "jpeg"
            tile_size: the tile size (grid size in pixels)
        """
        self.format = format
        self.tile_size = tile_size

    def encode_tile(
        self,
        f: BinaryIO,
        projection: Projection,
        bounds: PixelBounds,
        array: npt.NDArray[Any],
    ) -> None:
        if self.format == "png":
            array = array.transpose(1, 2, 0)
            if array.shape[2] == 1:
                array = array[:, :, 0]
            Image.fromarray(array).save(f, format=self.format.upper())

        elif self.format == "geotiff":
            crs = projection.crs
            transform = affine.Affine(
                projection.x_resolution,
                0,
                bounds[0] * projection.x_resolution,
                0,
                projection.y_resolution,
                bounds[1] * projection.y_resolution,
            )
            profile = {
                "driver": "GTiff",
                "compress": "lzw",
                "width": array.shape[2],
                "height": array.shape[1],
                "count": array.shape[0],
                "dtype": array.dtype.name,
                "crs": crs,
                "transform": transform,
            }
            with rasterio.open(f, "w", **profile) as dst:
                dst.write(array)

    def decode_tile(self, f: BinaryIO) -> npt.NDArray[Any]:
        if self.format == "png":
            array = np.array(Image.open(f, format=self.format.upper()))
            if len(array.shape) == 2:
                array = array[:, :, None]
            return array.transpose(2, 0, 1)

        elif self.format == "geotiff":
            with rasterio.open(f) as src:
                return src.read()

    def encode_raster(
        self,
        file_api: FileAPI,
        projection: Projection,
        bounds: PixelBounds,
        array: npt.NDArray[Any],
    ) -> None:
        """Encodes raster data.

        Args:
            file_api: the file API to write to
            projection: the projection of the raster data
            bounds: the bounds of the raster data in the projection
            array: the raster data (must be CHW)
        """
        start_tile = (bounds[0] // self.tile_size, bounds[1] // self.tile_size)
        end_tile = (bounds[2] // self.tile_size + 1, bounds[3] // self.tile_size + 1)
        extension = self.get_extension()

        # Pad the array so its corners are aligned with the tile grid.
        padding = (
            bounds[0] - start_tile[0] * self.tile_size,
            bounds[1] - start_tile[1] * self.tile_size,
            end_tile[0] * self.tile_size - bounds[2],
            end_tile[1] * self.tile_size - bounds[3],
        )
        array = np.pad(
            array, ((0, 0), (padding[1], padding[3]), (padding[0], padding[2]))
        )

        for col in range(start_tile[0], end_tile[0]):
            for row in range(start_tile[1], end_tile[1]):
                i = col - start_tile[0]
                j = row - start_tile[1]
                cur_array = array[
                    :,
                    j * self.tile_size : (j + 1) * self.tile_size,
                    i * self.tile_size : (i + 1) * self.tile_size,
                ]
                if np.count_nonzero(cur_array) == 0:
                    continue
                cur_bounds = (
                    col * self.tile_size,
                    row * self.tile_size,
                    (col + 1) * self.tile_size,
                    (row + 1) * self.tile_size,
                )
                with file_api.open(f"{col}_{row}.{extension}", "wb") as f:
                    self.encode_tile(f, projection, cur_bounds, cur_array)

    def decode_raster(
        self, file_api: FileAPI, bounds: PixelBounds
    ) -> Optional[npt.NDArray[Any]]:
        """Decodes raster data.

        Args:
            file_api: the file API to read from
            bounds: the bounds of the raster to read

        Returns:
            the raster data, or None if no image content is found
        """
        extension = self.get_extension()

        # Load tiles one at a time.
        start_tile = (bounds[0] // self.tile_size, bounds[1] // self.tile_size)
        end_tile = (
            (bounds[2] - 1) // self.tile_size + 1,
            (bounds[3] - 1) // self.tile_size + 1,
        )
        dst = None
        for col in range(start_tile[0], end_tile[0]):
            for row in range(start_tile[1], end_tile[1]):
                fname = f"{col}_{row}.{extension}"
                if not file_api.exists(fname):
                    continue
                with file_api.open(fname, "rb") as f:
                    src = self.decode_tile(f)

                if dst is None:
                    dst = np.array(
                        (src.shape[0], bounds[3] - bounds[1], bounds[2] - bounds[0]),
                        dtype=src.dtype,
                    )

                cur_col_off = col * self.tile_size
                cur_row_off = row * self.tile_size

                src_col_offset = max(bounds[0] - cur_col_off, 0)
                src_row_offset = max(bounds[1] - cur_row_off, 0)
                dst_col_offset = max(cur_col_off - bounds[0], 0)
                dst_row_offset = max(cur_row_off - bounds[1], 0)
                col_overlap = min(
                    src.shape[2] - src_col_offset, dst.shape[2] - dst_col_offset
                )
                row_overlap = min(
                    src.shape[1] - src_row_offset, dst.shape[1] - dst_row_offset
                )
                dst[
                    :,
                    dst_row_offset : dst_row_offset + row_overlap,
                    dst_col_offset : dst_col_offset + col_overlap,
                ] = src[
                    :,
                    src_row_offset : src_row_offset + row_overlap,
                    src_col_offset : src_col_offset + col_overlap,
                ]
        return dst

    def get_extension(self) -> str:
        if self.format == "png":
            return "png"
        elif self.format == "jpeg":
            return "jpg"
        elif self.format == "geotiff":
            return "tif"
        raise ValueError(f"unknown image format {self.format}")

    @staticmethod
    def from_config(name: str, config: dict[str, Any]) -> "ImageTileRasterFormat":
        return ImageTileRasterFormat(
            format=config.get("format", "geotiff"),
            tile_size=config.get("tile_size", 512),
        )


@RasterFormats.register("geotiff")
class GeotiffRasterFormat(RasterFormat):
    """A raster format that uses one big, tiled GeoTIFF with small block size."""

    fname = "geotiff.tif"

    def __init__(self, block_size: int = 512):
        self.block_size = block_size

    def encode_raster(
        self,
        file_api: FileAPI,
        projection: Projection,
        bounds: PixelBounds,
        array: npt.NDArray[Any],
    ) -> None:
        """Encodes raster data.

        Args:
            file_api: the file API to write to
            projection: the projection of the raster data
            bounds: the bounds of the raster data in the projection
            array: the raster data
        """
        crs = projection.crs
        transform = affine.Affine(
            projection.x_resolution,
            0,
            bounds[0] * projection.x_resolution,
            0,
            projection.y_resolution,
            bounds[1] * projection.y_resolution,
        )
        profile = {
            "driver": "GTiff",
            "compress": "lzw",
            "width": array.shape[2],
            "height": array.shape[1],
            "count": array.shape[0],
            "dtype": array.dtype.name,
            "crs": crs,
            "transform": transform,
        }
        if array.shape[2] > TILE_SIZE and array.shape[1] > TILE_SIZE:
            profile["tiled"] = True
            profile["blockxsize"] = TILE_SIZE
            profile["blockysize"] = TILE_SIZE
        with file_api.open(self.fname, "wb") as f:
            with rasterio.open(f, "w", **profile) as dst:
                dst.write(array)

    def decode_raster(
        self, file_api: FileAPI, bounds: PixelBounds
    ) -> Optional[npt.NDArray[Any]]:
        """Decodes raster data.

        Args:
            file_api: the file API to read from
            bounds: the bounds of the raster to read

        Returns:
            the raster data, or None if no image content is found
        """
        with file_api.open(self.fname, "rb") as f:
            with rasterio.open(f) as src:
                transform = src.transform
                x_resolution = transform.a
                y_resolution = transform.e
                offset = (
                    int(transform.c / x_resolution),
                    int(transform.f / y_resolution),
                )
                # bounds is in global pixel coordinates.
                # We first convert that to pixels relative to top-left of the raster.
                relative_bounds = [
                    bounds[0] - offset[0],
                    bounds[1] - offset[1],
                    bounds[2] - offset[0],
                    bounds[3] - offset[1],
                ]
                if (
                    relative_bounds[2] < 0
                    or relative_bounds[3] < 0
                    or relative_bounds[0] >= src.width
                    or relative_bounds[1] >= src.height
                ):
                    return None
                # Now get the actual pixels we will read, which must be contained in
                # the GeoTIFF.
                # Padding is (before_x, before_y, after_x, after_y) and will be used to
                # pad the output back to the originally requested bounds.
                padding = [0, 0, 0, 0]
                if relative_bounds[0] < 0:
                    padding[0] = -relative_bounds[0]
                    relative_bounds[0] = 0
                if relative_bounds[1] < 0:
                    padding[1] = -relative_bounds[1]
                    relative_bounds[1] = 0
                if relative_bounds[2] > src.width:
                    padding[2] = relative_bounds[2] - src.width
                    relative_bounds[2] = src.width
                if relative_bounds[3] > src.height:
                    padding[3] = relative_bounds[3] - src.height
                    relative_bounds[3] = src.height

                window = rasterio.windows.Window(
                    relative_bounds[0],
                    relative_bounds[1],
                    relative_bounds[2] - relative_bounds[0],
                    relative_bounds[3] - relative_bounds[1],
                )
                array = src.read(window=window)
                array = np.pad(
                    array, ((0, 0), (padding[1], padding[3]), (padding[0], padding[2]))
                )
                return array

    def get_raster_bounds(self, file_api: FileAPI) -> PixelBounds:
        with file_api.open(self.fname, "rb") as f:
            with rasterio.open(f) as src:
                transform = src.transform
                x_resolution = transform.a
                y_resolution = transform.e
                offset = (
                    int(transform.c / x_resolution),
                    int(transform.f / y_resolution),
                )
                return (
                    offset[0],
                    offset[1],
                    offset[0] + src.width,
                    offset[1] + src.height,
                )

    @staticmethod
    def from_config(name: str, config: dict[str, Any]) -> "GeotiffRasterFormat":
        return GeotiffRasterFormat(
            block_size=config.get("block_size", 512),
        )


def load_raster_format(config: RasterFormatConfig) -> RasterFormat:
    cls = RasterFormats.get_class(config.name)
    return cls.from_config(config.name, config.config_dict)
