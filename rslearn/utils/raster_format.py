from typing import Any, BinaryIO

import affine
import numpy as np
import numpy.typing as npt
import rasterio
from PIL import Image

from rslearn.config import RasterFormatConfig
from rslearn.const import TILE_SIZE

from .geometry import Projection


class RasterFormat:
    def encode_raster(
        self,
        f: BinaryIO,
        projection: Projection,
        bounds: tuple[int, int, int, int],
        image: npt.NDArray[Any],
    ) -> None:
        """Encodes a raster tile.

        Args:
            f: the file object to write the image bytes to
            projection: the projection of the image
            bounds: the bounds of the image in the projection
            image: the numpy array image content
        """
        raise NotImplementedError

    def decode_raster(self, f: BinaryIO) -> npt.NDArray[Any]:
        """Decodes a raster tile."""
        raise NotImplementedError

    def get_extension(self):
        """Returns a suitable filename extension for this format."""
        raise NotImplementedError


class GeotiffRasterFormat(RasterFormat):
    def encode_raster(
        self,
        f: BinaryIO,
        projection: Projection,
        bounds: tuple[int, int, int, int],
        image: npt.NDArray[Any],
    ) -> None:
        """Encodes a raster tile.

        Args:
            f: the file object to write the image bytes to
            projection: the projection of the image
            bounds: the bounds of the image in the projection
            image: the numpy array image content
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
            "width": image.shape[2],
            "height": image.shape[1],
            "count": image.shape[0],
            "dtype": image.dtype.name,
            "crs": crs,
            "transform": transform,
        }
        if image.shape[2] > TILE_SIZE and image.shape[1] > TILE_SIZE:
            profile["tiled"] = True
            profile["blockxsize"] = TILE_SIZE
            profile["blockysize"] = TILE_SIZE
        with rasterio.open(f, "w", **profile) as dst:
            dst.write(image)

    def decode_raster(self, f: BinaryIO) -> npt.NDArray[Any]:
        """Decodes a raster tile."""
        with rasterio.open(f) as src:
            return src.read()

    def get_extension(self):
        return "tif"

    @staticmethod
    def from_config(name: str, config: dict[str, Any]) -> "GeotiffRasterFormat":
        return GeotiffRasterFormat()


class ImageRasterFormat(RasterFormat):
    def __init__(self, format: str):
        self.format = format

    def encode_raster(
        self,
        f: BinaryIO,
        projection: Projection,
        bounds: tuple[int, int, int, int],
        image: npt.NDArray[Any],
    ) -> None:
        """Encodes a raster tile.

        Args:
            f: the file object to write the image bytes to
            projection: the projection of the image
            bounds: the bounds of the image in the projection
            image: the numpy array image content
        """
        image = image.transpose(1, 2, 0)

        if self.format == RasterFormat.PNG:
            Image.fromarray(image).save(f, format="PNG")

        elif self.format == RasterFormat.JPEG:
            Image.fromarray(image).save(f, format="JPEG")

        else:
            raise ValueError(f"unknown image format {self.format}")

    def decode_raster(self, f: BinaryIO) -> npt.NDArray[Any]:
        """Decodes a raster tile."""
        if self.format == RasterFormat.PNG:
            return np.array(Image.open(f, format="PNG")).transpose(2, 0, 1)

        if self.format == RasterFormat.JPEG:
            return np.array(Image.open(f, format="JPEG")).transpose(2, 0, 1)

        raise ValueError(f"unknown image format {self.format}")

    def get_extension(self):
        return self.format

    @staticmethod
    def from_config(name: str, config: dict[str, Any]) -> "ImageRasterFormat":
        return ImageRasterFormat(name)


registry = {
    "geotiff": GeotiffRasterFormat,
    "png": ImageRasterFormat,
    "jpeg": ImageRasterFormat,
}


def load_raster_format(config: RasterFormatConfig) -> RasterFormat:
    return registry[config.name].from_config(config.name, config.config_dict)
