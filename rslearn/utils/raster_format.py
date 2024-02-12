import io
from typing import Any

import affine
import numpy.typing as npt
import rasterio
from PIL import Image

from .geometry import Projection


class RasterFormat:
    def encode_image(
        self, projection: Projection, tile: tuple[int, int], image: npt.NDArray[Any]
    ) -> bytes:
        """Encodes an image of a tile to bytes.

        Args:
            projection: the projection of the image
            tile: the tile that the image corresponds to
            image: the numpy array image content

        Returns:
            encoded image bytes
        """
        raise NotImplementedError

    def get_extension(self):
        """Returns a suitable filename extension for this format."""
        raise NotImplementedError


class GeotiffRasterFormat(RasterFormat):
    def encode_image(
        self, projection: Projection, tile: tuple[int, int], image: npt.NDArray[Any]
    ) -> bytes:
        """Encodes an image of a tile to bytes.

        Args:
            projection: the projection of the image
            tile: the tile that the image corresponds to
            image: the numpy array image content

        Returns:
            encoded image bytes
        """
        crs = projection.crs
        transform = affine.Affine(
            projection.resolution,
            0,
            tile[0] * image.shape[2] * projection.resolution,
            0,
            -projection.resolution,
            tile[1] * image.shape[1] * -projection.resolution,
        )
        buf = io.BytesIO()
        with rasterio.open(
            buf,
            "w",
            driver="GTiff",
            compress="lzw",
            width=image.shape[2],
            height=image.shape[1],
            count=image.shape[0],
            dtype=image.dtype.name,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(image)
        return buf.getvalue()

    def get_extension(self):
        return "tif"

    @staticmethod
    def from_config(name: str, config: dict[str, Any]) -> "GeotiffRasterFormat":
        return GeotiffRasterFormat()


class ImageRasterFormat(RasterFormat):
    def __init__(self, format: str):
        self.format = format

    def encode_image(
        self, projection: Projection, tile: tuple[int, int], image: npt.NDArray[Any]
    ) -> bytes:
        """Encodes an image of a tile to bytes.

        Args:
            projection: the projection of the image
            tile: the tile that the image corresponds to
            image: the numpy array image content

        Returns:
            encoded image bytes
        """
        if self.format == RasterFormat.PNG:
            buf = io.BytesIO()
            Image.fromarray(image).save(buf, format="PNG")
            return buf.getvalue()

        if self.format == RasterFormat.JPEG:
            buf = io.BytesIO()
            Image.fromarray(image).save(buf, format="JPEG")
            return buf.getvalue()

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
