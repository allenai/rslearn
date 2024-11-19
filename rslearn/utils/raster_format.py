"""Abstract RasterFormat class."""

import json
from typing import Any, BinaryIO

import affine
import numpy as np
import numpy.typing as npt
import rasterio
from class_registry import ClassRegistry
from PIL import Image
from upath import UPath

from rslearn.config import RasterFormatConfig
from rslearn.const import TILE_SIZE
from rslearn.log_utils import get_logger
from rslearn.utils.fsspec import open_rasterio_upath_reader, open_rasterio_upath_writer

from .geometry import PixelBounds, Projection

RasterFormats = ClassRegistry()
logger = get_logger(__name__)


class RasterFormat:
    """An abstract class for writing raster data.

    Implementations of RasterFormat should support reading and writing raster data in
    a UPath. Raster data is a CxHxW numpy array.
    """

    def encode_raster(
        self,
        path: UPath,
        projection: Projection,
        bounds: PixelBounds,
        array: npt.NDArray[Any],
    ) -> None:
        """Encodes raster data.

        Args:
            path: the directory to write to
            projection: the projection of the raster data
            bounds: the bounds of the raster data in the projection
            array: the raster data
        """
        raise NotImplementedError

    def decode_raster(
        self, path: UPath, bounds: PixelBounds
    ) -> npt.NDArray[Any] | None:
        """Decodes raster data.

        Args:
            path: the directory to read from
            bounds: the bounds of the raster to read

        Returns:
            the raster data, or None if no image content is found
        """
        raise NotImplementedError


@RasterFormats.register("image_tile")
class ImageTileRasterFormat(RasterFormat):
    """A RasterFormat that stores data in image tiles corresponding to grid cells.

    A tile size defines the grid size in pixels. One file is created for each grid cell
    that the raster intersects. The image format is configurable. The images are named
    by their (possibly negative) column and row along the grid.
    """

    def __init__(self, format: str, tile_size: int = TILE_SIZE):
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
        """Encodes a single tile to a file.

        Args:
            f: the file object to write to
            projection: the projection (used for GeoTIFF metadata)
            bounds: the bounds in the projection (used for GeoTIFF metadata)
            array: the raster data at this tile
        """
        if self.format in ["png", "jpeg"]:
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
        """Decodes a single tile from a file.

        Args:
            f: the file object to read from
        """
        if self.format in ["png", "jpeg"]:
            array = np.array(Image.open(f, formats=[self.format.upper()]))
            if len(array.shape) == 2:
                array = array[:, :, None]
            return array.transpose(2, 0, 1)

        elif self.format == "geotiff":
            with rasterio.open(f) as src:
                return src.read()

    def encode_raster(
        self,
        path: UPath,
        projection: Projection,
        bounds: PixelBounds,
        array: npt.NDArray[Any],
    ) -> None:
        """Encodes raster data.

        Args:
            path: the directory to write to
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

        path.mkdir(parents=True, exist_ok=True)
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
                fname = path / f"{col}_{row}.{extension}"
                with fname.open("wb") as f:
                    self.encode_tile(f, projection, cur_bounds, cur_array)

    def decode_raster(
        self, path: UPath, bounds: PixelBounds
    ) -> npt.NDArray[Any] | None:
        """Decodes raster data.

        Args:
            path: the directory to read from
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
                fname = path / f"{col}_{row}.{extension}"
                if not fname.exists():
                    continue
                with fname.open("rb") as f:
                    src = self.decode_tile(f)

                if dst is None:
                    dst = np.zeros(
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
        """Returns the extension to use based on the configured image format."""
        if self.format == "png":
            return "png"
        elif self.format == "jpeg":
            return "jpg"
        elif self.format == "geotiff":
            return "tif"
        raise ValueError(f"unknown image format {self.format}")

    @staticmethod
    def from_config(name: str, config: dict[str, Any]) -> "ImageTileRasterFormat":
        """Create a ImageTileRasterFormat from a config dict.

        Args:
            name: the name of this format
            config: the config dict
        """
        return ImageTileRasterFormat(
            format=config.get("format", "geotiff"),
            tile_size=config.get("tile_size", 512),
        )


@RasterFormats.register("geotiff")
class GeotiffRasterFormat(RasterFormat):
    """A raster format that uses one big, tiled GeoTIFF with small block size."""

    fname = "geotiff.tif"

    def __init__(self, block_size: int = TILE_SIZE, always_enable_tiling: bool = False):
        """Initializes a GeotiffRasterFormat.

        Args:
            block_size: the block size to use in the output GeoTIFF
            always_enable_tiling: whether to always enable tiling when creating
                GeoTIFFs. The default is False so that tiling is only used if the size
                of the GeoTIFF exceeds the block_size on either dimension. If True,
                then tiling is always enabled (cloud-optimized GeoTIFF).
        """
        self.block_size = block_size
        self.always_enable_tiling = always_enable_tiling

    def encode_raster(
        self,
        path: UPath,
        projection: Projection,
        bounds: PixelBounds,
        array: npt.NDArray[Any],
    ) -> None:
        """Encodes raster data.

        Args:
            path: the directory to write to
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
            # Configure rasterio to use BIGTIFF when needed to write large files.
            # Without BIGTIFF it is up to 4 GB and trying to write larger files would
            # result in an error.
            "BIGTIFF": "IF_SAFER",
        }
        if (
            array.shape[2] > self.block_size
            or array.shape[1] > self.block_size
            or self.always_enable_tiling
        ):
            profile["tiled"] = True
            profile["blockxsize"] = self.block_size
            profile["blockysize"] = self.block_size

        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing geotiff to {path / self.fname}")
        with open_rasterio_upath_writer(path / self.fname, **profile) as dst:
            dst.write(array)

    def decode_raster(self, path: UPath, bounds: PixelBounds) -> npt.NDArray[Any]:
        """Decodes raster data.

        Args:
            path: the directory to read from
            bounds: the bounds of the raster to read

        Returns:
            the raster data, or None if no image content is found
        """
        with open_rasterio_upath_reader(path / self.fname) as src:
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

            # Make sure the requested bounds intersects the raster, otherwise the
            # windowed read cannot be performed.
            if (
                relative_bounds[2] < 0
                or relative_bounds[3] < 0
                or relative_bounds[0] >= src.width
                or relative_bounds[1] >= src.height
            ):
                # Assume all of the bands have the same dtype, so just use first
                # one (src.dtypes is list of dtype per band).
                return np.zeros(
                    (src.count, bounds[3] - bounds[1], bounds[2] - bounds[0]),
                    dtype=src.dtypes[0],
                )

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

    def get_raster_bounds(self, path: UPath) -> PixelBounds:
        """Returns the bounds of the stored raster.

        Args:
            path: the directory where the raster data was written

        Returns:
            the PixelBounds of the raster
        """
        with open_rasterio_upath_reader(path / self.fname) as src:
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
        """Create a GeotiffRasterFormat from a config dict.

        Args:
            name: the name of this format
            config: the config dict

        Returns:
            the GeotiffRasterFormat
        """
        kwargs = {}
        if "block_size" in config:
            kwargs["block_size"] = config["block_size"]
        if "always_enable_tiling" in config:
            kwargs["always_enable_tiling"] = config["always_enable_tiling"]
        return GeotiffRasterFormat(**kwargs)


@RasterFormats.register("single_image")
class SingleImageRasterFormat(RasterFormat):
    """A raster format that produces a single image called image.png/jpg.

    Primarily for ease-of-use with external tools that don't support georeferenced
    images and would rather have everything in pixel coordinate system.
    """

    def __init__(self, format: str = "png"):
        """Initialize a SingleImageRasterFormat.

        Args:
            format: the format, either png or jpeg
        """
        self.format = format

    def get_extension(self) -> str:
        """Get the filename extension to use when storing the image.

        Returns:
            the string filename extension, e.g. png or jpg
        """
        if self.format == "png":
            return "png"
        elif self.format == "jpeg":
            return "jpg"
        raise ValueError(f"unknown image format {self.format}")

    def encode_raster(
        self,
        path: UPath,
        projection: Projection,
        bounds: PixelBounds,
        array: npt.NDArray[Any],
    ) -> None:
        """Encodes raster data.

        Args:
            path: the directory to write to
            projection: the projection of the raster data
            bounds: the bounds of the raster data in the projection
            array: the raster data
        """
        path.mkdir(parents=True, exist_ok=True)
        fname = path / ("image." + self.get_extension())
        with fname.open("wb") as f:
            array = array.transpose(1, 2, 0)
            if array.shape[2] == 1:
                array = array[:, :, 0]
            Image.fromarray(array).save(f, format=self.format.upper())
        with (path / "metadata.json").open("w") as f:
            json.dump(
                {
                    "bounds": bounds,
                },
                f,
            )

    def decode_raster(
        self, path: UPath, bounds: PixelBounds
    ) -> npt.NDArray[Any] | None:
        """Decodes raster data.

        Args:
            path: the directory to read from
            bounds: the bounds of the raster to read

        Returns:
            the raster data, or None if no image content is found
        """
        image_fname = path / ("image." + self.get_extension())
        metadata_fname = path / "metadata.json"
        if metadata_fname.exists():
            with metadata_fname.open() as f:
                image_bounds = json.load(f)["bounds"]
        else:
            # Backwards compatibility -- assume that requested bounds matches the window bounds.
            image_bounds = bounds

        with image_fname.open("rb") as f:
            array = np.array(Image.open(f, formats=[self.format.upper()]))

        if len(array.shape) == 2:
            array = array[:, :, None]
        array = array.transpose(2, 0, 1)

        if bounds == image_bounds:
            return array

        # Need to extract relevant portion of image.
        dst = np.zeros(
            (array.shape[0], bounds[3] - bounds[1], bounds[2] - bounds[0]),
            dtype=array.dtype,
        )
        src_col_offset = max(bounds[0] - image_bounds[0], 0)
        src_row_offset = max(bounds[1] - image_bounds[1], 0)
        dst_col_offset = max(image_bounds[0] - bounds[0], 0)
        dst_row_offset = max(image_bounds[1] - bounds[1], 0)
        col_overlap = min(
            array.shape[2] - src_col_offset, dst.shape[2] - dst_col_offset
        )
        row_overlap = min(
            array.shape[1] - src_row_offset, dst.shape[1] - dst_row_offset
        )
        dst[
            :,
            dst_row_offset : dst_row_offset + row_overlap,
            dst_col_offset : dst_col_offset + col_overlap,
        ] = array[
            :,
            src_row_offset : src_row_offset + row_overlap,
            src_col_offset : src_col_offset + col_overlap,
        ]
        return dst

    @staticmethod
    def from_config(name: str, config: dict[str, Any]) -> "SingleImageRasterFormat":
        """Create a SingleImageRasterFormat from a config dict.

        Args:
            name: the name of this format
            config: the config dict

        Returns:
            the SingleImageRasterFormat
        """
        kwargs = {}
        if "format" in config:
            kwargs["format"] = config["format"]
        return SingleImageRasterFormat(**kwargs)


def load_raster_format(config: RasterFormatConfig) -> RasterFormat:
    """Loads a RasterFormat from a RasterFormatConfig.

    Args:
        config: the RasterFormatConfig configuration object specifying the
            RasterFormat.

    Returns:
        the loaded RasterFormat implementation
    """
    cls = RasterFormats.get_class(config.name)
    return cls.from_config(config.name, config.config_dict)
