"""Abstract RasterFormat class."""

import hashlib
import json
from typing import Any, BinaryIO

import affine
import numpy as np
import numpy.typing as npt
import rasterio
from PIL import Image
from rasterio.crs import CRS
from rasterio.enums import Resampling
from upath import UPath

from rslearn.const import TILE_SIZE
from rslearn.log_utils import get_logger
from rslearn.utils.fsspec import open_rasterio_upath_reader, open_rasterio_upath_writer

from .geometry import PixelBounds, Projection

logger = get_logger(__name__)


def get_bandset_dirname(bands: list[str]) -> str:
    """Get the directory name that should be used to store the given group of bands."""
    # We try to use a human-readable name with underscore as the delimiter, but if that
    # isn't straightforward then we use hash instead.
    if any(["_" in band for band in bands]):
        # In this case we hash the JSON representation of the bands.
        return hashlib.sha256(json.dumps(bands).encode()).hexdigest()
    dirname = "_".join(bands)
    if len(dirname) > 64:
        # Previously we simply joined the bands, but this can result in directory name
        # that is too long. In this case, now we use hash instead.
        # We use a different code path here where we hash the initial directory name
        # instead of the JSON, for historical reasons (to maintain backwards
        # compatibility).
        dirname = hashlib.sha256(dirname.encode()).hexdigest()
    return dirname


def get_raster_projection_and_bounds_from_transform(
    crs: CRS, transform: affine.Affine, width: int, height: int
) -> tuple[Projection, PixelBounds]:
    """Determine Projection and bounds from the specified CRS and transform.

    Args:
        crs: the coordinate reference system.
        transform: corresponding affine transform matrix.
        width: the array width
        height: the array height

    Returns:
        a tuple (projection, bounds).
    """
    x_resolution = transform.a
    y_resolution = transform.e
    projection = Projection(crs, x_resolution, y_resolution)
    offset = (
        int(round(transform.c / x_resolution)),
        int(round(transform.f / y_resolution)),
    )
    bounds = (offset[0], offset[1], offset[0] + width, offset[1] + height)
    return (projection, bounds)


def get_raster_projection_and_bounds(
    raster: rasterio.DatasetReader,
) -> tuple[Projection, PixelBounds]:
    """Determine the Projection and bounds of the specified raster.

    Args:
        raster: the raster dataset opened with rasterio.

    Returns:
        a tuple (projection, bounds).
    """
    return get_raster_projection_and_bounds_from_transform(
        raster.crs, raster.transform, raster.width, raster.height
    )


def get_transform_from_projection_and_bounds(
    projection: Projection, bounds: PixelBounds
) -> affine.Affine:
    """Get the affine transform that corresponds to the given projection and bounds.

    Args:
        projection: the projection. Only the resolutions are used.
        bounds: the bounding box. Only the top-left corner is used.
    """
    return affine.Affine(
        projection.x_resolution,
        0,
        bounds[0] * projection.x_resolution,
        0,
        projection.y_resolution,
        bounds[1] * projection.y_resolution,
    )


def adjust_projection_and_bounds_for_array(
    projection: Projection, bounds: PixelBounds, array: npt.NDArray
) -> tuple[Projection, PixelBounds]:
    """Adjust the projection and bounds to correspond to the resolution of the array.

    The returned projection and bounds cover the same spatial extent as the inputs, but
    are updated so that the width and height match that of the array.

    Args:
        projection: the original projection.
        bounds: the original bounds.
        array: the CHW array for which to compute an updated projection and bounds. The
            returned bounds will have the same width and height as this array.

    Returns:
        a tuple of adjusted (projection, bounds)
    """
    if array.shape[2] == (bounds[2] - bounds[0]) and array.shape[1] == (
        bounds[3] - bounds[1]
    ):
        return (projection, bounds)

    x_factor = array.shape[2] / (bounds[2] - bounds[0])
    y_factor = array.shape[1] / (bounds[3] - bounds[1])
    adjusted_projection = Projection(
        projection.crs,
        projection.x_resolution / x_factor,
        projection.y_resolution / y_factor,
    )
    adjusted_bounds = (
        round(bounds[0] * x_factor),
        round(bounds[1] * y_factor),
        round(bounds[0] * x_factor) + array.shape[2],
        round(bounds[1] * y_factor) + array.shape[1],
    )
    return (adjusted_projection, adjusted_bounds)


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
        self,
        path: UPath,
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
    ) -> npt.NDArray[Any]:
        """Decodes raster data.

        Args:
            path: the directory to read from
            projection: the projection to read the raster in.
            bounds: the bounds to read in the given projection.
            resampling: resampling method to use in case resampling is needed.

        Returns:
            the raster data
        """
        raise NotImplementedError

    def encode_stacked_raster(
        self,
        path: UPath,
        projection: Projection,
        bounds: PixelBounds,
        array: npt.NDArray[Any],
    ) -> None:
        """Encodes a stacked raster array into a single file.

        This is used for single-file materialization where all item groups in a layer
        are written to one file rather than separate files per item group.

        Args:
            path: the directory to write to
            projection: the projection of the raster data
            bounds: the bounds of the raster data in the projection
            array: the raster data with shape (T, C, H, W) where T is the number of
                item groups and C is the number of channels per group
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support stacked raster encoding"
        )

    def decode_stacked_raster(
        self,
        path: UPath,
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
    ) -> npt.NDArray[Any]:
        """Decodes stacked raster data.

        This is used for single-file materialization where all item groups in a layer
        are stored in one file.

        Args:
            path: the directory to read from
            projection: the projection to read the raster in.
            bounds: the bounds to read in the given projection.
            resampling: resampling method to use in case resampling is needed.

        Returns:
            the raster data with shape (T, C, H, W) where T is the number of item
            groups and C is the number of channels per group
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support stacked raster decoding"
        )

    @staticmethod
    def from_config(name: str, config: dict[str, Any]) -> "RasterFormat":
        """Create a RasterFormat from a config dict.

        Args:
            name: the name of this format
            config: the config dict

        Returns:
            the RasterFormat instance
        """
        raise NotImplementedError


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
        # Write metadata about the projection that we are writing under.
        # We also save dtype and number of bands so we can return correct shape when
        # there are no intersecting tiles.
        with (path / "metadata.json").open("w") as f:
            json.dump(
                {
                    "projection": projection.serialize(),
                    "dtype": array.dtype.name,
                    "num_bands": array.shape[0],
                },
                f,
            )

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
        self,
        path: UPath,
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
    ) -> npt.NDArray[Any]:
        """Decodes raster data.

        Args:
            path: the directory to read from
            projection: the projection to read the raster in.
            bounds: the bounds to read in the given projection.
            resampling: resampling method to use in case resampling is needed.

        Returns:
            the raster data
        """
        # Verify that the source data has the same projection as the requested one.
        # ImageTileRasterFormat currently does not support re-projecting.
        with (path / "metadata.json").open() as f:
            image_metadata = json.load(f)
        source_data_projection = Projection.deserialize(image_metadata["projection"])
        if source_data_projection != projection:
            raise NotImplementedError(
                "not implemented to re-project source data "
                + f"(source projection {source_data_projection} does not match requested projection {projection})"
            )

        extension = self.get_extension()

        # Load tiles one at a time.
        start_tile = (bounds[0] // self.tile_size, bounds[1] // self.tile_size)
        end_tile = (
            (bounds[2] - 1) // self.tile_size + 1,
            (bounds[3] - 1) // self.tile_size + 1,
        )
        dst_shape = (
            image_metadata["num_bands"],
            bounds[3] - bounds[1],
            bounds[2] - bounds[0],
        )
        dst = np.zeros(dst_shape, dtype=image_metadata["dtype"])
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


class GeotiffRasterFormat(RasterFormat):
    """A raster format that uses one big, tiled GeoTIFF with small block size."""

    fname = "geotiff.tif"

    def __init__(
        self,
        block_size: int = TILE_SIZE,
        always_enable_tiling: bool = False,
        geotiff_options: dict[str, Any] = {},
    ):
        """Initializes a GeotiffRasterFormat.

        Args:
            block_size: the block size to use in the output GeoTIFF
            always_enable_tiling: whether to always enable tiling when creating
                GeoTIFFs. The default is False so that tiling is only used if the size
                of the GeoTIFF exceeds the block_size on either dimension. If True,
                then tiling is always enabled (cloud-optimized GeoTIFF).
            geotiff_options: other options to pass to rasterio.open (for writes).
        """
        self.block_size = block_size
        self.always_enable_tiling = always_enable_tiling
        self.geotiff_options = geotiff_options

    def encode_raster(
        self,
        path: UPath,
        projection: Projection,
        bounds: PixelBounds,
        array: npt.NDArray[Any],
        fname: str | None = None,
        nodata_val: int | float | None = None,
    ) -> None:
        """Encodes raster data.

        Args:
            path: the directory to write to
            projection: the projection of the raster data
            bounds: the bounds of the raster data in the projection
            array: the raster data
            fname: override the filename to save as
            nodata_val: set the nodata value when writing the raster.
        """
        if fname is None:
            fname = self.fname

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
        # Set nodata_val if provided.
        if nodata_val is not None:
            profile["nodata"] = nodata_val

        profile.update(self.geotiff_options)

        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Writing geotiff to {path / fname}")
        with open_rasterio_upath_writer(path / fname, **profile) as dst:
            dst.write(array)

    def decode_raster(
        self,
        path: UPath,
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
        fname: str | None = None,
        nodata_val: int | float | None = None,
    ) -> npt.NDArray[Any]:
        """Decodes raster data.

        Args:
            path: the directory to read from
            projection: the projection to read the raster in.
            bounds: the bounds to read in the given projection.
            resampling: resampling method to use in case resampling is needed.
            fname: override the filename to read from
            nodata_val: override the nodata value in the raster when reading. Pixels in
                bounds that are not present in the source raster will be initialized to
                this value. Note that, if the raster specifies a nodata value, and
                some source pixels have that value, they will still be read under their
                original value; overriding the nodata value is primarily useful if the
                user wants out of bounds pixels to have a different value from the
                source pixels, e.g. if the source data has background and foreground
                classes (with background being nodata) but we want to read it in a
                different projection and have out of bounds pixels be a third "invalid"
                value.

        Returns:
            the raster data
        """
        if fname is None:
            fname = self.fname

        # Construct the transform to use for the warped dataset.
        wanted_transform = get_transform_from_projection_and_bounds(projection, bounds)
        with open_rasterio_upath_reader(path / fname) as src:
            with rasterio.vrt.WarpedVRT(
                src,
                crs=projection.crs,
                transform=wanted_transform,
                width=bounds[2] - bounds[0],
                height=bounds[3] - bounds[1],
                resampling=resampling,
                src_nodata=nodata_val,
            ) as vrt:
                return vrt.read()

    def get_raster_bounds(self, path: UPath) -> PixelBounds:
        """Returns the bounds of the stored raster.

        Args:
            path: the directory where the raster data was written

        Returns:
            the PixelBounds of the raster
        """
        with open_rasterio_upath_reader(path / self.fname) as src:
            _, bounds = get_raster_projection_and_bounds(src)
            return bounds

    stacked_fname = "stacked.tif"
    stacked_metadata_fname = "stacked_metadata.json"

    def encode_stacked_raster(
        self,
        path: UPath,
        projection: Projection,
        bounds: PixelBounds,
        array: npt.NDArray[Any],
    ) -> None:
        """Encodes a stacked raster array into a single GeoTIFF.

        The TCHW array is flattened to (T*C)xHxW for storage in the GeoTIFF.
        Metadata about the stacking dimensions is stored in a separate JSON file.

        Args:
            path: the directory to write to
            projection: the projection of the raster data
            bounds: the bounds of the raster data in the projection
            array: the raster data with shape (T, C, H, W) where T is the number of
                item groups and C is the number of channels per group
        """
        if array.ndim != 4:
            raise ValueError(f"Expected 4D array (T, C, H, W), got shape {array.shape}")

        num_groups, num_channels, height, width = array.shape

        # Flatten TCHW to (T*C)HW for storage
        flattened_array = array.reshape(num_groups * num_channels, height, width)

        # Write the stacked GeoTIFF using encode_raster
        self.encode_raster(
            path, projection, bounds, flattened_array, fname=self.stacked_fname
        )

        # Write metadata about the stacking dimensions
        metadata = {
            "num_groups": num_groups,
            "num_channels": num_channels,
        }
        with (path / self.stacked_metadata_fname).open("w") as f:
            json.dump(metadata, f)

    def decode_stacked_raster(
        self,
        path: UPath,
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
    ) -> npt.NDArray[Any]:
        """Decodes stacked raster data from a GeoTIFF.

        Args:
            path: the directory to read from
            projection: the projection to read the raster in.
            bounds: the bounds to read in the given projection.
            resampling: resampling method to use in case resampling is needed.

        Returns:
            the raster data with shape (T, C, H, W) where T is the number of item
            groups and C is the number of channels per group
        """
        # Read the metadata
        with (path / self.stacked_metadata_fname).open() as f:
            metadata = json.load(f)
        num_groups = metadata["num_groups"]
        num_channels = metadata["num_channels"]

        # Read the stacked GeoTIFF
        wanted_transform = get_transform_from_projection_and_bounds(projection, bounds)
        with open_rasterio_upath_reader(path / self.stacked_fname) as src:
            with rasterio.vrt.WarpedVRT(
                src,
                crs=projection.crs,
                transform=wanted_transform,
                width=bounds[2] - bounds[0],
                height=bounds[3] - bounds[1],
                resampling=resampling,
            ) as vrt:
                flattened_array = vrt.read()

        # Reshape from (T*C)HW back to TCHW
        height = flattened_array.shape[1]
        width = flattened_array.shape[2]
        return flattened_array.reshape(num_groups, num_channels, height, width)

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
        if "geotiff_options" in config:
            kwargs["geotiff_options"] = config["geotiff_options"]
        return GeotiffRasterFormat(**kwargs)


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

        # Since the image file doesn't include the georeferencing, we store it in an
        # auxiliary metadata file.
        with (path / "metadata.json").open("w") as f:
            json.dump(
                {
                    "projection": projection.serialize(),
                    "bounds": bounds,
                },
                f,
            )

    def decode_raster(
        self,
        path: UPath,
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
    ) -> npt.NDArray[Any]:
        """Decodes raster data.

        Args:
            path: the directory to read from
            projection: the projection to read the raster in.
            bounds: the bounds to read in the given projection.
            resampling: resampling method to use in case resampling is needed.

        Returns:
            the raster data
        """
        # Try to get the bounds of the saved image from the metadata file.
        # In old versions, the file may be missing the projection key.
        metadata_fname = path / "metadata.json"
        with metadata_fname.open() as f:
            image_metadata = json.load(f)

        image_bounds = image_metadata["bounds"]

        # If the projection key is set, verify that it matches the requested projection
        # since SingleImageRasterFormat currently does not support re-projecting.
        if "projection" in image_metadata:
            source_data_projection = Projection.deserialize(
                image_metadata["projection"]
            )
            if projection != source_data_projection:
                raise NotImplementedError(
                    "not implemented to re-project source data "
                    + f"(source projection {source_data_projection} does not match requested projection {projection})"
                )

        image_fname = path / ("image." + self.get_extension())
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
