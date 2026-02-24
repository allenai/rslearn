"""Abstract RasterFormat class."""

import hashlib
import json
from datetime import datetime
from typing import Any, BinaryIO

import affine
import einops
import numpy as np
import numpy.typing as npt
import pydantic
import rasterio
from PIL import Image
from rasterio.crs import CRS
from rasterio.enums import Resampling
from upath import UPath

from rslearn.const import TILE_SIZE
from rslearn.log_utils import get_logger
from rslearn.utils.array import copy_spatial_array
from rslearn.utils.fsspec import open_rasterio_upath_reader, open_rasterio_upath_writer
from rslearn.utils.raster_array import RasterArray

from .geometry import PixelBounds, Projection

logger = get_logger(__name__)

METADATA_FNAME = "metadata.json"


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
    a UPath. Raster data is represented as a RasterArray (C, T, H, W).
    """

    def encode_raster(
        self,
        path: UPath,
        projection: Projection,
        bounds: PixelBounds,
        raster: RasterArray,
    ) -> None:
        """Encodes raster data.

        Args:
            path: the directory to write to
            projection: the projection of the raster data
            bounds: the bounds of the raster data in the projection
            raster: the raster data
        """
        raise NotImplementedError

    def decode_raster(
        self,
        path: UPath,
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
    ) -> RasterArray:
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


class ImageTileRasterMetadata(pydantic.BaseModel):
    """Metadata sidecar for ImageTileRasterFormat."""

    projection: dict[str, Any]
    dtype: str
    num_bands: int
    timestamps: list[tuple[datetime, datetime]] | None = None


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
        raster: RasterArray,
    ) -> None:
        """Encodes raster data.

        Args:
            path: the directory to write to
            projection: the projection of the raster data
            bounds: the bounds of the raster data in the projection
            raster: the raster data (CTHW RasterArray, T must be 1)
        """
        array = raster.get_chw_array()

        metadata = ImageTileRasterMetadata(
            projection=projection.serialize(),
            dtype=array.dtype.name,
            num_bands=array.shape[0],
            timestamps=raster.timestamps,
        )
        with (path / METADATA_FNAME).open("w") as f:
            f.write(metadata.model_dump_json())

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
    ) -> RasterArray:
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
        with (path / METADATA_FNAME).open() as f:
            image_metadata = ImageTileRasterMetadata.model_validate_json(f.read())
        source_data_projection = Projection.deserialize(image_metadata.projection)
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
            image_metadata.num_bands,
            bounds[3] - bounds[1],
            bounds[2] - bounds[0],
        )
        dst = np.zeros(dst_shape, dtype=image_metadata.dtype)
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

                copy_spatial_array(
                    src,
                    dst,
                    src_offset=(cur_col_off, cur_row_off),
                    dst_offset=(bounds[0], bounds[1]),
                )

        # Wrap as CTHW with T=1.
        return RasterArray(
            array=dst[:, np.newaxis, :, :],
            timestamps=image_metadata.timestamps,
        )

    def get_extension(self) -> str:
        """Returns the extension to use based on the configured image format."""
        if self.format == "png":
            return "png"
        elif self.format == "jpeg":
            return "jpg"
        elif self.format == "geotiff":
            return "tif"
        raise ValueError(f"unknown image format {self.format}")


class GeotiffRasterMetadata(pydantic.BaseModel):
    """Metadata sidecar for GeotiffRasterFormat.

    All fields are optional for backward compatibility with legacy metadata files.
    """

    num_channels: int | None = None
    num_timesteps: int | None = None
    timestamps: list[tuple[datetime, datetime]] | None = None


class GeotiffRasterFormat(RasterFormat):
    """A raster format that uses one big, tiled GeoTIFF with small block size."""

    fname = "geotiff.tif"

    @staticmethod
    def encode_metadata(path: UPath, metadata: GeotiffRasterMetadata) -> None:
        """Write a GeotiffRasterMetadata sidecar to *path* / metadata.json."""
        with (path / METADATA_FNAME).open("w") as f:
            f.write(metadata.model_dump_json())

    @staticmethod
    def decode_metadata(path: UPath) -> GeotiffRasterMetadata | None:
        """Read the GeotiffRasterMetadata sidecar, or return None if absent."""
        metadata_path = path / METADATA_FNAME
        if not metadata_path.exists():
            return None
        with metadata_path.open() as f:
            return GeotiffRasterMetadata.model_validate_json(f.read())

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
        raster: RasterArray,
        fname: str | None = None,
        nodata_val: int | float | None = None,
    ) -> None:
        """Encodes raster data.

        Supports multi-timestep data (T > 1) by flattening (C, T, H, W) to
        (C*T, H, W) in the GeoTIFF and writing a metadata.json sidecar with
        ``num_channels``, ``num_timesteps``, and ``timestamps``.

        For T == 1, a metadata.json is only written when timestamps are present.

        Args:
            path: the directory to write to
            projection: the projection of the raster data
            bounds: the bounds of the raster data in the projection
            raster: the raster data (CTHW RasterArray)
            fname: override the filename to save as
            nodata_val: set the nodata value when writing the raster.
        """
        if fname is None:
            fname = self.fname

        c, t, h, w = raster.array.shape

        # Flatten CTHW -> (C*T, H, W) for the GeoTIFF.
        array = raster.array.reshape(c * t, h, w)

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

        # Write metadata.json sidecar when multi-timestep or timestamps are present.
        if t > 1 or raster.timestamps is not None:
            self.encode_metadata(
                path,
                GeotiffRasterMetadata(
                    num_channels=c,
                    num_timesteps=t,
                    timestamps=raster.timestamps,
                ),
            )

    def decode_raster(
        self,
        path: UPath,
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
        fname: str | None = None,
        nodata_val: int | float | None = None,
    ) -> RasterArray:
        """Decodes raster data.

        If a metadata.json sidecar exists with ``num_timesteps > 1``, the GeoTIFF
        is treated as (C*T, H, W) and reshaped back to (C, T, H, W).

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
            the raster data as a RasterArray (CTHW)
        """
        if fname is None:
            fname = self.fname

        metadata = self.decode_metadata(path)

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
                raw = vrt.read()  # (bands, H, W)

        # Reshape from (C*T, H, W) -> (C, T, H, W).
        if metadata and metadata.num_timesteps is not None:
            num_timesteps = metadata.num_timesteps
            num_channels = metadata.num_channels or raw.shape[0]
        else:
            num_timesteps = 1
            num_channels = raw.shape[0]
        array = raw.reshape(num_channels, num_timesteps, raw.shape[1], raw.shape[2])

        timestamps = metadata.timestamps if metadata else None
        return RasterArray(array=array, timestamps=timestamps)

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


class SingleImageRasterMetadata(pydantic.BaseModel):
    """Metadata sidecar for SingleImageRasterFormat."""

    projection: dict[str, Any] | None = None
    bounds: PixelBounds
    timestamps: list[tuple[datetime, datetime]] | None = None


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
        raster: RasterArray,
    ) -> None:
        """Encodes raster data.

        Args:
            path: the directory to write to
            projection: the projection of the raster data
            bounds: the bounds of the raster data in the projection
            raster: the raster data (CTHW RasterArray, T must be 1)
        """
        array = raster.get_chw_array()

        path.mkdir(parents=True, exist_ok=True)
        fname = path / ("image." + self.get_extension())
        with fname.open("wb") as f:
            # CHW -> HWC for PIL and squeeze channel dim for grayscale images.
            img_array = einops.rearrange(array, "c h w -> h w c")
            if img_array.shape[2] == 1:
                img_array = img_array[:, :, 0]
            Image.fromarray(img_array).save(f, format=self.format.upper())

        metadata = SingleImageRasterMetadata(
            projection=projection.serialize(),
            bounds=bounds,
            timestamps=raster.timestamps,
        )
        with (path / METADATA_FNAME).open("w") as f:
            f.write(metadata.model_dump_json())

    def decode_raster(
        self,
        path: UPath,
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
    ) -> RasterArray:
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
        with (path / METADATA_FNAME).open() as f:
            image_metadata = SingleImageRasterMetadata.model_validate_json(f.read())

        # If the projection key is set, verify that it matches the requested projection
        # since SingleImageRasterFormat currently does not support re-projecting.
        if image_metadata.projection is not None:
            source_data_projection = Projection.deserialize(image_metadata.projection)
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

        if bounds != image_metadata.bounds:
            # Need to extract relevant portion of image.
            dst = np.zeros(
                (array.shape[0], bounds[3] - bounds[1], bounds[2] - bounds[0]),
                dtype=array.dtype,
            )
            copy_spatial_array(
                array,
                dst,
                src_offset=(image_metadata.bounds[0], image_metadata.bounds[1]),
                dst_offset=(bounds[0], bounds[1]),
            )
            array = dst

        # Wrap as CTHW with T=1.
        return RasterArray(
            array=array[:, np.newaxis, :, :],
            timestamps=image_metadata.timestamps,
        )


class NumpyRasterFormat(RasterFormat):
    """A raster format that stores data as a raw NumPy .npy file.

    This is optimized for layers with many bands (e.g., 5000+) where GeoTIFF's
    per-band IFD overhead makes reading extremely slow. A 5110-band GeoTIFF
    takes ~1s to open due to per-band metadata parsing; the same data in .npy
    format loads in <1ms.

    The data is stored without spatial metadata (CRS/transform), so this format
    is best suited for layers where the spatial dimensions are trivial (e.g.,
    1×1) and no reprojection is needed during reading.

    The stored array must be [C, H, W]. On decode, if the stored spatial size
    is 1×1 and larger bounds are requested, the data is broadcast to match.
    """

    def __init__(self, fname: str = "data.npy"):
        """Initialize a NumpyRasterFormat.

        Args:
            fname: filename for the numpy array file.
        """
        self.fname = fname

    def encode_raster(
        self,
        path: UPath,
        projection: Projection,
        bounds: PixelBounds,
        raster: RasterArray,
    ) -> None:
        """Encodes raster data as a .npy file.

        Args:
            path: the directory to write to
            projection: the projection of the raster data (stored as metadata
                but not embedded in the file)
            bounds: the bounds of the raster data in the projection
            raster: the raster data (CTHW RasterArray)
        """
        path.mkdir(parents=True, exist_ok=True)
        # Store as CHW (flatten T dimension) for backward compatibility.
        array = raster.get_chw_array()
        np.save(path / self.fname, array)

    def decode_raster(
        self,
        path: UPath,
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
    ) -> RasterArray:
        """Decodes raster data from a .npy file.

        If the stored array is 1×1 spatially and the requested bounds are
        larger, the single pixel is broadcast (replicated) to the requested
        size.

        Args:
            path: the directory to read from
            projection: the projection to read the raster in (ignored; no
                reprojection is performed).
            bounds: the bounds to read in the given projection.
            resampling: resampling method (ignored; nearest-neighbor broadcast
                is always used for 1×1 data).

        Returns:
            the raster data as a RasterArray (CTHW).
        """
        data = np.load(path / self.fname)
        out_h = bounds[3] - bounds[1]
        out_w = bounds[2] - bounds[0]
        if data.shape[1] == 1 and data.shape[2] == 1 and (out_h != 1 or out_w != 1):
            data = np.broadcast_to(data, (data.shape[0], out_h, out_w)).copy()
        # Wrap CHW as CTHW with T=1.
        return RasterArray(chw_array=data)
