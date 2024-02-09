import io
from enum import Enum
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import rasterio.io
import rasterio.transform
from PIL import Image
from rasterio.crs import CRS

from rslearn.tile_stores import LayerMetadata, TileStore

RESOLUTION_EPSILON = 1e-6
TILE_SIZE = 512


class RasterFormat(Enum):
    GEOTIFF = "geotiff"
    PNG = "png"
    JPEG = "jpeg"


class DType(Enum):
    Native = "native"
    Uint8 = "uint8"
    Uint16 = "uint16"
    Uint32 = "uint32"
    Float32 = "float32"


class RasterOptions:
    def __init__(
        self,
        band_sets: Optional[list[list[str]]] = None,
        format: RasterFormat = RasterFormat.GEOTIFF,
        dtype: DType = DType.Native,
        zoom_offset: int = 0,
    ) -> None:
        """Creates a new RasterOptions instance.

        Args:
            band_sets: a list of band sets, each of which is a list of band names that
                should be stored together
            format: the format to store tiles in
            dtype: the pixel value type to store tiles in
            zoom_offset: non-negative integer, store images at window resolution
                divided by 2^(zoom_offset).
        """
        self.band_sets = band_sets
        self.format = format
        self.dtype = dtype
        self.zoom_offset = zoom_offset


class ArrayWithTransform:
    def __init__(
        self,
        array: npt.NDArray[Any],
        bands: list[str],
        transform: rasterio.transform.Affine,
    ) -> None:
        """Create a new ArrayWithTransform instance.

        Args:
            array: the numpy array (C, H, W) storing the raster data.
            bands: the names of the bands on the channel dimension.
            transform: the transform from pixel coordinates to projection coordinates.
        """
        self.array = array
        self.bands = bands
        self.transform = transform

    def pixel_bounds(self) -> tuple[int, int, int, int]:
        """Returns the bounds of the array in global pixel coordinates.

        The bounds is computed based on the stored transform.

        The returned coordinates are (left, top, right, bottom).
        """
        left = self.transform.c
        top = self.transform.f
        # Resolutions in projection units per pixel.
        x_resolution = self.transform.a
        y_resolution = self.transform.e
        start = (int(left / x_resolution), int(top / y_resolution))
        end = (start[0] + self.array.shape[2], start[1] + self.array.shape[1])
        return (start[0], start[1], end[0], end[1])

    def get_tile(self, band: str, tile: tuple[int, int]) -> npt.NDArray[Any]:
        """Returns portion of image corresponding to a band and tile.

        Args:
            band: the name of the band to retrieve
            tile: the tile to retrieve

        Returns:
            The portion of the image corresponding to the requested band and tile.
        """
        bounds = self.pixel_bounds()
        x1 = tile[0] * TILE_SIZE - bounds[0]
        y1 = tile[1] * TILE_SIZE - bounds[1]
        x2 = x1 + TILE_SIZE
        y2 = y1 + TILE_SIZE
        # Need to pad output if x1/y1/x2/y2 are out of bounds.
        # The padding is (before_y, after_y, before_x, after_x)
        padding = [0, 0, 0, 0]
        if x1 < 0:
            padding[2] = -x1
            x1 = 0
        if y1 < 0:
            padding[0] = -y1
            y1 = 0
        if x2 > self.array.shape[2]:
            padding[3] = x2 - self.array.shape[2]
            x2 = self.array.shape[2]
        if y2 > self.array.shape[1]:
            padding[1] = y2 - self.array.shape[1]
            y2 = self.array.shape[1]
        band_idx = self.bands.index(band)
        tile = self.array[band_idx, y1:y2, x1:x2]
        return np.pad(tile, ((padding[0], padding[1]), (padding[2], padding[3])))


def encode_data(
    crs: CRS,
    resolution: float,
    tile: tuple[int, int],
    image: npt.NDArray[Any],
    raster_options: RasterOptions,
) -> bytes:
    if raster_options.dtype != DType.Native:
        image = image.astype(raster_options.DType)

    if raster_options.format == RasterFormat.GEOTIFF:
        pass

    if raster_options.format == RasterFormat.PNG:
        buf = io.BytesIO()
        Image.fromarray(image).save(buf, format="PNG")
        return buf.getvalue()


def ingest_from_rasters(
    tile_store: TileStore,
    layer_prefix: tuple[str, ...],
    rasters: list[tuple[rasterio.io.DatasetReader, list[str]]],
    projections: list[tuple[CRS, float]],
    raster_options: RasterOptions,
) -> None:
    """Ingests in-memory rasterio datasets corresponding to an item into the tile store.

    Args:
        tile_store: the tile store to ingest into
        layer_prefix: the prefix to store this item under in the TileStore. The CRS and
            resolution will be appended to the prefix.
        rasters: a list of (raster, band_names) tuples of rasters
        projections: a list of (crs, resolution) tuples of projections in which the
            item is needed
        raster_options: configuration options
    """
    for crs, resolution in projections:
        resolution /= 2**raster_options.zoom_offset

        # Get the layer to use for this crs/resolution.
        # If it is marked completed, then we don't need to do any ingestion.
        # The caller should check this first though (to save time downloading the
        # rasters).
        layer_id = layer_prefix + (crs.to_string(), str(resolution))
        ts_layer = tile_store.create_layer(
            layer_id, LayerMetadata(crs, resolution, None, {})
        )
        if ts_layer.get_metadata().properties.get("completed"):
            continue

        # Warp each raster to this CRS and resolution if needed.
        warped_arrays: list[ArrayWithTransform] = []
        for raster, band_names in rasters:
            array = raster.read()

            pixel_size_x, pixel_size_y = raster.res

            if (
                raster.crs == crs
                and abs(pixel_size_x - resolution) < RESOLUTION_EPSILON
                and abs(pixel_size_y - resolution) < RESOLUTION_EPSILON
            ):
                # Include the top-left pixel index.
                warped_arrays.append(
                    ArrayWithTransform(array, band_names, raster.transform)
                )
                continue

            # Compute the suggested target transform.
            dst_transform, dst_width, dst_height = (
                rasterio.warp.calculate_default_transform(
                    # Source info.
                    src_crs=raster.crs,
                    width=raster.width,
                    height=raster.height,
                    left=raster.bounds[0],
                    bottom=raster.bounds[1],
                    right=raster.bounds[2],
                    top=raster.bounds[3],
                    # Destination info.
                    dst_crs=crs,
                    resolution=resolution,
                )
            )
            # Re-project the raster to the destination crs, resolution, and transform.
            warped_array = np.zeros(
                (array.shape[0], dst_height, dst_width), dtype=array.dtype
            )
            rasterio.warp.reproject(
                source=array,
                src_crs=raster.crs,
                src_transform=raster.transform,
                destination=warped_array,
                dst_crs=crs,
                dst_transform=dst_transform,
            )
            warped_arrays.append(
                ArrayWithTransform(warped_array, band_names, dst_transform)
            )

        # Create dict to lookup from band -> ArrayWithTransform that has the band.
        warped_array_by_band = {}
        for warped_array in warped_arrays:
            for band_name in warped_array.bands:
                warped_array_by_band[band_name] = warped_array

        # Get the union bounds of the warped arrays (left, top, right, bottom).
        # We will use this to iterate over possible tiles.
        union_bounds = None
        for array in warped_arrays:
            bounds = array.pixel_bounds()
            if union_bounds is None:
                union_bounds = bounds
            else:
                union_bounds = (
                    min(union_bounds[0], bounds[0]),
                    min(union_bounds[1], bounds[1]),
                    max(union_bounds[2], bounds[2]),
                    max(union_bounds[3], bounds[3]),
                )

        # Determine what the band sets are.
        # If not specified in raster_options, it defaults to the all bands as provided
        # by the data source.
        band_sets: list[list[str]] = []
        if raster_options.band_sets:
            band_sets = raster_options.band_sets
        else:
            for _, band_names in rasters:
                band_sets.append(band_names)

        # Collect the image data associated with non-zero tiles.
        tile1 = (union_bounds[0] // TILE_SIZE, union_bounds[1] // TILE_SIZE)
        tile2 = (
            (union_bounds[2] + TILE_SIZE - 1) // TILE_SIZE,
            (union_bounds[3] + TILE_SIZE - 1) // TILE_SIZE,
        )
        datas: list[tuple[int, int, bytes]] = []
        for tile_col in range(tile1[0], tile2[0]):
            for tile_row in range(tile1[1], tile2[1]):
                tile = (tile_col, tile_row)
                for band_set in band_sets:
                    image_bands = []
                    any_nonzero = False
                    for band in band_set:
                        band_content = warped_array_by_band[band].get_tile(band, tile)
                        image_bands.append(band_content)
                        any_nonzero = any_nonzero or np.count_nonzero(band_content) > 0

                    if not any_nonzero:
                        continue

                    image = np.stack(image_bands, axis=2)
                    encoded_content = encode_data(
                        crs, resolution, tile, image, raster_options
                    )
                    datas.append((tile[0], tile[1], encoded_content))

        ts_layer.save_tiles(datas)
        ts_layer.set_property("completed", True)
