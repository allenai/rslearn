from typing import Any

import numpy as np
import numpy.typing as npt
import rasterio.io
import rasterio.transform

from rslearn.config import BandSetConfig
from rslearn.tile_stores import LayerMetadata, TileStore
from rslearn.utils import Projection, STGeometry, is_same_resolution
from rslearn.utils.raster_format import GeotiffRasterFormat

TILE_SIZE = 512


class ArrayWithTransform:
    def __init__(
        self,
        array: npt.NDArray[Any],
        transform: rasterio.transform.Affine,
    ) -> None:
        """Create a new ArrayWithTransform instance.

        Args:
            array: the numpy array (C, H, W) storing the raster data.
            transform: the transform from pixel coordinates to projection coordinates.
        """
        self.array = array
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

    def get_tile(self, tile: tuple[int, int]) -> npt.NDArray[Any]:
        """Returns portion of image corresponding to a tile.

        Args:
            tile: the tile to retrieve

        Returns:
            The portion of the image corresponding to the requested tile.
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
        tile = self.array[:, y1:y2, x1:x2]
        return np.pad(
            tile, ((0, 0), (padding[0], padding[1]), (padding[2], padding[3]))
        )


def get_final_projection(projection: Projection, band_set: BandSetConfig) -> Projection:
    """Gets the final projection based on window projection and band set config.

    The band set config may apply a non-zero zoom offset that modifies the window's
    projection.

    Args:
        projection: the window's projection
        band_set: band set configuration object

    Returns:
        updated projection with zoom offset applied
    """
    if band_set.zoom_offset == 0:
        return projection
    return Projection(
        projection.crs,
        projection.resolution / 2**band_set.zoom_offset,
    )


def get_needed_projections(
    tile_store: TileStore,
    raster_bands: list[str],
    band_sets: list[BandSetConfig],
    geometries: list[STGeometry],
) -> list[Projection]:
    """Determines the projections of an item that are needed for a given raster file.

    Projections that appear in geometries are skipped if a corresponding layer is
    present in tile_store with metadata marked completed.

    Args:
        tile_store: TileStore prefixed with the item name and file name
        raster_bands: list of bands contained in the raster file
        band_sets: configured band sets
        geometries: list of geometries for which the item is needed

    Returns:
        list of Projection objects for which the item has not been ingested yet
    """
    # Identify which band set configs are relevant to this raster.
    raster_bands = set(raster_bands)
    relevant_band_sets = []
    for band_set in band_sets:
        is_match = False
        for band in band_set.bands:
            if band not in raster_bands:
                continue
            is_match = True
            break
        if not is_match:
            continue
        relevant_band_sets.append(band_set)

    all_projections = {geometry.projection for geometry in geometries}
    needed_projections = []
    for projection in all_projections:
        for band_set in relevant_band_sets:
            final_projection = get_final_projection(projection, band_set)
            ts_layer = tile_store.get_layer((str(final_projection),))
            if ts_layer and ts_layer.get_metadata().properties.get("completed"):
                continue
            needed_projections.append(final_projection)
    return needed_projections


def ingest_raster(
    tile_store: TileStore,
    raster: rasterio.io.DatasetReader,
    projection: Projection,
) -> None:
    """Ingests an in-memory rasterio dataset into the tile store.

    Args:
        tile_store: the tile store to ingest into, prefixed with the item name and
            raster band names
        raster: the rasterio raster
        projection: the projection to save the raster as
    """
    format = GeotiffRasterFormat()

    # Get layer in tile store to save under.
    ts_layer = tile_store.create_layer(
        (str(projection),), LayerMetadata(projection, None, {})
    )
    if ts_layer.get_metadata().properties.get("completed"):
        return

    # Warp each raster to this projection if needed.
    array = raster.read()

    pixel_size_x, pixel_size_y = raster.res

    if (
        raster.crs == projection.crs
        and is_same_resolution(pixel_size_x, projection.resolution)
        and is_same_resolution(pixel_size_y, projection.resolution)
    ):
        # Include the top-left pixel index.
        warped_array = ArrayWithTransform(array, raster.transform)

    else:
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
                dst_crs=projection.crs,
                resolution=projection.resolution,
            )
        )
        # Re-project the raster to the destination crs, resolution, and transform.
        dst_array = np.zeros((array.shape[0], dst_height, dst_width), dtype=array.dtype)
        rasterio.warp.reproject(
            source=array,
            src_crs=raster.crs,
            src_transform=raster.transform,
            destination=dst_array,
            dst_crs=projection.crs,
            dst_transform=dst_transform,
        )
        warped_array = ArrayWithTransform(dst_array, dst_transform)

    # Collect the image data associated with non-zero tiles.
    bounds = warped_array.pixel_bounds()
    tile1 = (bounds[0] // TILE_SIZE, bounds[1] // TILE_SIZE)
    tile2 = (
        (bounds[2] + TILE_SIZE - 1) // TILE_SIZE,
        (bounds[3] + TILE_SIZE - 1) // TILE_SIZE,
    )
    datas: list[tuple[int, int, bytes]] = []
    for tile_col in range(tile1[0], tile2[0]):
        for tile_row in range(tile1[1], tile2[1]):
            tile = (tile_col, tile_row)
            image = warped_array.get_tile(tile)
            if np.count_nonzero(image) == 0:
                continue

            encoded_content = format.encode_image(projection, tile, image)
            datas.append((tile[0], tile[1], encoded_content))

    ts_layer.save_tiles(datas, extension=format.get_extension())
    ts_layer.set_property("completed", True)
