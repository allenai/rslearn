from datetime import datetime
from typing import Any, Optional

import affine
import numpy as np
import numpy.typing as npt
import rasterio.enums
import rasterio.io
import rasterio.transform

from rslearn.config import BandSetConfig
from rslearn.const import TILE_SIZE
from rslearn.tile_stores import LayerMetadata, TileStore
from rslearn.utils import Projection, STGeometry


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
            final_projection, _ = band_set.get_final_projection_and_bounds(
                projection, None
            )
            ts_layer = tile_store.get_layer((str(final_projection),))
            if ts_layer and ts_layer.get_metadata().properties.get("completed"):
                continue
            needed_projections.append(final_projection)
    return needed_projections


def ingest_raster(
    tile_store: TileStore,
    raster: rasterio.io.DatasetReader,
    projection: Projection,
    time_range: Optional[tuple[datetime, datetime]],
) -> None:
    """Ingests an in-memory rasterio dataset into the tile store.

    Args:
        tile_store: the tile store to ingest into, prefixed with the item name and
            raster band names
        raster: the rasterio raster
        projection: the projection to save the raster as
        time_range: time range of the raster
    """

    # Get layer in tile store to save under.
    ts_layer = tile_store.create_layer(
        (str(projection),), LayerMetadata(projection, None, {})
    )
    if ts_layer.get_metadata().properties.get("completed"):
        return

    # Warp each raster to this projection if needed.
    array = raster.read()

    needs_warping = True
    if isinstance(raster.transform, affine.Affine):
        raster_projection = Projection(
            raster.crs, raster.transform.a, raster.transform.e
        )
        needs_warping = raster_projection != projection

    if not needs_warping:
        # Include the top-left pixel index.
        warped_array = ArrayWithTransform(array, raster.transform)

    else:
        # Compute the suggested target transform.
        # rasterio negates the y resolution itself so here we have to negate it.
        (
            dst_transform,
            dst_width,
            dst_height,
        ) = rasterio.warp.calculate_default_transform(
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
            resolution=(projection.x_resolution, -projection.y_resolution),
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
            resampling=rasterio.enums.Resampling.bilinear,
        )
        warped_array = ArrayWithTransform(dst_array, dst_transform)

    # Collect the image data associated with non-zero tiles.
    bounds = warped_array.pixel_bounds()
    tile1 = (bounds[0] // TILE_SIZE, bounds[1] // TILE_SIZE)
    tile2 = (
        bounds[2] // TILE_SIZE,
        bounds[3] // TILE_SIZE,
    )
    datas: list[tuple[int, int, npt.NDArray[Any]]] = []
    for tile_col in range(min(tile1[0], tile2[0]), max(tile1[0], tile2[0]) + 1):
        for tile_row in range(min(tile1[1], tile2[1]), max(tile1[1], tile2[1]) + 1):
            tile = (tile_col, tile_row)
            image = warped_array.get_tile(tile)
            if np.count_nonzero(image) == 0:
                continue
            datas.append((tile[0], tile[1], image))

    ts_layer.save_rasters(datas)
    ts_layer.set_property("completed", True)
