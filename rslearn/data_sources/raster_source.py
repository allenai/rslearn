"""Helper functions for raster data sources."""

from collections.abc import Callable
from datetime import datetime
from typing import Any

import affine
import numpy as np
import numpy.typing as npt
import rasterio.enums
import rasterio.io
import rasterio.transform
from rasterio.crs import CRS

from rslearn.config import BandSetConfig, RasterFormatConfig, RasterLayerConfig
from rslearn.const import TILE_SIZE
from rslearn.dataset import Window
from rslearn.log_utils import get_logger
from rslearn.tile_stores import LayerMetadata, TileStore
from rslearn.utils import Projection, STGeometry
from rslearn.utils.raster_format import load_raster_format

logger = get_logger(__name__)


class ArrayWithTransform:
    """Stores an array along with the transform associated with the array."""

    def __init__(
        self, array: npt.NDArray[Any], crs: CRS, transform: rasterio.transform.Affine
    ) -> None:
        """Create a new ArrayWithTransform instance.

        Args:
            array: the numpy array (C, H, W) storing the raster data.
            crs: the CRS of the array
            transform: the transform from pixel coordinates to projection coordinates.
        """
        self.array = array
        self.crs = crs
        self.transform = transform

        # Store additional data matching rasterio.io.DatasetReader so we can use them
        # interchangeably in ingest_raster.
        self.width = self.array.shape[2]
        self.height = self.array.shape[1]

        # Left/top in projection units.
        self.left = self.transform.c
        self.top = self.transform.f

        # Resolution in projection units per pixel.
        self.x_resolution = self.transform.a
        self.y_resolution = self.transform.e

        # Right/bottom and bounds in projection units.
        self.right = self.left + self.width * self.x_resolution
        self.bottom = self.top + self.height * self.y_resolution
        self.bounds = [
            min(self.left, self.right),
            min(self.top, self.bottom),
            max(self.left, self.right),
            max(self.top, self.bottom),
        ]

    def read(self) -> npt.NDArray[Any]:
        """Reads the array.

        This is to mimic the rasterio.DatasetReader API.

        Returns:
            the array
        """
        return self.array

    def close(self) -> None:
        """This is to mimic the rasterio.DatasetReader API.

        The close function is a no-op.
        """
        pass

    def pixel_bounds(self) -> tuple[int, int, int, int]:
        """Returns the bounds of the array in global pixel coordinates.

        The bounds is computed based on the stored transform.

        The returned coordinates are (left, top, right, bottom).
        """
        start = (int(self.left / self.x_resolution), int(self.top / self.y_resolution))
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
    raster_bands_set = set(raster_bands)
    relevant_band_set_list = []
    for band_set in band_sets:
        is_match = False
        if band_set.bands is None:
            continue
        for band in band_set.bands:
            if band not in raster_bands_set:
                continue
            is_match = True
            break
        if not is_match:
            continue
        relevant_band_set_list.append(band_set)

    all_projections = {geometry.projection for geometry in geometries}
    needed_projections = []
    for projection in all_projections:
        for band_set in relevant_band_set_list:
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
    raster: rasterio.io.DatasetReader | ArrayWithTransform,
    projection: Projection,
    time_range: tuple[datetime, datetime] | None = None,
    layer_config: RasterLayerConfig | None = None,
    array_callback: Callable[[npt.NDArray[Any]], npt.NDArray[Any]] | None = None,
) -> None:
    """Ingests an in-memory rasterio dataset into the tile store.

    Args:
        tile_store: the tile store to ingest into, prefixed with the item name and
            raster band names
        raster: the rasterio raster
        projection: the projection to save the raster as
        time_range: time range of the raster
        layer_config: the RasterLayerConfig of this layer
        array_callback: callback function to apply on the array read from the raster
    """
    # Get layer in tile store to save under.
    ts_layer = tile_store.create_layer(
        (str(projection),), LayerMetadata(projection, time_range, {})
    )
    if ts_layer.get_metadata().properties.get("completed"):
        return

    # Warp each raster to this projection if needed.
    array = raster.read()
    if array_callback:
        array = array_callback(array)

    needs_warping = True
    if isinstance(raster.transform, affine.Affine):
        raster_projection = Projection(
            raster.crs, raster.transform.a, raster.transform.e
        )
        needs_warping = raster_projection != projection

    if not needs_warping:
        # Include the top-left pixel index.
        warped_array = ArrayWithTransform(array, raster.crs, raster.transform)

    else:
        # Compute the suggested target transform.
        # rasterio negates the y resolution itself so here we have to negate it.
        raster_bounds: rasterio.coords.BoundingBox = raster.bounds
        (dst_transform, dst_width, dst_height) = (
            rasterio.warp.calculate_default_transform(
                # Source info.
                src_crs=raster.crs,
                width=raster.width,
                height=raster.height,
                left=raster_bounds.left,
                bottom=raster_bounds.bottom,
                right=raster_bounds.right,
                top=raster_bounds.top,
                # Destination info.
                dst_crs=projection.crs,
                resolution=(projection.x_resolution, -projection.y_resolution),
            )
        )

        resampling_method = rasterio.enums.Resampling.bilinear
        if layer_config:
            resampling_method = layer_config.resampling_method

        # Re-project the raster to the destination crs, resolution, and transform.
        dst_array = np.zeros((array.shape[0], dst_height, dst_width), dtype=array.dtype)
        rasterio.warp.reproject(
            source=array,
            src_crs=raster.crs,
            src_transform=raster.transform,
            destination=dst_array,
            dst_crs=projection.crs,
            dst_transform=dst_transform,
            resampling=resampling_method,
        )
        warped_array = ArrayWithTransform(dst_array, projection.crs, dst_transform)

    ts_layer.write_raster(warped_array.pixel_bounds(), warped_array.array)
    ts_layer.set_property("completed", True)


def materialize_raster(
    raster: rasterio.io.DatasetReader | ArrayWithTransform,
    window: Window,
    layer_name: str,
    band_cfg: BandSetConfig,
) -> None:
    """Materialize a given raster for a window.

    Currently it is only supported for materializing one band set.

    Args:
        raster: the raster data
        window: the window to materialize
        layer_name: the layer
        band_cfg: the band configuration
    """
    window_projection, window_bounds = band_cfg.get_final_projection_and_bounds(
        window.projection, window.bounds
    )
    if window_bounds is None:
        raise ValueError(f"No windowbounds specified for {layer_name}")
    # Re-project to just extract the window.
    array = raster.read()
    window_width = window_bounds[2] - window_bounds[0]
    window_height = window_bounds[3] - window_bounds[1]
    dst_transform = rasterio.transform.Affine(
        window_projection.x_resolution,
        0,
        window_bounds[0] * window_projection.x_resolution,
        0,
        window_projection.y_resolution,
        window_bounds[1] * window_projection.y_resolution,
    )
    dst_array = np.zeros(
        (array.shape[0], window_height, window_width), dtype=array.dtype
    )
    rasterio.warp.reproject(
        source=array,
        src_crs=raster.crs,
        src_transform=raster.transform,
        destination=dst_array,
        dst_crs=window_projection.crs,
        dst_transform=dst_transform,
        resampling=rasterio.enums.Resampling.bilinear,
    )
    if band_cfg.bands is None or band_cfg.format is None:
        raise ValueError(
            f"No bands or format specified for {layer_name} materialization"
        )
    # Write the array to layer directory.
    layer_dir = window.path / "layers" / layer_name
    out_dir = layer_dir / "_".join(band_cfg.bands)
    out_dir.mkdir(parents=True, exist_ok=True)
    raster_format = load_raster_format(
        RasterFormatConfig(band_cfg.format["name"], band_cfg.format)
    )
    raster_format.encode_raster(out_dir, window_projection, window_bounds, dst_array)
    (layer_dir / "completed").touch()
