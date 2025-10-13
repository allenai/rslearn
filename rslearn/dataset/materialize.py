"""Classes to implement dataset materialization."""

from collections.abc import Callable
from typing import Any, Generic, TypeVar

import numpy as np
import numpy.typing as npt
from rasterio.enums import Resampling

from rslearn.config import (
    BandSetConfig,
    CompositingMethod,
    LayerConfig,
    RasterFormatConfig,
    RasterLayerConfig,
    VectorLayerConfig,
)
from rslearn.data_sources.data_source import ItemType
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import PixelBounds, Projection
from rslearn.utils.raster_format import load_raster_format
from rslearn.utils.vector_format import load_vector_format

from .remap import Remapper, load_remapper
from .window import Window

_MaterializerT = TypeVar("_MaterializerT", bound="Materializer")


class _MaterializerRegistry(dict[str, type["Materializer"]]):
    """Registry for Materializer classes."""

    def register(
        self, name: str
    ) -> Callable[[type[_MaterializerT]], type[_MaterializerT]]:
        """Decorator to register a materializer class."""

        def decorator(cls: type[_MaterializerT]) -> type[_MaterializerT]:
            self[name] = cls
            return cls

        return decorator


Materializers = _MaterializerRegistry()


LayerConfigType = TypeVar("LayerConfigType", bound=LayerConfig)


class Materializer(Generic[LayerConfigType]):
    """An abstract class that materializes data from a tile store."""

    def materialize(
        self,
        tile_store: TileStoreWithLayer,
        window: Window,
        layer_name: str,
        layer_cfg: LayerConfigType,
        item_groups: list[list[ItemType]],
    ) -> None:
        """Materialize portions of items corresponding to this window into the dataset.

        Args:
            tile_store: the tile store where the items have been ingested (unprefixed)
            window: the window to materialize
            layer_name: the name of the layer to materialize
            layer_cfg: the configuration of the layer to materialize
            item_groups: the items associated with this window and layer
        """
        raise NotImplementedError


def read_raster_window_from_tiles(
    dst: npt.NDArray[Any],
    tile_store: TileStoreWithLayer,
    item_name: str,
    bands: list[str],
    projection: Projection,
    bounds: PixelBounds,
    src_indexes: list[int],
    dst_indexes: list[int],
    nodata_vals: list[float],
    remapper: Remapper | None = None,
    resampling: Resampling = Resampling.bilinear,
) -> None:
    """Read a window of raster data from tiles in a tile store.

    Pixels in the destination array are only overwritten if not already non-zero.

    Args:
        dst: the destination numpy array
        tile_store: the TileStore to read from.
        item_name: the item name.
        bands: the bands that identify the raster we want to read.
        projection: the projection of the dst array.
        bounds: the bounds of the dst array.
        src_indexes: the source band indexes to use
        dst_indexes: corresponding destination band indexes for each source band index
        nodata_vals: the nodata values for each band, to determine which parts of dst
            should be overwritten.
        remapper: optional remapper to apply on the source pixel values
        resampling: how to resample the pixels in case re-projection is needed.
    """
    # Only read the portion of the raster that overlaps with dst.
    # This way we can avoid creating big arrays that are all empty which speeds things
    # up for large windows.
    src_bounds = tile_store.get_raster_bounds(item_name, bands, projection)
    intersection = (
        max(bounds[0], src_bounds[0]),
        max(bounds[1], src_bounds[1]),
        min(bounds[2], src_bounds[2]),
        min(bounds[3], src_bounds[3]),
    )
    if intersection[2] <= intersection[0] or intersection[3] <= intersection[1]:
        return

    dst_col_offset = intersection[0] - bounds[0]
    dst_row_offset = intersection[1] - bounds[1]

    src = tile_store.read_raster(
        item_name, bands, projection, intersection, resampling=resampling
    )
    src = src[src_indexes, :, :]
    if remapper:
        src = remapper(src, dst.dtype)

    dst_crop = dst[
        :,
        dst_row_offset : dst_row_offset + src.shape[1],
        dst_col_offset : dst_col_offset + src.shape[2],
    ]

    # Create mask indicating where dst has no data (based on nodata_vals).
    # We overwrite dst at pixels where all the bands are nodata.
    nodata_vals_arr = np.array(nodata_vals)[:, None, None]
    mask = (dst_crop[dst_indexes, :, :] == nodata_vals_arr).min(axis=0)

    for src_index, dst_index in enumerate(dst_indexes):
        dst_crop[dst_index, mask] = src[src_index, mask]


def get_needed_band_sets_and_indexes(
    item: ItemType,
    bands: list[str],
    tile_store: TileStoreWithLayer,
) -> list[tuple[list[str], list[int], list[int]]]:
    """Identify indexes of required bands in tile store.

    Returns:
        A list for each tile-store layer that contains at least
        one requested band, a tuple: (src_bands, src_idx, dst_idx) where
        - src_bands: the full band list for that layer,
        - src_idx: indexes into src_bands of the bands that were requested,
        - dst_idx: corresponding indexes in the requested `bands` list.
    """
    # Identify which tile store layer(s) to read to get the configured bands.
    wanted_band_indexes = {}
    for i, band in enumerate(bands):
        wanted_band_indexes[band] = i

    available_bands = tile_store.get_raster_bands(item.name)
    needed_band_sets_and_indexes = []

    for src_bands in available_bands:
        needed_src_indexes = []
        needed_dst_indexes = []
        for i, band in enumerate(src_bands):
            if band not in wanted_band_indexes:
                continue
            needed_src_indexes.append(i)
            needed_dst_indexes.append(wanted_band_indexes[band])
            del wanted_band_indexes[band]
        if len(needed_src_indexes) == 0:
            continue
        needed_band_sets_and_indexes.append(
            (src_bands, needed_src_indexes, needed_dst_indexes)
        )

    if len(wanted_band_indexes) > 0:
        # This item doesn't have all the needed bands, so skip it.
        return []

    return needed_band_sets_and_indexes


def build_first_valid_composite(
    group: list[ItemType],
    nodata_vals: list[Any],
    bands: list[str],
    bounds: PixelBounds,
    band_dtype: Any,
    tile_store: TileStoreWithLayer,
    projection: Projection,
    remapper: Remapper | None,
    resampling_method: Resampling = Resampling.bilinear,
) -> npt.NDArray[np.generic]:
    """Build a composite by selecting the first valid pixel of items in the group.

    A composite of shape of (bands,bounds) is created by iterating over items in
    group in order and selecting the first pixel that is not nodata per index.

    Args:
        group: list of items to composite together
        nodata_vals: list of nodata values for each band
        bands: list of band names to include in the composite
        bounds: pixel bounds defining the spatial extent of the composite
        band_dtype: data type for the output bands
        tile_store: tile store containing the actual raster data
        projection: spatial projection for the composite
        remapper: remapper to apply to pixel values, or None
        resampling_method: resampling method to use when reprojecting

    Returns:
        Composite of shape (bands, bounds) built from all items in the group

    """
    # Initialize the destination array to the nodata values.
    # We default the nodata value to 0.
    dst = np.zeros(
        (len(bands), bounds[3] - bounds[1], bounds[2] - bounds[0]),
        dtype=band_dtype,
    )

    for idx, nodata_val in enumerate(nodata_vals):
        dst[idx] = nodata_val

    for item in group:
        needed_band_sets_and_indexes = get_needed_band_sets_and_indexes(
            item, bands, tile_store
        )

        for (
            src_bands,
            src_indexes,
            dst_indexes,
        ) in needed_band_sets_and_indexes:
            cur_nodata_vals = [nodata_vals[idx] for idx in dst_indexes]
            read_raster_window_from_tiles(
                dst=dst,
                tile_store=tile_store,
                item_name=item.name,
                bands=src_bands,
                projection=projection,
                bounds=bounds,
                src_indexes=src_indexes,
                dst_indexes=dst_indexes,
                nodata_vals=cur_nodata_vals,
                remapper=remapper,
                resampling=resampling_method,
            )

    return dst


def read_and_stack_raster_windows(
    group: list[ItemType],
    bounds: PixelBounds,
    bands: list[str],
    tile_store: TileStoreWithLayer,
    projection: Projection,
    nodata_vals: list[Any],
    remapper: Remapper | None,
    band_dtype: Any,
    resampling_method: Resampling = Resampling.bilinear,
) -> npt.NDArray[np.generic]:
    """Create a stack of extent aligned raster windows.

    Args:
        group: Iterable of items (e.g., scene metadata objects) to read data from.
        bounds: Pixel bounds as (xmin, ymin, xmax, ymax) defining the spatial extent.
        bands: List of band names to include in the output.
        tile_store: Tile store containing the raster tiles for the items.
        projection: Projection object specifying the spatial reference system.
        nodata_vals: List of nodata values corresponding to each band.
        band_dtype: Data type for the output raster (e.g., np.uint16, np.float32).
        remapper: Optional remapper object to transform pixel values after reading.
        resampling_method: Resampling method to use when reading/reprojecting tiles.

    Returns:
        NumPy array of shape (num_items, num_bands, height, width) containing
        the stacked rasters for all items, with nodata values filled where data
        is missing.
    """
    height = bounds[3] - bounds[1]
    width = bounds[2] - bounds[0]
    window_shape = (len(bands), height, width)

    extent_aligned_raster_windows: list[np.ndarray] = []

    for item in group:
        # Initialize destination array to nodata
        dst = np.empty(window_shape, dtype=band_dtype)
        for idx, nodata_val in enumerate(nodata_vals):
            dst[idx, :, :] = nodata_val

        # Determine which source band sets/indexes are needed for this item
        needed_band_sets_and_indexes = get_needed_band_sets_and_indexes(
            item, bands, tile_store
        )

        # Fill the destination window from the tile store
        for src_bands, src_indexes, dst_indexes in needed_band_sets_and_indexes:
            cur_nodata_vals = [nodata_vals[idx] for idx in dst_indexes]
            read_raster_window_from_tiles(
                dst=dst,
                tile_store=tile_store,
                item_name=item.name,
                bands=src_bands,
                projection=projection,
                bounds=bounds,
                src_indexes=src_indexes,
                dst_indexes=dst_indexes,
                nodata_vals=cur_nodata_vals,
                remapper=remapper,
                resampling=resampling_method,
            )

        extent_aligned_raster_windows.append(dst)

    # Stack along a new axis (items axis): (N_items, N_bands, H, W)
    stacked_arrays = np.stack(extent_aligned_raster_windows, axis=0)
    return stacked_arrays


def mask_stacked_rasters(
    stacked_rasters: npt.NDArray[np.generic],
    nodata_vals: list[Any],
) -> np.ma.MaskedArray:
    """Masks the stacked rasters - each items band with the corresponding nodata val.

    Args:
        stacked_rasters: NumPy array of shape (num_items, num_bands, height, width)
            containing raster values for each item in the group.
        nodata_vals: Sequence of nodata values, one per band, used to identify invalid
            pixels in the stacked rasters.

    Returns:
        np.ma.MaskedArray with the same shape as `stacked_rasters`, where all
        pixels equal to the per-band nodata value are masked.
    """
    # Create mask based on nodata values
    nodata_vals_array = np.array(nodata_vals).reshape(1, -1, 1, 1)
    valid_mask = stacked_rasters != nodata_vals_array

    # Create masked array for all bands
    masked_data = np.ma.masked_where(~valid_mask, stacked_rasters)

    return masked_data


def build_mean_composite(
    group: list[ItemType],
    nodata_vals: list[Any],
    bands: list[str],
    bounds: PixelBounds,
    band_dtype: Any,
    tile_store: TileStoreWithLayer,
    projection: Projection,
    remapper: Remapper | None,
    resampling_method: Resampling = Resampling.bilinear,
) -> npt.NDArray[np.generic]:
    """Build a composite by computing the mean of valid pixels across items in the group.

    A composite of shape (bands, bounds) is created by computing the per-pixel mean of
    valid (non-nodata) pixels across all items in the group.

    Args:
        group: list of items to composite together
        nodata_vals: list of nodata values for each band
        bands: list of band names to include in the composite
        bounds: pixel bounds defining the spatial extent of the composite
        band_dtype: data type for the output bands
        tile_store: tile store containing the raster data
        projection: spatial projection for the composite
        remapper: remapper to apply to pixel values, or None
        resampling_method: resampling method to use when reprojecting

    Returns:
        Composite of shape (bands, bounds) having per-pixel mean of all items in the group
    """
    # TODO: Might want to add a running sum/count based method to reduce memory utilization

    stacked_arrays = read_and_stack_raster_windows(
        group=group,
        bounds=bounds,
        bands=bands,
        tile_store=tile_store,
        projection=projection,
        nodata_vals=nodata_vals,
        band_dtype=band_dtype,
        remapper=remapper,
        resampling_method=resampling_method,
    )

    # Mask stacked arrays with nodata values of each band
    masked_data = mask_stacked_rasters(stacked_arrays, nodata_vals)

    # Compute mean along the items axis for all
    mean_result = np.ma.mean(masked_data, axis=0)

    # Fill masked values and convert to target dtype
    fill_vals = np.array(nodata_vals).reshape(-1, 1, 1)
    result = np.ma.filled(mean_result, fill_value=fill_vals).astype(band_dtype)

    return result


def build_median_composite(
    group: list[ItemType],
    nodata_vals: list[Any],
    bands: list[str],
    bounds: PixelBounds,
    band_dtype: Any,
    tile_store: TileStoreWithLayer,
    projection: Projection,
    remapper: Remapper | None,
    resampling_method: Resampling = Resampling.bilinear,
) -> npt.NDArray[np.generic]:
    """Build a composite by computing the median of valid pixels across items in the group.

    A composite of shape (bands, bounds) is created by computing the per-pixel median of
    valid (non-nodata) pixels across all items in the group.

    Args:
        group: list of items to composite together
        nodata_vals: list of nodata values for each band
        bands: list of band names to include in the composite
        bounds: pixel bounds defining the spatial extent of the composite
        band_dtype: data type for the output bands
        tile_store: tile store containing the raster data
        projection: spatial projection for the composite
        remapper: remapper to apply to pixel values, or None
        resampling_method: resampling method to use when reprojecting

    Returns:
        Composite of shape (bands, bounds) having per-pixel median of all items in the group
    """
    stacked_arrays = read_and_stack_raster_windows(
        group=group,
        bounds=bounds,
        bands=bands,
        tile_store=tile_store,
        projection=projection,
        nodata_vals=nodata_vals,
        band_dtype=band_dtype,
        remapper=remapper,
        resampling_method=resampling_method,
    )

    # Mask stacked arrays with nodata values of each band
    masked_data = mask_stacked_rasters(stacked_arrays, nodata_vals)

    # Compute median along the items axis for all
    mean_result = np.ma.median(masked_data, axis=0)

    # Fill masked values and convert to target dtype
    fill_vals = np.array(nodata_vals).reshape(-1, 1, 1)
    result = np.ma.filled(mean_result, fill_value=fill_vals).astype(band_dtype)

    return result


compositing_methods = {
    CompositingMethod.FIRST_VALID: build_first_valid_composite,
    CompositingMethod.MEAN: build_mean_composite,
    CompositingMethod.MEDIAN: build_median_composite,
}


def build_composite(
    group: list[ItemType],
    compositing_method: CompositingMethod,
    tile_store: TileStoreWithLayer,
    layer_cfg: RasterLayerConfig,
    band_cfg: BandSetConfig,
    projection: Projection,
    bounds: PixelBounds,
    remapper: Remapper | None,
) -> npt.NDArray[np.generic]:
    """Build a temporal composite for specified bands from items in the group.

    Args:
        group: list of items to composite together
        compositing_method: Which method to use for compositing. First valid chooses the first valid value per pixel, mean takes the mean value per pixel
        tile_store: tile store containing the raster data
        layer_cfg: the configuration of the layer to materialize
        band_cfg: the configuration of the layer to materialize. Contains the bands to process.
        projection: spatial projection for the composite
        bounds: pixel bounds defining the spatial extent of the composite
        remapper: remapper to apply to pixel values, or None
    """
    nodata_vals = band_cfg.nodata_vals
    if nodata_vals is None:
        nodata_vals = [0 for _ in band_cfg.bands]

    return compositing_methods[compositing_method](
        group=group,
        nodata_vals=nodata_vals,
        bands=band_cfg.bands,
        bounds=bounds,
        band_dtype=band_cfg.dtype.value,
        tile_store=tile_store,
        projection=projection,
        resampling_method=layer_cfg.resampling_method,
        remapper=remapper,
    )


@Materializers.register("raster")
class RasterMaterializer(Materializer[RasterLayerConfig]):
    """A Materializer for raster data."""

    def materialize(
        self,
        tile_store: TileStoreWithLayer,
        window: Window,
        layer_name: str,
        layer_cfg: RasterLayerConfig,
        item_groups: list[list[ItemType]],
    ) -> None:
        """Materialize portions of items corresponding to this window into the dataset.

        Args:
            tile_store: the tile store where the items have been ingested
            window: the window to materialize
            layer_name: name of the layer to materialize
            layer_cfg: the configuration of the layer to materialize
            item_groups: the items associated with this window and layer
        """
        assert isinstance(layer_cfg, RasterLayerConfig)

        for band_cfg in layer_cfg.band_sets:
            # band_cfg could specify zoom_offset and maybe other parameters that affect
            # projection/bounds, so use the corrected projection/bounds.
            projection, bounds = band_cfg.get_final_projection_and_bounds(
                window.projection, window.bounds
            )

            # Also load remapper if set.
            remapper = None
            if band_cfg.remap:
                remapper = load_remapper(band_cfg.remap)

            raster_format = load_raster_format(
                RasterFormatConfig(band_cfg.format["name"], band_cfg.format)
            )

            for group_id, group in enumerate(item_groups):
                composite = build_composite(
                    group=group,
                    compositing_method=layer_cfg.compositing_method,
                    tile_store=tile_store,
                    layer_cfg=layer_cfg,
                    band_cfg=band_cfg,
                    projection=projection,
                    bounds=bounds,
                    remapper=remapper,
                )
                raster_format.encode_raster(
                    window.get_raster_dir(layer_name, band_cfg.bands, group_id),
                    projection,
                    bounds,
                    composite,
                )

        for group_id in range(len(item_groups)):
            window.mark_layer_completed(layer_name, group_id)


@Materializers.register("vector")
class VectorMaterializer(Materializer):
    """A Materializer for vector data."""

    def materialize(
        self,
        tile_store: TileStoreWithLayer,
        window: Window,
        layer_name: str,
        layer_cfg: LayerConfig,
        item_groups: list[list[ItemType]],
    ) -> None:
        """Materialize portions of items corresponding to this window into the dataset.

        Args:
            tile_store: the tile store where the items have been ingested (unprefixed)
            window: the window to materialize
            layer_name: the layer to materialize
            layer_cfg: the configuration of the layer to materialize
            item_groups: the items associated with this window and layer
        """
        assert isinstance(layer_cfg, VectorLayerConfig)
        vector_format = load_vector_format(layer_cfg.format)

        for group_id, group in enumerate(item_groups):
            features: list[Feature] = []

            for item in group:
                cur_features = tile_store.read_vector(
                    item.name, window.projection, window.bounds
                )
                features.extend(cur_features)

            vector_format.encode_vector(
                window.get_layer_dir(layer_name, group_id), features
            )

        for group_id in range(len(item_groups)):
            window.mark_layer_completed(layer_name, group_id)
