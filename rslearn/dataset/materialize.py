"""Classes to implement dataset materialization."""

from datetime import datetime
from typing import Any

import numpy as np
import numpy.typing as npt
from rasterio.enums import Resampling

from rslearn.config import (
    BandSetConfig,
    CompositingMethod,
    LayerConfig,
)
from rslearn.data_sources.data_source import ItemType
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import PixelBounds, Projection
from rslearn.utils.raster_array import RasterArray

from .remap import Remapper, load_remapper
from .window import Window


class Materializer:
    """An abstract class that materializes data from a tile store."""

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
            layer_name: the name of the layer to materialize
            layer_cfg: the configuration of the layer to materialize
            item_groups: the items associated with this window and layer
        """
        raise NotImplementedError


def read_raster_window_from_tiles(
    tile_store: TileStoreWithLayer,
    item: ItemType,
    bands: list[str],
    projection: Projection,
    bounds: PixelBounds,
    nodata_vals: list[float],
    band_dtype: npt.DTypeLike,
    remapper: Remapper | None = None,
    resampling: Resampling = Resampling.bilinear,
    dst: RasterArray | None = None,
) -> RasterArray | None:
    """Read an item's raster data from tiles into a window-aligned RasterArray.

    Handles band mapping and spatial intersection internally. Uses first-valid
    nodata logic: only overwrites pixels where all bands equal their nodata value.

    Args:
        tile_store: the TileStore to read from.
        item: the item to read.
        bands: the requested band names (determines dst band order).
        projection: the projection of the window.
        bounds: the pixel bounds of the window.
        nodata_vals: the nodata values for each requested band.
        band_dtype: data type for the output array.
        remapper: optional remapper to apply on the source pixel values.
        resampling: how to resample pixels if re-projection is needed.
        dst: optional pre-allocated RasterArray to write into. If None, a new
            one is allocated with T from the item's tile store data and H/W
            from the given bounds.

    Returns:
        The dst RasterArray (allocated or provided), or None if the item has no
        matching bands and dst was not provided.
    """
    needed = get_needed_band_sets_and_indexes(item, bands, tile_store)
    if not needed:
        return dst

    for src_bands, src_indexes, dst_indexes in needed:
        src_bounds = tile_store.get_raster_bounds(item.name, src_bands, projection)
        intersection = (
            max(bounds[0], src_bounds[0]),
            max(bounds[1], src_bounds[1]),
            min(bounds[2], src_bounds[2]),
            min(bounds[3], src_bounds[3]),
        )
        if intersection[2] <= intersection[0] or intersection[3] <= intersection[1]:
            continue

        raster_array = tile_store.read_raster(
            item.name, src_bands, projection, intersection, resampling=resampling
        )

        if (
            raster_array.timestamps is None
            and raster_array.array.shape[1] == 1
            and item.geometry.time_range is not None
        ):
            # The TileStore returned a single-timestep raster and the item had a time
            # range, but it didn't make it into the RasterArray.
            # This can happen with data sources implementing TileStore interface that
            # omit the time range for efficiency, since they just see the item name and
            # would need to perform a lookup to get the time range.
            # We add it in here.
            raster_array = RasterArray(
                array=raster_array.array,
                timestamps=[item.geometry.time_range],
            )

        src = raster_array.array  # (C_src, T, H_int, W_int)

        if dst is None:
            num_timesteps = src.shape[1]
            dst_arr = np.empty(
                (
                    len(bands),
                    num_timesteps,
                    bounds[3] - bounds[1],
                    bounds[2] - bounds[0],
                ),
                dtype=band_dtype,
            )
            for idx, nodata_val in enumerate(nodata_vals):
                dst_arr[idx, :, :, :] = nodata_val
            dst = RasterArray(array=dst_arr, timestamps=raster_array.timestamps)

        if src.shape[1] != dst.array.shape[1]:
            raise ValueError(
                f"Item {item.name!r} has T={src.shape[1]} in tile store but "
                f"dst has T={dst.array.shape[1]}"
            )

        dst_col = intersection[0] - bounds[0]
        dst_row = intersection[1] - bounds[1]

        src_sel = src[src_indexes, :, :, :]  # (C_sel, T, H_int, W_int)
        if remapper:
            src_sel = remapper(src_sel, band_dtype)

        out_crop = dst.array[
            :,
            :,
            dst_row : dst_row + src_sel.shape[2],
            dst_col : dst_col + src_sel.shape[3],
        ]  # (C, T, H_int, W_int) view

        # First-valid: only overwrite pixels where all dst bands are nodata.
        cur_nodata = np.array(
            [nodata_vals[dst_index] for dst_index in dst_indexes], dtype=band_dtype
        ).reshape(-1, 1, 1, 1)
        mask = (out_crop[dst_indexes] == cur_nodata).min(axis=0)  # (T, H_int, W_int)

        src_typed = src_sel.astype(band_dtype)
        for src_index, dst_index in enumerate(dst_indexes):
            out_crop[dst_index][mask] = src_typed[src_index][mask]

    return dst


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
    band_dtype: npt.DTypeLike,
    tile_store: TileStoreWithLayer,
    projection: Projection,
    remapper: Remapper | None,
    resampling_method: Resampling = Resampling.bilinear,
    request_time_range: tuple[datetime, datetime] | None = None,
) -> RasterArray:
    """Build a composite by selecting the first valid pixel of items in the group.

    A composite of shape (C, T, H, W) is created by iterating over items in group in
    order and selecting the first pixel that is not nodata per index. All items must
    have the same number of timesteps (T).

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
        request_time_range: unused, accepted for interface compatibility

    Returns:
        RasterArray with shape (C, T, H, W) built from all items in the group.

    """
    dst: RasterArray | None = None
    for item in group:
        dst = read_raster_window_from_tiles(
            tile_store=tile_store,
            item=item,
            bands=bands,
            projection=projection,
            bounds=bounds,
            nodata_vals=nodata_vals,
            band_dtype=band_dtype,
            remapper=remapper,
            resampling=resampling_method,
            dst=dst,
        )

    if dst is None:
        # Create a single-timestep nodata raster.
        height = bounds[3] - bounds[1]
        width = bounds[2] - bounds[0]
        arr = np.empty((len(bands), 1, height, width), dtype=band_dtype)
        for idx, nodata_val in enumerate(nodata_vals):
            arr[idx, :, :, :] = nodata_val
        dst = RasterArray(array=arr)

    return dst


def read_raster_windows(
    group: list[ItemType],
    bands: list[str],
    tile_store: TileStoreWithLayer,
    projection: Projection,
    bounds: PixelBounds,
    nodata_vals: list[Any],
    band_dtype: npt.DTypeLike,
    remapper: Remapper | None = None,
    resampling_method: Resampling = Resampling.bilinear,
) -> list[RasterArray]:
    """Read each item in the group into a window-aligned RasterArray.

    Each returned RasterArray has shape (C, T, H, W) where C = len(bands),
    T is determined by the item's data in the tile store, and H/W match the
    window bounds.

    Args:
        group: items to read. We create one RasterArray per item.
        bands: requested band names. The bands in the RasterArrays will appear in
            this order.
        tile_store: tile store containing raster data.
        projection: target projection.
        bounds: target pixel bounds.
        nodata_vals: nodata values for each band.
        band_dtype: output data type.
        remapper: optional remapper.
        resampling_method: resampling method.

    Returns:
        A list of RasterArrays, one per item that had matching bands.
    """
    results: list[RasterArray] = []
    for item in group:
        result = read_raster_window_from_tiles(
            tile_store=tile_store,
            item=item,
            bands=bands,
            projection=projection,
            bounds=bounds,
            nodata_vals=nodata_vals,
            band_dtype=band_dtype,
            remapper=remapper,
            resampling=resampling_method,
        )
        if result is not None:
            results.append(result)
    return results


def mask_stacked_rasters(
    stacked_rasters: npt.NDArray[np.generic],
    nodata_vals: list[Any],
) -> np.ma.MaskedArray:
    """Masks the stacked rasters - each items band with the corresponding nodata val.

    Args:
        stacked_rasters: numpy array of shape (num_items, num_bands, T, height, width)
            containing raster values for each item in the group.
        nodata_vals: Sequence of nodata values, one per band, used to identify invalid
            pixels in the stacked rasters.

    Returns:
        np.ma.MaskedArray with the same shape as `stacked_rasters`, where all
        pixels equal to the per-band nodata value are masked.
    """
    nodata_vals_array = np.array(nodata_vals).reshape(1, -1, 1, 1, 1)
    valid_mask = stacked_rasters != nodata_vals_array

    # Create masked array for all bands
    masked_data = np.ma.masked_where(~valid_mask, stacked_rasters)

    return masked_data


def build_mean_composite(
    group: list[ItemType],
    nodata_vals: list[Any],
    bands: list[str],
    bounds: PixelBounds,
    band_dtype: npt.DTypeLike,
    tile_store: TileStoreWithLayer,
    projection: Projection,
    remapper: Remapper | None,
    resampling_method: Resampling = Resampling.bilinear,
    request_time_range: tuple[datetime, datetime] | None = None,
) -> RasterArray:
    """Build a composite by computing the mean of valid pixels across items in the group.

    A RasterArray with shape (C, T, H, W) is created by computing the per-pixel mean of
    valid (non-nodata) pixels across all items in the group. All items must have the same
    number of timesteps T.

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
        request_time_range: unused, accepted for interface compatibility

    Returns:
        RasterArray with shape (C, T, H, W) having per-pixel mean of all items.
    """
    rasters = read_raster_windows(
        group=group,
        bands=bands,
        tile_store=tile_store,
        projection=projection,
        bounds=bounds,
        nodata_vals=nodata_vals,
        band_dtype=band_dtype,
        remapper=remapper,
        resampling_method=resampling_method,
    )

    num_timesteps = rasters[0].array.shape[1]
    for raster in rasters[1:]:
        if raster.array.shape[1] != num_timesteps:
            raise ValueError(
                f"All items must have the same number of timesteps, "
                f"got T={num_timesteps} and T={raster.array.shape[1]}"
            )

    # Stack into (N_items, C, T, H, W) and mask nodata values per band.
    stacked_arrays = np.stack([raster.array for raster in rasters], axis=0)
    masked_data = mask_stacked_rasters(stacked_arrays, nodata_vals)

    # Compute mean along the items axis.
    mean_result = np.ma.mean(masked_data, axis=0)

    # Fill masked values and convert to target dtype.
    fill_vals = np.array(nodata_vals).reshape(-1, 1, 1, 1)
    cthw = np.ma.filled(mean_result, fill_value=fill_vals).astype(band_dtype)
    return RasterArray(array=cthw, timestamps=rasters[0].timestamps)


def build_median_composite(
    group: list[ItemType],
    nodata_vals: list[Any],
    bands: list[str],
    bounds: PixelBounds,
    band_dtype: npt.DTypeLike,
    tile_store: TileStoreWithLayer,
    projection: Projection,
    remapper: Remapper | None,
    resampling_method: Resampling = Resampling.bilinear,
    request_time_range: tuple[datetime, datetime] | None = None,
) -> RasterArray:
    """Build a composite by computing the median of valid pixels across items in the group.

    A RasterArray with shape (C, T, H, W) is created by computing the per-pixel median of
    valid (non-nodata) pixels across all items in the group. All items must have the same
    number of timesteps T.

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
        request_time_range: unused, accepted for interface compatibility

    Returns:
        RasterArray with shape (C, T, H, W) having per-pixel median of all items.
    """
    rasters = read_raster_windows(
        group=group,
        bands=bands,
        tile_store=tile_store,
        projection=projection,
        bounds=bounds,
        nodata_vals=nodata_vals,
        band_dtype=band_dtype,
        remapper=remapper,
        resampling_method=resampling_method,
    )

    num_timesteps = rasters[0].array.shape[1]
    for raster in rasters[1:]:
        if raster.array.shape[1] != num_timesteps:
            raise ValueError(
                f"All items must have the same number of timesteps, "
                f"got T={num_timesteps} and T={raster.array.shape[1]}"
            )

    # Stack into (N_items, C, T, H, W) and mask nodata values per band.
    stacked_arrays = np.stack([raster.array for raster in rasters], axis=0)
    masked_data = mask_stacked_rasters(stacked_arrays, nodata_vals)

    # Compute median along the items axis.
    median_result = np.ma.median(masked_data, axis=0)

    # Fill masked values and convert to target dtype.
    fill_vals = np.array(nodata_vals).reshape(-1, 1, 1, 1)
    cthw = np.ma.filled(median_result, fill_value=fill_vals).astype(band_dtype)
    return RasterArray(array=cthw, timestamps=rasters[0].timestamps)


def build_temporal_stack_composite(
    group: list[ItemType],
    nodata_vals: list[Any],
    bands: list[str],
    bounds: PixelBounds,
    band_dtype: npt.DTypeLike,
    tile_store: TileStoreWithLayer,
    projection: Projection,
    remapper: Remapper | None,
    resampling_method: Resampling = Resampling.bilinear,
    request_time_range: tuple[datetime, datetime] | None = None,
) -> RasterArray:
    """Build a CTHW RasterArray by reading items' multi-timestep rasters.

    Uses first-valid spatial compositing: each item is read into a window-aligned
    RasterArray via read_raster_windows, then the per-timestep data is merged
    into a single output array.

    1. Read all items into window-aligned RasterArrays (handles band mapping and
       spatial intersection internally).
    2. Clip each raster's timestamps to the request time range (if provided).
    3. Compute the union of timestamps across items and build a sorted list with
       an index mapping.
    4. Allocate a single (C, T, H, W) output array initialized to nodata.
    5. For each item, write its data into the output array at the mapped timestep
       indices using first-valid logic.

    Args:
        group: items in the item group (may span multiple timesteps).
        nodata_vals: nodata values for each band.
        bands: requested band names.
        bounds: target pixel bounds.
        band_dtype: output data type.
        tile_store: tile store containing raster data.
        projection: target projection.
        remapper: optional remapper.
        resampling_method: resampling method.
        request_time_range: if provided, timesteps that don't intersect this range
            are excluded from the output.

    Returns:
        A RasterArray with shape (C, T, H, W) and associated timestamps.
    """
    height = bounds[3] - bounds[1]
    width = bounds[2] - bounds[0]

    # --- Step 1: read all items into window-aligned RasterArrays. ---
    rasters = read_raster_windows(
        group=group,
        bands=bands,
        tile_store=tile_store,
        projection=projection,
        bounds=bounds,
        nodata_vals=nodata_vals,
        band_dtype=band_dtype,
        remapper=remapper,
        resampling_method=resampling_method,
    )

    if not rasters:
        raise ValueError("No valid items found for temporal stack")

    # --- Step 2: clip each raster's timestamps to the request time range. ---
    if request_time_range is not None:
        w_start, w_end = request_time_range
        clipped_rasters: list[RasterArray] = []
        for raster in rasters:
            if raster.timestamps is None:
                raise ValueError(
                    "SPATIAL_MOSAIC_TEMPORAL_STACK requires items to have timestamps"
                )
            keep_indices = [
                timestep_idx
                for timestep_idx, (ts_start, ts_end) in enumerate(raster.timestamps)
                if ts_start < w_end and ts_end > w_start
            ]
            if not keep_indices:
                continue
            kept_timestamps = [
                raster.timestamps[timestep_idx] for timestep_idx in keep_indices
            ]
            kept_array = raster.array[:, keep_indices, :, :]
            clipped_rasters.append(
                RasterArray(array=kept_array, timestamps=kept_timestamps)
            )
        rasters = clipped_rasters

    if not rasters:
        raise ValueError("No valid timesteps after clipping to request time range")

    # --- Step 3: collect the union of all per-timestep timestamps. ---
    all_timestamps: set[tuple[datetime, datetime]] = set()
    for raster in rasters:
        if raster.timestamps is None:
            raise ValueError(
                "SPATIAL_MOSAIC_TEMPORAL_STACK requires items to have timestamps"
            )
        all_timestamps.update(raster.timestamps)

    sorted_timestamps = sorted(all_timestamps)
    time_range_to_timestep_index = {
        time_range: idx for idx, time_range in enumerate(sorted_timestamps)
    }
    num_timesteps = len(sorted_timestamps)

    # --- Step 4: allocate the output CTHW array, initialized to nodata. ---
    output = np.empty((len(bands), num_timesteps, height, width), dtype=band_dtype)
    for band_idx, nodata_val in enumerate(nodata_vals):
        output[band_idx, :, :, :] = nodata_val
    nodata_arr = np.array(nodata_vals, dtype=band_dtype).reshape(-1, 1, 1, 1)

    # --- Step 5: write each item's data into the output at correct timestep slots. ---
    for raster in rasters:
        assert raster.timestamps is not None
        output_timestep_indices = [
            time_range_to_timestep_index[time_range] for time_range in raster.timestamps
        ]
        dst_slice = output[:, output_timestep_indices, :, :]  # (C, T_item, H, W) copy
        src_slice = raster.array  # (C, T_item, H, W)

        # First-valid: only overwrite pixels where all bands are nodata.
        # mask is 1 at pixels/timesteps that are nodata, so that np.where will use
        # src_slice, and 0 elsewhere, so that np.where will use dst_slice.
        mask = (dst_slice == nodata_arr).min(axis=0)  # (T_item, H, W)
        output[:, output_timestep_indices, :, :] = np.where(
            mask[np.newaxis], src_slice, dst_slice
        )

    return RasterArray(array=output, timestamps=sorted_timestamps)


compositing_methods = {
    CompositingMethod.FIRST_VALID: build_first_valid_composite,
    CompositingMethod.MEAN: build_mean_composite,
    CompositingMethod.MEDIAN: build_median_composite,
    CompositingMethod.SPATIAL_MOSAIC_TEMPORAL_STACK: build_temporal_stack_composite,
}


def build_composite(
    group: list[ItemType],
    compositing_method: CompositingMethod,
    tile_store: TileStoreWithLayer,
    layer_cfg: LayerConfig,
    band_cfg: BandSetConfig,
    projection: Projection,
    bounds: PixelBounds,
    remapper: Remapper | None,
    request_time_range: tuple[datetime, datetime] | None = None,
) -> RasterArray:
    """Build a composite for specified bands from items in the group.

    Args:
        group: list of items to composite together
        compositing_method: Which method to use for compositing.
        tile_store: tile store containing the raster data
        layer_cfg: the configuration of the layer to materialize
        band_cfg: the configuration of the layer to materialize. Contains the bands to process.
        projection: spatial projection for the composite
        bounds: pixel bounds defining the spatial extent of the composite
        remapper: remapper to apply to pixel values, or None
        request_time_range: optional request time range, passed through to compositing method.

    Returns:
        A RasterArray produced by the chosen compositing method.
    """
    nodata_vals = band_cfg.nodata_vals
    if nodata_vals is None:
        nodata_vals = [0 for _ in band_cfg.bands]

    return compositing_methods[compositing_method](
        group=group,
        nodata_vals=nodata_vals,
        bands=band_cfg.bands,
        bounds=bounds,
        band_dtype=band_cfg.dtype.get_numpy_dtype(),
        tile_store=tile_store,
        projection=projection,
        resampling_method=layer_cfg.resampling_method.get_rasterio_resampling(),
        remapper=remapper,
        request_time_range=request_time_range,
    )


class RasterMaterializer(Materializer):
    """A Materializer for raster data."""

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
            tile_store: the tile store where the items have been ingested
            window: the window to materialize
            layer_name: name of the layer to materialize
            layer_cfg: the configuration of the layer to materialize
            item_groups: the items associated with this window and layer
        """
        # Compute the request time range: the window time range adjusted by
        # time_offset/duration from the data source config.
        if layer_cfg.data_source is not None:
            request_time_range = layer_cfg.data_source.get_request_time_range(
                window.time_range
            )
        else:
            request_time_range = window.time_range

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

            raster_format = band_cfg.instantiate_raster_format()

            for group_id, group in enumerate(item_groups):
                raster = build_composite(
                    group=group,
                    compositing_method=layer_cfg.compositing_method,
                    tile_store=tile_store,
                    layer_cfg=layer_cfg,
                    band_cfg=band_cfg,
                    projection=projection,
                    bounds=bounds,
                    remapper=remapper,
                    request_time_range=request_time_range,
                )

                raster_format.encode_raster(
                    window.get_raster_dir(layer_name, band_cfg.bands, group_id),
                    projection,
                    bounds,
                    raster,
                )

        for group_id in range(len(item_groups)):
            window.mark_layer_completed(layer_name, group_id)


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
        vector_format = layer_cfg.instantiate_vector_format()

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
