"""Low-level helpers for reading raster data from tile stores."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from rasterio.enums import Resampling

from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.geometry import PixelBounds, Projection
from rslearn.utils.raster_array import RasterArray

from .remap import Remapper

if TYPE_CHECKING:
    from rslearn.data_sources.data_source import ItemType


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
    wanted_band_indexes = {}
    for i, band in enumerate(bands):
        wanted_band_indexes[band] = i

    available_bands = tile_store.get_raster_bands(item)
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
        return []

    return needed_band_sets_and_indexes


def read_raster_window_from_tiles(
    tile_store: TileStoreWithLayer,
    item: ItemType,
    bands: list[str],
    projection: Projection,
    bounds: PixelBounds,
    nodata_vals: Sequence[int | float],
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
        src_bounds = tile_store.get_raster_bounds(item, src_bands, projection)
        intersection = (
            max(bounds[0], src_bounds[0]),
            max(bounds[1], src_bounds[1]),
            min(bounds[2], src_bounds[2]),
            min(bounds[3], src_bounds[3]),
        )
        if intersection[2] <= intersection[0] or intersection[3] <= intersection[1]:
            continue

        raster_array = tile_store.read_raster(
            item, src_bands, projection, intersection, resampling=resampling
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


def read_raster_windows(
    group: list[ItemType],
    bands: list[str],
    tile_store: TileStoreWithLayer,
    projection: Projection,
    bounds: PixelBounds,
    nodata_vals: Sequence[int | float],
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
    nodata_vals: Sequence[int | float],
) -> np.ma.MaskedArray:
    """Mask stacked rasters -- each item's band with the corresponding nodata val.

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
    masked_data = np.ma.masked_where(~valid_mask, stacked_rasters)
    return masked_data
