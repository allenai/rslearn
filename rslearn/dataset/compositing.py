"""Built-in and abstract compositing methods for raster materialization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from rasterio.enums import Resampling

from rslearn.config import CompositingMethod
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.array import nodata_eq
from rslearn.utils.geometry import PixelBounds, Projection
from rslearn.utils.raster_array import RasterArray, RasterMetadata

from .remap import Remapper
from .tile_utils import (
    mask_stacked_rasters,
    read_raster_window_from_tiles,
    read_raster_windows,
)

if TYPE_CHECKING:
    from rslearn.data_sources.data_source import ItemType

logger = get_logger(__name__)

_NODATA_REQUIRED_MSG = (
    "{name} compositing requires explicit nodata values. "
    "Set nodata_vals in BandSetConfig or use a data source that "
    "provides nodata metadata."
)


class Compositor(ABC):
    """Abstract base for compositing methods.

    All built-in compositing methods and custom (jsonargparse-injectable) ones
    share this interface.
    """

    @abstractmethod
    def build_composite(
        self,
        group: list[ItemType],
        nodata_vals: Sequence[int | float] | None,
        bands: list[str],
        bounds: PixelBounds,
        band_dtype: npt.DTypeLike,
        tile_store: TileStoreWithLayer,
        projection: Projection,
        resampling_method: Resampling,
        remapper: Remapper | None,
        request_time_range: tuple[datetime, datetime] | None = None,
    ) -> RasterArray:
        """Build a composite from items in the group.

        Args:
            group: list of items to composite together.
            nodata_vals: per-band nodata values.
            bands: band names to include in the composite.
            bounds: pixel bounds for the spatial extent.
            band_dtype: numpy dtype for the output.
            tile_store: tile store for reading raster data.
            projection: target spatial projection.
            resampling_method: rasterio resampling enum.
            remapper: optional pixel-value remapper.
            request_time_range: optional time range for the request.

        Returns:
            A RasterArray produced by this compositing method.
        """
        ...


class FirstValidCompositor(Compositor):
    """Select the first valid (non-nodata) pixel across items in order."""

    def build_composite(
        self,
        group: list[ItemType],
        nodata_vals: Sequence[int | float] | None,
        bands: list[str],
        bounds: PixelBounds,
        band_dtype: npt.DTypeLike,
        tile_store: TileStoreWithLayer,
        projection: Projection,
        resampling_method: Resampling,
        remapper: Remapper | None,
        request_time_range: tuple[datetime, datetime] | None = None,
    ) -> RasterArray:
        """Build a first-valid composite."""
        # When the source has no declared nodata, fall back to 0 for the
        # first-valid merge logic.  The output metadata will still reflect
        # the *original* nodata_vals (None → no nodata tag on the file).
        effective_nodata = nodata_vals
        if effective_nodata is None:
            logger.warning(
                "No nodata values available for FIRST_VALID compositing; "
                "defaulting to 0 for merge logic. Set nodata_vals in "
                "BandSetConfig or use a data source with nodata metadata "
                "to avoid this warning."
            )
            effective_nodata = (0,) * len(bands)

        dst: RasterArray | None = None
        for item in group:
            dst = read_raster_window_from_tiles(
                tile_store=tile_store,
                item=item,
                bands=bands,
                projection=projection,
                bounds=bounds,
                nodata_vals=effective_nodata,
                band_dtype=band_dtype,
                remapper=remapper,
                resampling=resampling_method,
                dst=dst,
            )

        if dst is None:
            height = bounds[3] - bounds[1]
            width = bounds[2] - bounds[0]
            arr = np.zeros((len(bands), 1, height, width), dtype=band_dtype)
            dst = RasterArray(array=arr)

        dst.metadata.nodata_values = (
            tuple(nodata_vals) if nodata_vals is not None else None
        )
        return dst


class MeanCompositor(Compositor):
    """Per-pixel mean of valid (non-nodata) values across items."""

    def build_composite(
        self,
        group: list[ItemType],
        nodata_vals: Sequence[int | float] | None,
        bands: list[str],
        bounds: PixelBounds,
        band_dtype: npt.DTypeLike,
        tile_store: TileStoreWithLayer,
        projection: Projection,
        resampling_method: Resampling,
        remapper: Remapper | None,
        request_time_range: tuple[datetime, datetime] | None = None,
    ) -> RasterArray:
        """Build a mean composite."""
        if nodata_vals is None:
            raise ValueError(_NODATA_REQUIRED_MSG.format(name="MEAN"))
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

        stacked_arrays = np.stack([r.array for r in rasters], axis=0)
        masked_data = mask_stacked_rasters(stacked_arrays, nodata_vals)
        mean_result = np.ma.mean(masked_data, axis=0)

        fill_vals = np.array(nodata_vals).reshape(-1, 1, 1, 1)
        cthw = np.ma.filled(mean_result, fill_value=fill_vals).astype(band_dtype)
        metadata = RasterMetadata(nodata_values=tuple(nodata_vals))
        return RasterArray(
            array=cthw, timestamps=rasters[0].timestamps, metadata=metadata
        )


class MedianCompositor(Compositor):
    """Per-pixel median of valid (non-nodata) values across items."""

    def build_composite(
        self,
        group: list[ItemType],
        nodata_vals: Sequence[int | float] | None,
        bands: list[str],
        bounds: PixelBounds,
        band_dtype: npt.DTypeLike,
        tile_store: TileStoreWithLayer,
        projection: Projection,
        resampling_method: Resampling,
        remapper: Remapper | None,
        request_time_range: tuple[datetime, datetime] | None = None,
    ) -> RasterArray:
        """Build a median composite."""
        if nodata_vals is None:
            raise ValueError(_NODATA_REQUIRED_MSG.format(name="MEDIAN"))
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

        stacked_arrays = np.stack([r.array for r in rasters], axis=0)
        masked_data = mask_stacked_rasters(stacked_arrays, nodata_vals)
        median_result = np.ma.median(masked_data, axis=0)

        fill_vals = np.array(nodata_vals).reshape(-1, 1, 1, 1)
        cthw = np.ma.filled(median_result, fill_value=fill_vals).astype(band_dtype)
        metadata = RasterMetadata(nodata_values=tuple(nodata_vals))
        return RasterArray(
            array=cthw, timestamps=rasters[0].timestamps, metadata=metadata
        )


class SpatialMosaicTemporalStackCompositor(Compositor):
    """Spatial first-valid compositing per timestep, stacked along T."""

    def build_composite(
        self,
        group: list[ItemType],
        nodata_vals: Sequence[int | float] | None,
        bands: list[str],
        bounds: PixelBounds,
        band_dtype: npt.DTypeLike,
        tile_store: TileStoreWithLayer,
        projection: Projection,
        resampling_method: Resampling,
        remapper: Remapper | None,
        request_time_range: tuple[datetime, datetime] | None = None,
    ) -> RasterArray:
        """Build a spatial-mosaic temporal-stack composite."""
        if nodata_vals is None:
            raise ValueError(
                _NODATA_REQUIRED_MSG.format(name="SPATIAL_MOSAIC_TEMPORAL_STACK")
            )
        height = bounds[3] - bounds[1]
        width = bounds[2] - bounds[0]

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

        # Clip timestamps to request time range.
        if request_time_range is not None:
            w_start, w_end = request_time_range
            clipped: list[RasterArray] = []
            for raster in rasters:
                if raster.timestamps is None:
                    raise ValueError(
                        "SPATIAL_MOSAIC_TEMPORAL_STACK requires items to have timestamps"
                    )
                keep = [
                    i
                    for i, (ts_s, ts_e) in enumerate(raster.timestamps)
                    if ts_s < w_end and ts_e > w_start
                ]
                if not keep:
                    continue
                clipped.append(
                    RasterArray(
                        array=raster.array[:, keep, :, :],
                        timestamps=[raster.timestamps[i] for i in keep],
                        metadata=RasterMetadata(nodata_values=tuple(nodata_vals)),
                    )
                )
            rasters = clipped

        if not rasters:
            raise ValueError("No valid timesteps after clipping to request time range")

        # Union of timestamps across all items.
        all_timestamps: set[tuple[datetime, datetime]] = set()
        for raster in rasters:
            if raster.timestamps is None:
                raise ValueError(
                    "SPATIAL_MOSAIC_TEMPORAL_STACK requires items to have timestamps"
                )
            all_timestamps.update(raster.timestamps)

        sorted_timestamps = sorted(all_timestamps)
        ts_to_idx = {tr: idx for idx, tr in enumerate(sorted_timestamps)}
        num_timesteps = len(sorted_timestamps)

        output = np.empty((len(bands), num_timesteps, height, width), dtype=band_dtype)
        for band_idx, nodata_val in enumerate(nodata_vals):
            output[band_idx, :, :, :] = nodata_val
        nodata_arr = np.array(nodata_vals, dtype=band_dtype).reshape(-1, 1, 1, 1)

        for raster in rasters:
            assert raster.timestamps is not None
            out_idxs = [ts_to_idx[tr] for tr in raster.timestamps]
            dst_slice = output[:, out_idxs, :, :]
            src_slice = raster.array

            mask = nodata_eq(dst_slice, nodata_arr).min(axis=0)
            output[:, out_idxs, :, :] = np.where(mask[np.newaxis], src_slice, dst_slice)

        metadata = RasterMetadata(nodata_values=tuple(nodata_vals))
        return RasterArray(
            array=output, timestamps=sorted_timestamps, metadata=metadata
        )


class _TemporalReducerCompositor(Compositor):
    """Base for temporal reducers that build a stack then reduce along T.

    Subclasses implement `reduce` to collapse the temporal axis of a masked array.
    """

    @abstractmethod
    def reduce(self, masked_data: np.ma.MaskedArray) -> np.ma.MaskedArray:
        """Reduce a (C, T, H, W) masked array along the T axis (axis=1).

        Args:
            masked_data: masked array with shape (C, T, H, W) where nodata
                pixels are masked.

        Returns:
            Reduced masked array with shape (C, H, W).
        """
        ...

    def build_composite(
        self,
        group: list[ItemType],
        nodata_vals: Sequence[int | float] | None,
        bands: list[str],
        bounds: PixelBounds,
        band_dtype: npt.DTypeLike,
        tile_store: TileStoreWithLayer,
        projection: Projection,
        resampling_method: Resampling,
        remapper: Remapper | None,
        request_time_range: tuple[datetime, datetime] | None = None,
    ) -> RasterArray:
        """Build a temporal-stack then reduce along T."""
        if nodata_vals is None:
            raise ValueError(_NODATA_REQUIRED_MSG.format(name="temporal reducer"))
        stacked = SpatialMosaicTemporalStackCompositor().build_composite(
            group=group,
            nodata_vals=nodata_vals,
            bands=bands,
            bounds=bounds,
            band_dtype=band_dtype,
            tile_store=tile_store,
            projection=projection,
            resampling_method=resampling_method,
            remapper=remapper,
            request_time_range=request_time_range,
        )

        nodata_arr = np.array(nodata_vals, dtype=band_dtype).reshape(-1, 1, 1, 1)
        masked_data = np.ma.masked_where(
            nodata_eq(stacked.array, nodata_arr), stacked.array
        )
        reduced = self.reduce(masked_data)

        fill_vals = np.array(nodata_vals, dtype=band_dtype).reshape(-1, 1, 1)
        chw = np.ma.filled(reduced, fill_value=fill_vals).astype(band_dtype)

        output_time_range = request_time_range
        if output_time_range is None and stacked.timestamps:
            output_time_range = (
                min(tr[0] for tr in stacked.timestamps),
                max(tr[1] for tr in stacked.timestamps),
            )

        return RasterArray(
            chw_array=chw, time_range=output_time_range, metadata=stacked.metadata
        )


class TemporalMeanCompositor(_TemporalReducerCompositor):
    """Reduce a multi-temporal raster stack to one timestep via temporal mean."""

    def reduce(self, masked_data: np.ma.MaskedArray) -> np.ma.MaskedArray:
        """Reduce along T via mean."""
        return np.ma.mean(masked_data, axis=1)


class TemporalMaxCompositor(_TemporalReducerCompositor):
    """Reduce a multi-temporal raster stack to one timestep via temporal max."""

    def reduce(self, masked_data: np.ma.MaskedArray) -> np.ma.MaskedArray:
        """Reduce along T via max."""
        return np.ma.max(masked_data, axis=1)


class TemporalMinCompositor(_TemporalReducerCompositor):
    """Reduce a multi-temporal raster stack to one timestep via temporal min."""

    def reduce(self, masked_data: np.ma.MaskedArray) -> np.ma.MaskedArray:
        """Reduce along T via min."""
        return np.ma.min(masked_data, axis=1)


BUILTIN_COMPOSITORS: dict[CompositingMethod, Compositor] = {
    CompositingMethod.FIRST_VALID: FirstValidCompositor(),
    CompositingMethod.MEAN: MeanCompositor(),
    CompositingMethod.MEDIAN: MedianCompositor(),
    CompositingMethod.SPATIAL_MOSAIC_TEMPORAL_STACK: SpatialMosaicTemporalStackCompositor(),
    CompositingMethod.TEMPORAL_MEAN: TemporalMeanCompositor(),
    CompositingMethod.TEMPORAL_MAX: TemporalMaxCompositor(),
    CompositingMethod.TEMPORAL_MIN: TemporalMinCompositor(),
}
