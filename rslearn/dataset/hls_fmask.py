"""HLS Fmask-based compositor for cloud-aware FIRST_VALID compositing."""

import math
from collections.abc import Iterator
from datetime import datetime
from typing import Literal

import numpy as np
import numpy.typing as npt
from rasterio.enums import Resampling

from rslearn.data_sources.data_source import ItemType
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.geometry import PixelBounds, Projection
from rslearn.utils.raster_array import RasterArray

from .compositing import BandSetCompositeRequest, Compositor, FirstValidCompositor
from .remap import Remapper
from .tile_utils import get_needed_band_sets_and_indexes, read_raster_window_from_tiles
from .window import Window

logger = get_logger(__name__)


class HlsFmaskFirstValid(Compositor):
    """Sort items by HLS Fmask cloudiness, then apply FIRST_VALID compositing.

    This compositor reads the HLS ``Fmask``/``fmask`` QA band for each item and scores
    cloudiness using cloud and cirrus bit fractions over the requested window.
    Lower scores are preferred.
    """

    CIRRUS_BIT = 1 << 0
    CLOUD_BIT = 1 << 1

    def __init__(
        self,
        fmask_band: str = "fmask",
        fmask_nodata_value: int = 255,
        scoring_resolution: float | None = None,
        on_missing_fmask: Literal["error", "skip_item"] = "error",
        cirrus_weight: int = 1,
        cloud_weight: int = 5,
    ) -> None:
        """Create a new HlsFmaskFirstValid compositor.

        Args:
            fmask_band: band name for HLS QA mask ("fmask" for Hls2, "Fmask" for
                Hls2S30/Hls2L30).
            fmask_nodata_value: nodata code in the HLS Fmask raster.
            scoring_resolution: optional pixel size for a window-level scoring grid.
            on_missing_fmask: whether missing Fmask raises an error or drops an item.
            cirrus_weight: weight for cirrus bit (bit 0).
            cloud_weight: weight for cloud bit (bit 1).
        """
        self.fmask_band = fmask_band
        self.fmask_nodata_value = fmask_nodata_value
        self.scoring_resolution = scoring_resolution
        self.on_missing_fmask = on_missing_fmask
        self.cirrus_weight = cirrus_weight
        self.cloud_weight = cloud_weight

        if self.scoring_resolution is not None and self.scoring_resolution <= 0:
            raise ValueError("scoring_resolution must be positive")

    def _rescale_grid(
        self,
        projection: Projection,
        bounds: PixelBounds,
        target_resolution: float,
    ) -> tuple[Projection, PixelBounds]:
        """Return a grid with the same spatial extent at a different resolution."""
        target_x_resolution = math.copysign(target_resolution, projection.x_resolution)
        target_y_resolution = math.copysign(target_resolution, projection.y_resolution)
        x_factor = abs(projection.x_resolution) / target_resolution
        y_factor = abs(projection.y_resolution) / target_resolution
        target_width = round((bounds[2] - bounds[0]) * x_factor)
        target_height = round((bounds[3] - bounds[1]) * y_factor)
        target_left = round(bounds[0] * x_factor)
        target_top = round(bounds[1] * y_factor)
        return (
            Projection(
                projection.crs,
                target_x_resolution,
                target_y_resolution,
            ),
            (
                target_left,
                target_top,
                target_left + target_width,
                target_top + target_height,
            ),
        )

    def _get_scoring_grid(
        self,
        projection: Projection,
        bounds: PixelBounds,
        window: Window | None = None,
    ) -> tuple[Projection, PixelBounds]:
        """Get the grid on which cloud ranking should be evaluated."""
        base_projection = window.projection if window is not None else projection
        base_bounds = window.bounds if window is not None else bounds
        if self.scoring_resolution is None:
            return base_projection, base_bounds
        return self._rescale_grid(
            base_projection, base_bounds, self.scoring_resolution
        )

    def _score_item(
        self,
        item: ItemType,
        tile_store: TileStoreWithLayer,
        projection: Projection,
        bounds: PixelBounds,
        resampling_method: Resampling,
    ) -> float | None:
        """Score a single item using HLS Fmask class fractions."""
        scoring_bands = [self.fmask_band]
        needed_band_sets_and_indexes = get_needed_band_sets_and_indexes(
            item, scoring_bands, tile_store
        )
        if len(needed_band_sets_and_indexes) == 0:
            if self.on_missing_fmask == "skip_item":
                logger.debug(
                    "missing Fmask scoring bands %s for item %s",
                    scoring_bands,
                    item.name,
                )
                return None
            raise ValueError(
                f"missing scoring bands {scoring_bands} for item {item.name}"
            )

        raster = read_raster_window_from_tiles(
            tile_store=tile_store,
            item=item,
            bands=scoring_bands,
            projection=projection,
            bounds=bounds,
            nodata_val=self.fmask_nodata_value,
            band_dtype=np.uint8,
            resampling=resampling_method,
        )
        if raster is None:
            return None

        fmask = raster.array[0, 0, :, :].astype(np.uint8, copy=False)
        valid = fmask != self.fmask_nodata_value
        valid_count = int(valid.sum())
        if valid_count == 0:
            return None

        cirrus_frac = float(np.count_nonzero(valid & ((fmask & self.CIRRUS_BIT) != 0)))
        cloud_frac = float(np.count_nonzero(valid & ((fmask & self.CLOUD_BIT) != 0)))

        cirrus_frac /= valid_count
        cloud_frac /= valid_count

        score = cirrus_frac * self.cirrus_weight + cloud_frac * self.cloud_weight
        logger.debug(
            "HLS Fmask for %s: cirrus=%.3f cloud=%.3f score=%.3f",
            item.name,
            cirrus_frac,
            cloud_frac,
            score,
        )
        return score

    def _sort_group(
        self,
        group: list[ItemType],
        tile_store: TileStoreWithLayer,
        scoring_projection: Projection,
        scoring_bounds: PixelBounds,
        resampling_method: Resampling,
    ) -> list[ItemType]:
        """Sort a group once for the requested scoring grid."""
        scored: list[tuple[float, ItemType]] = []
        for item in group:
            score = self._score_item(
                item,
                tile_store,
                scoring_projection,
                scoring_bounds,
                resampling_method,
            )
            if score is None:
                logger.debug("no usable HLS Fmask score for item %s", item.name)
                continue
            scored.append((score, item))

        scored.sort(key=lambda t: t[0])
        return [item for _, item in scored]

    def build_composites(
        self,
        group: list[ItemType],
        requests: list[BandSetCompositeRequest],
        tile_store: TileStoreWithLayer,
        window: Window | None = None,
        request_time_range: tuple[datetime, datetime] | None = None,
    ) -> Iterator[RasterArray]:
        """Yield composites for all band sets, sharing ranking work when possible."""
        sorted_groups: dict[
            tuple[Projection, PixelBounds, Resampling],
            list[ItemType],
        ] = {}

        for request in requests:
            cur_group = group
            if len(group) > 1:
                scoring_projection, scoring_bounds = self._get_scoring_grid(
                    request.projection,
                    request.bounds,
                    window=window,
                )
                cache_key = (
                    scoring_projection,
                    scoring_bounds,
                    request.resampling_method,
                )
                if cache_key in sorted_groups:
                    cur_group = sorted_groups[cache_key]
                else:
                    cur_group = self._sort_group(
                        group,
                        tile_store,
                        scoring_projection,
                        scoring_bounds,
                        request.resampling_method,
                    )
                    sorted_groups[cache_key] = cur_group

            yield FirstValidCompositor().build_composite(
                group=cur_group,
                nodata_val=request.nodata_val,
                bands=request.bands,
                bounds=request.bounds,
                band_dtype=request.band_dtype,
                tile_store=tile_store,
                projection=request.projection,
                resampling_method=request.resampling_method,
                remapper=request.remapper,
                request_time_range=request_time_range,
            )

    def build_composite(
        self,
        group: list[ItemType],
        nodata_val: int | float | None,
        bands: list[str],
        bounds: PixelBounds,
        band_dtype: npt.DTypeLike,
        tile_store: TileStoreWithLayer,
        projection: Projection,
        resampling_method: Resampling,
        remapper: Remapper | None,
        request_time_range: tuple[datetime, datetime] | None = None,
    ) -> RasterArray:
        """Score items by Fmask cloudiness, sort, then delegate to FIRST_VALID."""
        if len(group) > 1:
            scoring_projection, scoring_bounds = self._get_scoring_grid(
                projection, bounds
            )
            group = self._sort_group(
                group,
                tile_store,
                scoring_projection,
                scoring_bounds,
                resampling_method,
            )

        return FirstValidCompositor().build_composite(
            group=group,
            nodata_val=nodata_val,
            bands=bands,
            bounds=bounds,
            band_dtype=band_dtype,
            tile_store=tile_store,
            projection=projection,
            resampling_method=resampling_method,
            remapper=remapper,
            request_time_range=request_time_range,
        )
