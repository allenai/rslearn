"""Sentinel-2 SCL-based compositors."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import numpy.typing as npt
from rasterio.enums import Resampling

from rslearn.data_sources.data_source import ItemType
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.geometry import PixelBounds, Projection
from rslearn.utils.raster_array import RasterArray

from .compositing import Compositor, FirstValidCompositor
from .remap import Remapper
from .tile_utils import get_needed_band_sets_and_indexes, read_raster_window_from_tiles

logger = get_logger(__name__)

SENTINEL2_SCL_NODATA_VALUE = 0
"""Sentinel-2 SCL class for no data."""

SENTINEL2_SCL_CLEAR_VALUES = (4, 5, 6)
"""Default Sentinel-2 SCL classes treated as clear: vegetation, bare soil, water."""


@dataclass(frozen=True)
class SCLClearCoverScore:
    """Clear-cover score for one Sentinel-2 item."""

    clear_cover: float
    clear_fraction: float
    valid_cover: float
    clear_pixels: int
    valid_pixels: int
    total_pixels: int


class Sentinel2SCLFirstValid(Compositor):
    """Sort items by Sentinel-2 SCL cloudiness, then apply FIRST_VALID compositing.

    For each item in the group, this compositor reads the SCL (scene classification)
    band over the target window and computes a weighted cloudiness score:

    - 3: cloud shadow
    - 8: medium probability cloud
    - 9: high probability cloud
    - 10: thin cirrus
    - 11: snow/ice

    Items are sorted best-to-worst (lowest score first) and then composited with
    FIRST_VALID in that order.
    """

    def __init__(
        self,
        scl_band: str = "SCL",
        cloud_shadow_weight: int = 1,
        medium_cloud_weight: int = 1,
        high_cloud_weight: int = 5,
        cirrus_weight: int = 1,
        snow_ice_weight: int = 1,
    ) -> None:
        """Create a new Sentinel2SCLFirstValid compositor.

        Args:
            scl_band: band name for Sentinel-2 scene classification layer.
            cloud_shadow_weight: weight for SCL=3 pixels.
            medium_cloud_weight: weight for SCL=8 pixels.
            high_cloud_weight: weight for SCL=9 pixels.
            cirrus_weight: weight for SCL=10 pixels.
            snow_ice_weight: weight for SCL=11 pixels.
        """
        self.scl_band = scl_band
        self.cloud_shadow_weight = cloud_shadow_weight
        self.medium_cloud_weight = medium_cloud_weight
        self.high_cloud_weight = high_cloud_weight
        self.cirrus_weight = cirrus_weight
        self.snow_ice_weight = snow_ice_weight

    def _score_item(
        self,
        item: ItemType,
        tile_store: TileStoreWithLayer,
        projection: Projection,
        bounds: PixelBounds,
        resampling_method: Resampling,
    ) -> float | None:
        """Score a single item using SCL class fractions.

        Returns:
            Weighted cloudiness score where lower values are preferred. Returns None
            when no data is available at all for the requested window.
        """
        scoring_bands = [self.scl_band]

        needed_band_sets_and_indexes = get_needed_band_sets_and_indexes(
            item, scoring_bands, tile_store
        )
        if len(needed_band_sets_and_indexes) == 0:
            raise ValueError(
                f"missing scoring bands {scoring_bands} for item {item.name}"
            )

        raster = read_raster_window_from_tiles(
            tile_store=tile_store,
            item=item,
            bands=scoring_bands,
            projection=projection,
            bounds=bounds,
            nodata_val=SENTINEL2_SCL_NODATA_VALUE,
            band_dtype=np.uint8,
            resampling=resampling_method,
        )
        if raster is None:
            return None

        scl = raster.array[0, 0, :, :]
        shadow_frac = float((scl == 3).mean())
        medium_cloud_frac = float((scl == 8).mean())
        high_cloud_frac = float((scl == 9).mean())
        cirrus_frac = float((scl == 10).mean())
        snow_ice_frac = float((scl == 11).mean())

        score = (
            shadow_frac * self.cloud_shadow_weight
            + medium_cloud_frac * self.medium_cloud_weight
            + high_cloud_frac * self.high_cloud_weight
            + cirrus_frac * self.cirrus_weight
            + snow_ice_frac * self.snow_ice_weight
        )
        logger.debug(
            "Sentinel-2 SCL for %s: shadow=%.3f medium=%.3f high=%.3f "
            "cirrus=%.3f snow_ice=%.3f score=%.3f",
            item.name,
            shadow_frac,
            medium_cloud_frac,
            high_cloud_frac,
            cirrus_frac,
            snow_ice_frac,
            score,
        )
        return score

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
        """Score items by SCL cloudiness, sort, then delegate to FIRST_VALID."""
        if len(group) > 1:
            scored: list[tuple[float, ItemType]] = []
            for item in group:
                score = self._score_item(
                    item, tile_store, projection, bounds, resampling_method
                )
                if score is None:
                    # Missing image at the requested window. We skip this candidate
                    # since we cannot score it and it likely also has no usable data
                    # for the final FIRST_VALID compositing pass.
                    logger.debug(
                        "no data for Sentinel-2 SCL scoring of item %s", item.name
                    )
                    continue
                scored.append((score, item))

            scored.sort(key=lambda t: t[0])
            group = [item for _, item in scored]

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


class Sentinel2SCLBestClear(Compositor):
    """Select the single Sentinel-2 item with highest SCL clear cover.

    For each item in the group, this compositor reads the Sentinel-2 Scene
    Classification Layer (SCL) over the target window and computes:

    - clear_cover: pixels with SCL in ``clear_values`` divided by total pixels
    - clear_fraction: pixels with SCL in ``clear_values`` divided by valid pixels
    - valid_cover: pixels not equal to SCL nodata divided by total pixels

    The item with the highest clear cover wins. Ties are broken by valid cover, then
    raw clear pixel count. Only the selected item is materialized; pixels are not
    filled from later items.
    """

    DEFAULT_CLEAR_VALUES = SENTINEL2_SCL_CLEAR_VALUES

    def __init__(
        self,
        scl_band: str = "SCL",
        clear_values: list[int] | None = None,
        min_clear_fraction: float = 0.0,
        min_valid_cover: float = 0.0,
    ) -> None:
        """Create a new Sentinel2SCLBestClear compositor.

        Args:
            scl_band: band name for Sentinel-2 scene classification layer.
            clear_values: SCL values to count as clear. Defaults to 4 vegetation,
                5 bare soil, and 6 water.
            min_clear_fraction: minimum fraction of valid pixels that must be clear
                for an item to be considered.
            min_valid_cover: minimum fraction of total pixels that must be valid
                for an item to be considered.
        """
        self.scl_band = scl_band
        self.clear_values = (
            list(self.DEFAULT_CLEAR_VALUES)
            if clear_values is None
            else list(clear_values)
        )
        if not self.clear_values:
            raise ValueError("clear_values must contain at least one SCL class")
        if SENTINEL2_SCL_NODATA_VALUE in self.clear_values:
            raise ValueError(
                "clear_values cannot include Sentinel-2 SCL nodata class 0"
            )
        if min_clear_fraction < 0 or min_clear_fraction > 1:
            raise ValueError("min_clear_fraction must be between 0 and 1")
        if min_valid_cover < 0 or min_valid_cover > 1:
            raise ValueError("min_valid_cover must be between 0 and 1")
        self.min_clear_fraction = min_clear_fraction
        self.min_valid_cover = min_valid_cover

    def _score_item(
        self,
        item: ItemType,
        tile_store: TileStoreWithLayer,
        projection: Projection,
        bounds: PixelBounds,
        resampling_method: Resampling,
    ) -> SCLClearCoverScore | None:
        """Score a single item by SCL clear cover.

        Returns:
            Clear-cover score where higher is preferred. Returns None when no data is
            available at all for the requested window.
        """
        scoring_bands = [self.scl_band]
        needed_band_sets_and_indexes = get_needed_band_sets_and_indexes(
            item, scoring_bands, tile_store
        )
        if len(needed_band_sets_and_indexes) == 0:
            raise ValueError(
                f"missing scoring bands {scoring_bands} for item {item.name}"
            )

        raster = read_raster_window_from_tiles(
            tile_store=tile_store,
            item=item,
            bands=scoring_bands,
            projection=projection,
            bounds=bounds,
            nodata_val=SENTINEL2_SCL_NODATA_VALUE,
            band_dtype=np.uint8,
            resampling=resampling_method,
        )
        if raster is None:
            return None

        scl = raster.array[0, 0, :, :]
        total_pixels = int(scl.size)
        valid_pixels = int(np.count_nonzero(scl != SENTINEL2_SCL_NODATA_VALUE))
        clear_pixels = int(np.count_nonzero(np.isin(scl, self.clear_values)))
        clear_cover = clear_pixels / total_pixels if total_pixels else 0.0
        clear_fraction = clear_pixels / valid_pixels if valid_pixels else 0.0
        valid_cover = valid_pixels / total_pixels if total_pixels else 0.0

        logger.debug(
            "Sentinel-2 SCL clear cover for %s: clear_cover=%.3f "
            "clear_fraction=%.3f valid=%.3f clear_pixels=%d total_pixels=%d",
            item.name,
            clear_cover,
            clear_fraction,
            valid_cover,
            clear_pixels,
            total_pixels,
        )
        return SCLClearCoverScore(
            clear_cover=clear_cover,
            clear_fraction=clear_fraction,
            valid_cover=valid_cover,
            clear_pixels=clear_pixels,
            valid_pixels=valid_pixels,
            total_pixels=total_pixels,
        )

    def _select_best_item(
        self,
        group: list[ItemType],
        tile_store: TileStoreWithLayer,
        projection: Projection,
        bounds: PixelBounds,
        resampling_method: Resampling,
    ) -> ItemType | None:
        """Select the item with highest SCL clear cover."""
        scored: list[tuple[SCLClearCoverScore, ItemType]] = []
        for item in group:
            score = self._score_item(
                item, tile_store, projection, bounds, resampling_method
            )
            if score is None:
                logger.debug(
                    "no data for Sentinel-2 SCL clear-cover scoring of item %s",
                    item.name,
                )
                continue
            if score.clear_fraction < self.min_clear_fraction:
                logger.debug(
                    "dropping item %s: SCL clear fraction %.3f is below %.3f",
                    item.name,
                    score.clear_fraction,
                    self.min_clear_fraction,
                )
                continue
            if score.valid_cover < self.min_valid_cover:
                logger.debug(
                    "dropping item %s: SCL valid cover %.3f is below %.3f",
                    item.name,
                    score.valid_cover,
                    self.min_valid_cover,
                )
                continue
            scored.append((score, item))

        if not scored:
            return None

        scored.sort(
            key=lambda t: (
                t[0].clear_cover,
                t[0].valid_cover,
                t[0].clear_pixels,
            ),
            reverse=True,
        )
        return scored[0][1]

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
        """Select the best-clear item, then materialize only that item."""
        if len(group) > 1:
            best_item = self._select_best_item(
                group, tile_store, projection, bounds, resampling_method
            )
            group = [best_item] if best_item is not None else []

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
