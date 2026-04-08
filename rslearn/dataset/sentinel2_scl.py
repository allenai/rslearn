"""Sentinel-2 SCL-based compositor for cloud-aware FIRST_VALID compositing."""

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
        nodata_val = 0

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
            nodata_val=nodata_val,
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
            "Sentinel-2 SCL for %s: shadow=%.3f medium=%.3f high=%.3f cirrus=%.3f snow_ice=%.3f score=%.3f",
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
