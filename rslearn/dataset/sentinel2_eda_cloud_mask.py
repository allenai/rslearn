"""Sentinel-2 EDA cloud-mask compositor for cloud-aware FIRST_VALID compositing."""

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


class Sentinel2EDACloudMaskFirstValid(Compositor):
    """Sort items by Sentinel-2 EDA cloud-mask classes, then apply FIRST_VALID.

    For each item in the group, this compositor reads one cloud-mask band over the
    requested window and computes a weighted score from class fractions:

    - 0: null / nodata
    - 1: clear
    - 2: cloud
    - 3: cloud shadow
    - 4: thin cloud

    Lower scores are preferred.
    """

    def __init__(
        self,
        cloud_mask_band: str = "eda_cloud_mask",
        null_weight: int = 5,
        clear_weight: int = 0,
        cloud_weight: int = 5,
        shadow_weight: int = 1,
        thin_cloud_weight: int = 1,
        unknown_weight: int = 5,
    ) -> None:
        """Create a new Sentinel2EDACloudMaskFirstValid compositor.

        Args:
            cloud_mask_band: band name for EDA cloud mask.
            null_weight: weight for class 0 pixels.
            clear_weight: weight for class 1 pixels.
            cloud_weight: weight for class 2 pixels.
            shadow_weight: weight for class 3 pixels.
            thin_cloud_weight: weight for class 4 pixels.
            unknown_weight: weight for classes outside 0..4.
        """
        self.cloud_mask_band = cloud_mask_band
        self.null_weight = null_weight
        self.clear_weight = clear_weight
        self.cloud_weight = cloud_weight
        self.shadow_weight = shadow_weight
        self.thin_cloud_weight = thin_cloud_weight
        self.unknown_weight = unknown_weight

    def _score_item(
        self,
        item: ItemType,
        tile_store: TileStoreWithLayer,
        projection: Projection,
        bounds: PixelBounds,
        resampling_method: Resampling,
    ) -> float | None:
        """Score a single item using EDA cloud-mask class fractions."""
        scoring_bands = [self.cloud_mask_band]
        # EDA cloud-mask class 0 represents null/nodata, so keep 0 as nodata.
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

        mask = raster.array[0, 0, :, :]
        null_frac = float((mask == 0).mean())
        clear_frac = float((mask == 1).mean())
        cloud_frac = float((mask == 2).mean())
        shadow_frac = float((mask == 3).mean())
        thin_cloud_frac = float((mask == 4).mean())
        unknown_frac = float((~np.isin(mask, [0, 1, 2, 3, 4])).mean())

        score = (
            null_frac * self.null_weight
            + clear_frac * self.clear_weight
            + cloud_frac * self.cloud_weight
            + shadow_frac * self.shadow_weight
            + thin_cloud_frac * self.thin_cloud_weight
            + unknown_frac * self.unknown_weight
        )
        logger.debug(
            "Sentinel-2 EDA cloud mask for %s: null=%.3f clear=%.3f cloud=%.3f "
            "shadow=%.3f thin=%.3f unknown=%.3f score=%.3f",
            item.name,
            null_frac,
            clear_frac,
            cloud_frac,
            shadow_frac,
            thin_cloud_frac,
            unknown_frac,
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
        """Score items by EDA cloudiness, sort, then delegate to FIRST_VALID."""
        if len(group) > 1:
            scored: list[tuple[float, ItemType]] = []
            for item in group:
                score = self._score_item(
                    item, tile_store, projection, bounds, resampling_method
                )
                if score is None:
                    logger.debug(
                        "no data for Sentinel-2 EDA cloud-mask scoring of item %s",
                        item.name,
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
