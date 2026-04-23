"""OmniCloudMask-based compositor for cloud-aware FIRST_VALID compositing."""

import math
from datetime import datetime

import numpy as np
import numpy.typing as npt
from omnicloudmask import predict_from_array
from rasterio.enums import Resampling

from rslearn.data_sources.data_source import ItemType
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.geometry import PixelBounds, Projection
from rslearn.utils.raster_array import RasterArray

from .compositing import BandSetCompositeRequest, Compositor, FirstValidCompositor
from .remap import Remapper
from .tile_utils import get_needed_band_sets_and_indexes, read_raster_window_from_tiles

logger = get_logger(__name__)


class OmniCloudMaskFirstValid(Compositor):
    """Sorts items by OmniCloudMask cloud score, then applies FIRST_VALID compositing.

    For each item in the group, reads R/G/NIR bands from the tile store, runs
    OmniCloudMask inference, and scores by cloud class fractions. Items are then
    sorted best-to-worst and passed to the built-in first-valid compositor.

    Requires the ``omnicloudmask`` package (optional dependency).

    Note that the R/G/NIR bands will be read twice during materialization: once to
    perform cloudiness scoring, and once during compositing to form the materialized
    image. When ``scoring_resolution`` is set, scoring is done once on a grid at
    that resolution for the whole window and the resulting item order is reused
    across band-set materialization passes.

    For best ranking quality, use windows with at least 96x96 pixels. Smaller
    windows can run with padding but may have reduced cloud-mask accuracy.
    """

    def __init__(
        self,
        red_band: str = "B04",
        green_band: str = "B03",
        nir_band: str = "B8A",
        min_inference_size: int = 32,
        scoring_resolution: float | None = None,
        clear_weight: int = 0,
        thick_cloud_weight: int = 5,
        thin_cloud_weight: int = 1,
        cloud_shadow_weight: int = 1,
    ) -> None:
        """Create a new OmniCloudMaskFirstValid.

        Args:
            red_band: band name for red (e.g. "B04" or "red").
            green_band: band name for green (e.g. "B03" or "green").
            nir_band: band name for NIR (e.g. "B8A" or "nir08").
            min_inference_size: OmniCloudMask requires at least this many pixels
                per spatial dimension; smaller windows are padded. Padding
                does not add spatial context, so it is still recommended to use
                windows of at least 96x96 pixels.
            scoring_resolution: optional pixel size for the scoring grid. When
                set, scoring is done once on a window-level grid at this
                resolution and the resulting ranking is reused across band sets.
                When unset, scoring is performed on each materialization grid.
            clear_weight: weight for clear pixels when computing the score.
            thick_cloud_weight: weight for thick cloud pixels when computing the score.
            thin_cloud_weight: weight for thin cloud pixels when computing the score.
            cloud_shadow_weight: weight for cloud shadow pixels when computing the score.
        """
        self.red_band = red_band
        self.green_band = green_band
        self.nir_band = nir_band
        self.min_inference_size = min_inference_size
        self.scoring_resolution = scoring_resolution
        self.clear_weight = clear_weight
        self.thick_cloud_weight = thick_cloud_weight
        self.thin_cloud_weight = thin_cloud_weight
        self.cloud_shadow_weight = cloud_shadow_weight

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

    def _get_window_grid(
        self,
        projection: Projection,
        bounds: PixelBounds,
        window_projection: Projection | None = None,
        window_bounds: PixelBounds | None = None,
        target_resolution: float | None = None,
    ) -> tuple[Projection, PixelBounds]:
        """Get a window-level scoring grid, optionally rescaled to a target resolution."""
        base_projection = window_projection or projection
        base_bounds = window_bounds or bounds
        if target_resolution is None:
            return base_projection, base_bounds
        return self._rescale_grid(base_projection, base_bounds, target_resolution)

    def _get_scoring_grid(
        self,
        projection: Projection,
        bounds: PixelBounds,
        window_projection: Projection | None = None,
        window_bounds: PixelBounds | None = None,
    ) -> tuple[Projection, PixelBounds]:
        """Get the grid on which cloud ranking should be evaluated."""
        if self.scoring_resolution is not None:
            return self._get_window_grid(
                projection,
                bounds,
                window_projection,
                window_bounds,
                self.scoring_resolution,
            )

        return projection, bounds

    def _score_item(
        self,
        item: ItemType,
        tile_store: TileStoreWithLayer,
        projection: Projection,
        bounds: PixelBounds,
        resampling_method: Resampling,
    ) -> float | None:
        """Score a single item using OmniCloudMask.

        OmniCloudMask classifies each pixel into one of four classes:
          0 = clear, 1 = thick cloud, 2 = thin cloud, 3 = cloud shadow.

        We score by multiplying the number of pixels in each class by the corresponding
        configurable weight. The default is 5*thick_cloud + thin_cloud + cloud_shadow.

        Returns: the score, where lower scores should be preferred.
        """
        scoring_bands = [self.red_band, self.green_band, self.nir_band]
        # The NODATA values for the scoring bands are expected to be 0 for both Sentinel-2
        # and Landsat.
        nodata_val = 0

        # The bands should be available, raise error if not.
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
            band_dtype=np.float32,
            resampling=resampling_method,
        )
        if raster is None:
            # No data at all -- return None to discard this candidate.
            # This could happen if the geometry metadata during prepare contained the
            # window, but the actual raster did not end up containing the window.
            return None

        arr = raster.array[:, 0, :, :]  # (3, H, W)

        _, h, w = arr.shape
        pad_h = max(0, self.min_inference_size - h)
        pad_w = max(0, self.min_inference_size - w)
        if pad_h > 0 or pad_w > 0:
            arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")

        mask = predict_from_array(input_array=arr)

        # Evaluate only the original (unpadded) region.
        mask = mask[:h, :w]
        clear_frac = float((mask == 0).mean())
        thick_frac = float((mask == 1).mean())
        thin_frac = float((mask == 2).mean())
        shadow_frac = float((mask == 3).mean())

        score = (
            clear_frac * self.clear_weight
            + thick_frac * self.thick_cloud_weight
            + thin_frac * self.thin_cloud_weight
            + shadow_frac * self.cloud_shadow_weight
        )
        logger.debug(
            "OmniCloudMask for %s: clear=%.3f thick=%.3f thin=%.3f shadow=%.3f score=%.3f",
            item.name,
            clear_frac,
            thick_frac,
            thin_frac,
            shadow_frac,
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
                # Missing image. We skip this item since we can't score it and
                # anyway it likely also doesn't have the bands needed for compositing.
                logger.debug("no data for OmniCloudMask scoring of item %s", item.name)
                continue
            scored.append((score, item))

        scored.sort(key=lambda t: t[0])
        return [item for _, item in scored]

    def build_composites(
        self,
        group: list[ItemType],
        requests: list[BandSetCompositeRequest],
        tile_store: TileStoreWithLayer,
        window_projection: Projection | None = None,
        window_bounds: PixelBounds | None = None,
        request_time_range: tuple[datetime, datetime] | None = None,
    ) -> list[RasterArray]:
        """Build composites for all band sets, sharing ranking work when possible."""
        sorted_groups: dict[
            tuple[Projection, PixelBounds, Resampling],
            list[ItemType],
        ] = {}
        rasters: list[RasterArray] = []

        for request in requests:
            cur_group = group
            if len(group) > 1:
                scoring_projection, scoring_bounds = self._get_scoring_grid(
                    request.projection,
                    request.bounds,
                    window_projection=window_projection,
                    window_bounds=window_bounds,
                )
                cache_key = (
                    scoring_projection,
                    scoring_bounds,
                    request.resampling_method,
                )
                cur_group = sorted_groups.get(cache_key)
                if cur_group is None:
                    cur_group = self._sort_group(
                        group,
                        tile_store,
                        scoring_projection,
                        scoring_bounds,
                        request.resampling_method,
                    )
                    sorted_groups[cache_key] = cur_group

            rasters.append(
                FirstValidCompositor().build_composite(
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
            )

        return rasters

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
        """Score items by cloud cover, sort, then delegate to FIRST_VALID."""
        if len(group) > 1:
            scoring_projection, scoring_bounds = self._get_scoring_grid(
                projection,
                bounds,
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
