"""OmniCloudMask-based compositor for cloud-aware FIRST_VALID compositing."""

from datetime import datetime
from typing import Any

import numpy as np
import numpy.typing as npt
from omnicloudmask import predict_from_array
from rasterio.enums import Resampling

from rslearn.data_sources.data_source import ItemType
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.geometry import PixelBounds, Projection
from rslearn.utils.raster_array import RasterArray

from .compositing import Compositor, FirstValidCompositor
from .remap import Remapper
from .tile_utils import read_raster_window_from_tiles

logger = get_logger(__name__)


class OmniCloudMaskFirstValid(Compositor):
    """Sorts items by OmniCloudMask cloud score, then applies FIRST_VALID compositing.

    For each item in the group, reads R/G/NIR bands from the tile store, runs
    OmniCloudMask inference, and scores by cloud class fractions. Items are then
    sorted best-to-worst (least cloudy first) and passed to the built-in
    first-valid compositor.

    Requires the ``omnicloudmask`` package (optional dependency).
    """

    def __init__(
        self,
        red_band: str = "B04",
        green_band: str = "B03",
        nir_band: str = "B8A",
        min_inference_size: int = 32,
    ) -> None:
        """Initialize.

        Args:
            red_band: band name for red (e.g. "B04" or "red").
            green_band: band name for green (e.g. "B03" or "green").
            nir_band: band name for NIR (e.g. "B8A" or "nir08").
            min_inference_size: OmniCloudMask requires at least this many pixels
                per spatial dimension; smaller windows are padded.
        """
        self.red_band = red_band
        self.green_band = green_band
        self.nir_band = nir_band
        self.min_inference_size = min_inference_size

    def _score_item(
        self,
        item: ItemType,
        tile_store: TileStoreWithLayer,
        projection: Projection,
        bounds: PixelBounds,
        resampling_method: Resampling,
    ) -> tuple[float, float, float, float]:
        """Score a single item using OmniCloudMask.

        OmniCloudMask classifies each pixel into one of four classes:
          0 = clear, 1 = thick cloud, 2 = thin cloud, 3 = cloud shadow.

        We compute the fraction of pixels in each class and return a sort-key
        tuple designed so that Python's default ascending sort ranks the
        clearest images first:

            (thick_frac, -clear_frac, thin_frac, shadow_frac)

        Primary sort: least thick cloud. Tiebreaker: most clear pixels
        (negated so lower = more clear). Then thin cloud, then shadow.

        Returns:
            Sort key tuple ``(thick_frac, -clear_frac, thin_frac, shadow_frac)``.
        """
        scoring_bands = [self.red_band, self.green_band, self.nir_band]
        nodata_vals = [0.0, 0.0, 0.0]

        raster = read_raster_window_from_tiles(
            tile_store=tile_store,
            item=item,
            bands=scoring_bands,
            projection=projection,
            bounds=bounds,
            nodata_vals=nodata_vals,
            band_dtype=np.float32,
            resampling=resampling_method,
        )
        if raster is None:
            # No data at all -- treat as worst possible score.
            return (2.0, 1.0, 2.0, 2.0)

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
        return (thick_frac, -clear_frac, thin_frac, shadow_frac)

    def build_composite(
        self,
        group: list[ItemType],
        nodata_vals: list[Any],
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
            scored: list[tuple[tuple[float, float, float, float], ItemType]] = []
            for item in group:
                score = self._score_item(
                    item, tile_store, projection, bounds, resampling_method
                )
                logger.debug(
                    "OmniCloudMask score for %s: thick=%.3f clear=%.3f "
                    "thin=%.3f shadow=%.3f",
                    item.name,
                    score[0],
                    -score[1],
                    score[2],
                    score[3],
                )
                scored.append((score, item))

            scored.sort(key=lambda t: t[0])
            group = [item for _, item in scored]

        return FirstValidCompositor().build_composite(
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
