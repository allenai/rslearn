"""Utilities for pixel-level cloud scoring using OmniCloudMask."""

from collections.abc import Callable

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT

from rslearn.data_sources.data_source import Item
from rslearn.log_utils import get_logger
from rslearn.utils.geometry import STGeometry

logger = get_logger(__name__)


def _read_band(
    url: str,
    crs: object,
    out_transform: object,
    width: int,
    height: int,
) -> np.ndarray:
    """Read one raster band from a URL into the given output grid."""
    with rasterio.open(url) as src:
        with WarpedVRT(
            src,
            crs=crs,
            transform=out_transform,
            width=width,
            height=height,
            resampling=Resampling.bilinear,
        ) as vrt:
            return vrt.read(1).astype(np.float32)


def _compute_cloud_class_fractions(
    item: Item,
    geometry: STGeometry,
    get_url: Callable[[Item, str], str],
    red_asset_key: str,
    green_asset_key: str,
    nir_asset_key: str,
    resolution: float,
) -> tuple[float, float, float, float]:
    """Compute OmniCloudMask class fractions for one item within the geometry.

    Reads R/G/NIR bands from the item's asset URLs, runs OmniCloudMask
    inference, and returns per-class fractions in this order:
    clear (0), thick cloud (1), thin cloud (2), cloud shadow (3).

    Args:
        item: the item to score.
        geometry: the window geometry defining the spatial extent and CRS.
            ``geometry.shp`` is in pixel coordinates; this function converts
            to CRS units before calling rasterio.
        get_url: callable(item, asset_key) → accessible URL for the band.
        red_asset_key: asset key for the red band (e.g. "B04").
        green_asset_key: asset key for the green band (e.g. "B03").
        nir_asset_key: asset key for the NIR band (e.g. "B8A").
        resolution: spatial resolution to read at (in CRS units, e.g. metres
            for a UTM projection).

    Returns:
        tuple of class fractions ``(clear, thick, thin, shadow)``, each in [0, 1].
    """
    from omnicloudmask import predict_from_array

    # geometry.shp is in pixel coordinates; convert to CRS units by multiplying
    # by the projection resolutions (pixel * resolution = CRS unit).
    px_minx, px_miny, px_maxx, px_maxy = geometry.shp.bounds
    x_res = geometry.projection.x_resolution
    y_res = geometry.projection.y_resolution

    crs_coords_x = sorted([px_minx * x_res, px_maxx * x_res])
    crs_coords_y = sorted([px_miny * y_res, px_maxy * y_res])
    crs_left, crs_right = crs_coords_x
    crs_bottom, crs_top = crs_coords_y

    width = max(1, int(abs(crs_right - crs_left) / resolution))
    height = max(1, int(abs(crs_top - crs_bottom) / resolution))
    out_transform = from_bounds(crs_left, crs_bottom, crs_right, crs_top, width, height)
    crs = geometry.projection.crs

    bands = []
    for asset_key in (red_asset_key, green_asset_key, nir_asset_key):
        url = get_url(item, asset_key)
        bands.append(_read_band(url, crs, out_transform, width, height))

    scene = np.stack(bands)  # (3, H, W)

    # OmniCloudMask requires at least 32×32 pixels; pad if necessary.
    min_size = 32
    _, h, w = scene.shape
    pad_h = max(0, min_size - h)
    pad_w = max(0, min_size - w)
    if pad_h > 0 or pad_w > 0:
        scene = np.pad(scene, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")

    mask = predict_from_array(input_array=scene)

    # Only evaluate over the original (unpadded) region.
    mask = mask[:h, :w]
    clear_frac = float((mask == 0).mean())
    thick_frac = float((mask == 1).mean())
    thin_frac = float((mask == 2).mean())
    shadow_frac = float((mask == 3).mean())
    return clear_frac, thick_frac, thin_frac, shadow_frac


def sort_items_by_omnicloudmask(
    items: list[Item],
    geometry: STGeometry,
    get_url: Callable[[Item, str], str],
    red_asset_key: str,
    green_asset_key: str,
    nir_asset_key: str,
    resolution: float = 20.0,
) -> list[Item]:
    """Sort items by OmniCloudMask classes with thick-cloud-first prioritization.

    For each item, reads R/G/NIR bands within the geometry bounds and runs
    OmniCloudMask inference in that window. Ranking prioritizes *minimizing thick
    cloud fraction* (class 1) first, because thick cloud is the most severe cloud
    failure mode for downstream quality.

    Tie-breakers are, in order:
    1. higher clear fraction (class 0),
    2. lower thin cloud fraction (class 2),
    3. lower cloud shadow fraction (class 3).

    Items that fail to be scored (e.g. missing asset URLs or read errors) are
    placed at the end of the list.

    Args:
        items: candidate items to score and sort.
        geometry: window geometry (defines spatial extent and CRS for reads).
        get_url: callable(item, asset_key) → URL string for reading that band.
            Use this to inject URL signing (e.g. for Planetary Computer).
        red_asset_key: asset key for the red (B04) band.
        green_asset_key: asset key for the green (B03) band.
        nir_asset_key: asset key for the NIR (B8A) band.
        resolution: resolution to read bands at (geometry CRS units, default 20 m).

    Returns:
        ``items`` sorted best-to-worst by OmniCloudMask class fractions.
    """
    scores: list[tuple[tuple[float, float, float, float], Item]] = []

    for item in items:
        try:
            clear_frac, thick_frac, thin_frac, shadow_frac = (
                _compute_cloud_class_fractions(
                    item,
                    geometry,
                    get_url,
                    red_asset_key,
                    green_asset_key,
                    nir_asset_key,
                    resolution,
                )
            )
            logger.debug(
                "OmniCloudMask fractions for %s: thick=%.3f clear=%.3f thin=%.3f shadow=%.3f",
                item.name,
                thick_frac,
                clear_frac,
                thin_frac,
                shadow_frac,
            )
        except Exception:
            logger.warning(
                "OmniCloudMask scoring failed for item %s; placing last",
                item.name,
                exc_info=True,
            )
            # Sort key is (thick asc, -clear asc, thin asc, shadow asc).
            # Use out-of-range sentinel values so failures always sort last.
            sort_key = (2.0, 1.0, 2.0, 2.0)
        else:
            sort_key = (thick_frac, -clear_frac, thin_frac, shadow_frac)

        scores.append((sort_key, item))

    scores.sort(key=lambda t: t[0])
    return [item for _, item in scores]
