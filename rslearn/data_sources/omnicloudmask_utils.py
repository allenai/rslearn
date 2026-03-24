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


def _compute_clear_fraction(
    item: Item,
    geometry: STGeometry,
    get_url: Callable[[Item, str], str],
    red_asset_key: str,
    green_asset_key: str,
    nir_asset_key: str,
    resolution: float,
) -> float:
    """Compute the clear-pixel fraction for one item within the geometry.

    Reads R/G/NIR bands from the item's asset URLs, runs OmniCloudMask
    inference, and returns the fraction of pixels classified as clear (class 0).

    Args:
        item: the item to score.
        geometry: the window geometry defining the spatial extent and CRS.
        get_url: callable(item, asset_key) → accessible URL for the band.
        red_asset_key: asset key for the red band (e.g. "B04").
        green_asset_key: asset key for the green band (e.g. "B03").
        nir_asset_key: asset key for the NIR band (e.g. "B8A").
        resolution: spatial resolution to read at (in geometry CRS units, e.g. metres).

    Returns:
        fraction of pixels classified as clear, in [0, 1].
    """
    from omnicloudmask import predict_from_array

    minx, miny, maxx, maxy = geometry.shp.bounds
    width = max(1, int(abs(maxx - minx) / resolution))
    height = max(1, int(abs(maxy - miny) / resolution))
    out_transform = from_bounds(minx, miny, maxx, maxy, width, height)
    crs = geometry.projection.crs

    bands = []
    for asset_key in (red_asset_key, green_asset_key, nir_asset_key):
        url = get_url(item, asset_key)
        bands.append(_read_band(url, crs, out_transform, width, height))

    scene = np.stack(bands)  # (3, H, W)
    mask = predict_from_array(input_array=scene)
    return float((mask == 0).mean())


def sort_items_by_omnicloudmask(
    items: list[Item],
    geometry: STGeometry,
    get_url: Callable[[Item, str], str],
    red_asset_key: str,
    green_asset_key: str,
    nir_asset_key: str,
    resolution: float = 20.0,
) -> list[Item]:
    """Sort items by descending clear-pixel fraction using OmniCloudMask.

    For each item, reads R/G/NIR bands within the geometry bounds and runs
    OmniCloudMask inference to estimate the fraction of clear pixels in that
    window. Items are returned sorted descending by clear fraction so that
    clearest items come first — making them preferred by mosaicing/compositing
    logic in ``match_candidate_items_to_window``.

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
        ``items`` sorted by descending clear-pixel fraction.
    """
    scores: list[tuple[float, Item]] = []

    for item in items:
        try:
            frac = _compute_clear_fraction(
                item,
                geometry,
                get_url,
                red_asset_key,
                green_asset_key,
                nir_asset_key,
                resolution,
            )
            logger.debug(
                "OmniCloudMask clear fraction for item %s: %.3f", item.name, frac
            )
        except Exception:
            logger.warning(
                "OmniCloudMask scoring failed for item %s; placing last",
                item.name,
                exc_info=True,
            )
            frac = -1.0

        scores.append((frac, item))

    scores.sort(key=lambda t: t[0], reverse=True)
    return [item for _, item in scores]
