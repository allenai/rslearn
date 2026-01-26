"""Normalization functions for raster data visualization."""

from collections.abc import Callable
from enum import StrEnum

import numpy as np

from rslearn.log_utils import get_logger

logger = get_logger(__name__)


class NormalizationMethod(StrEnum):
    """Normalization methods for raster data visualization."""

    SENTINEL2_RGB = "sentinel2_rgb"
    """Divide by 10 and clip (for Sentinel-2 B04/B03/B02 bands)."""

    PERCENTILE = "percentile"
    """Use 2-98 percentile clipping."""

    MINMAX = "minmax"
    """Use min-max stretch."""


def _normalize_sentinel2_rgb(band: np.ndarray) -> np.ndarray:
    """Normalize band using Sentinel-2 RGB method (divide by 10 and clip).

    Args:
        band: Input band data

    Returns:
        Normalized band as uint8 array
    """
    band = band / 10.0
    band = np.clip(band, 0, 255).astype(np.uint8)
    return band


def _normalize_percentile(band: np.ndarray) -> np.ndarray:
    """Normalize band using 2-98 percentile clipping.

    Args:
        band: Input band data

    Returns:
        Normalized band as uint8 array
    """
    valid_pixels = band[~np.isnan(band)]
    if len(valid_pixels) == 0:
        return np.zeros_like(band, dtype=np.uint8)
    vmin, vmax = np.nanpercentile(valid_pixels, (2, 98))
    if vmax == vmin:
        return np.zeros_like(band, dtype=np.uint8)
    band = np.clip(band, vmin, vmax)
    band = ((band - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    return band


def _normalize_minmax(band: np.ndarray) -> np.ndarray:
    """Normalize band using min-max stretch.

    Args:
        band: Input band data

    Returns:
        Normalized band as uint8 array
    """
    vmin, vmax = np.nanmin(band), np.nanmax(band)
    if vmax == vmin:
        return np.zeros_like(band, dtype=np.uint8)
    band = np.clip(band, vmin, vmax)
    band = ((band - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    return band


_NORMALIZATION_FUNCTIONS: dict[
    NormalizationMethod, Callable[[np.ndarray], np.ndarray]
] = {
    NormalizationMethod.SENTINEL2_RGB: _normalize_sentinel2_rgb,
    NormalizationMethod.PERCENTILE: _normalize_percentile,
    NormalizationMethod.MINMAX: _normalize_minmax,
}


def normalize_band(
    band: np.ndarray, method: str | NormalizationMethod = "sentinel2_rgb"
) -> np.ndarray:
    """Normalize band to 0-255 range.

    Args:
        band: Input band data
        method: Normalization method (string or NormalizationMethod enum)
            - 'sentinel2_rgb': Divide by 10 and clip (for B04/B03/B02)
            - 'percentile': Use 2-98 percentile clipping
            - 'minmax': Use min-max stretch

    Returns:
        Normalized band as uint8 array
    """
    method_enum = NormalizationMethod(method) if isinstance(method, str) else method
    normalize_func = _NORMALIZATION_FUNCTIONS.get(method_enum)
    if normalize_func is None:
        raise ValueError(f"Unknown normalization method: {method_enum}")
    return normalize_func(band)


def normalize_array(
    array: np.ndarray, method: str | NormalizationMethod = "sentinel2_rgb"
) -> np.ndarray:
    """Normalize a multi-band array to 0-255 range.

    Args:
        array: Input array with shape (channels, height, width) from RasterFormat.decode_raster
        method: Normalization method (applied per-band, string or NormalizationMethod enum)

    Returns:
        Normalized array as uint8 with shape (height, width, channels)
    """
    if array.ndim == 3:
        array = np.moveaxis(array, 0, -1)

    normalized = np.zeros_like(array, dtype=np.uint8)
    for i in range(array.shape[-1]):
        normalized[..., i] = normalize_band(array[..., i], method)

    return normalized
