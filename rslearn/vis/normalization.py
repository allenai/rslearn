"""Normalization functions for raster data visualization."""

import numpy as np

from rslearn.log_utils import get_logger

logger = get_logger(__name__)


def normalize_band(band: np.ndarray, method: str = "sentinel2_rgb") -> np.ndarray:
    """Normalize band to 0-255 range.

    Args:
        band: Input band data
        method: Normalization method
            - 'sentinel2_rgb': Divide by 10 and clip (for B04/B03/B02)
            - 'percentile': Use 2-98 percentile clipping
            - 'minmax': Use min-max stretch

    Returns:
        Normalized band as uint8 array
    """
    if method == "sentinel2_rgb":
        band = band / 10.0
        band = np.clip(band, 0, 255).astype(np.uint8)
    elif method == "percentile":
        valid_pixels = band[~np.isnan(band)]
        if len(valid_pixels) == 0:
            return np.zeros_like(band, dtype=np.uint8)
        vmin, vmax = np.nanpercentile(valid_pixels, (2, 98))
        if vmax == vmin:
            return np.zeros_like(band, dtype=np.uint8)
        band = np.clip(band, vmin, vmax)
        band = ((band - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    elif method == "minmax":
        vmin, vmax = np.nanmin(band), np.nanmax(band)
        if vmax == vmin:
            return np.zeros_like(band, dtype=np.uint8)
        band = np.clip(band, vmin, vmax)
        band = ((band - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return band


def normalize_array(array: np.ndarray, method: str = "sentinel2_rgb") -> np.ndarray:
    """Normalize a multi-band array to 0-255 range.

    Args:
        array: Input array with shape (bands, height, width) or (height, width, bands)
        method: Normalization method (applied per-band)

    Returns:
        Normalized array as uint8
    """
    # Handle (bands, height, width) format
    if array.ndim == 3 and array.shape[0] < array.shape[2]:
        array = np.moveaxis(array, 0, -1)  # Move to (height, width, bands)
    
    normalized = np.zeros_like(array, dtype=np.uint8)
    for i in range(array.shape[-1]):
        normalized[..., i] = normalize_band(array[..., i], method)
    
    return normalized

