"""Functions for rendering raster sensor images (e.g., Sentinel-2, Landsat)."""

import numpy as np

from .normalization import normalize_array


def render_sensor_image(
    array: np.ndarray,
    normalization_method: str,
) -> np.ndarray:
    """Render a raster sensor image array as a numpy array.

    Args:
        array: Array with shape (channels, height, width) from RasterFormat.decode_raster
        normalization_method: Normalization method to apply

    Returns:
        Array with shape (height, width, channels) as uint8
    """
    normalized = normalize_array(array, normalization_method)

    # If more than 3 channels, take only the first 3 for RGB
    if normalized.shape[-1] > 3:
        normalized = normalized[:, :, :3]

    return normalized
