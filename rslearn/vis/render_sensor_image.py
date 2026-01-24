"""Functions for rendering raster sensor images (e.g., Sentinel-2, Landsat)."""

from io import BytesIO

import numpy as np
from PIL import Image

from .normalization import normalize_array
from .utils import VISUALIZATION_IMAGE_SIZE


def render_sensor_image_as_bytes(
    array: np.ndarray,
    normalization_method: str,
) -> bytes:
    """Render a raster sensor image array as PNG bytes.

    Args:
        array: Array with shape (channels, height, width) from RasterFormat.decode_raster
        normalization_method: Normalization method to apply

    Returns:
        PNG image bytes
    """
    normalized = normalize_array(array, normalization_method)

    if normalized.shape[-1] == 1:
        img = Image.fromarray(normalized[:, :, 0], mode="L")
    elif normalized.shape[-1] == 3:
        img = Image.fromarray(normalized, mode="RGB")
    else:
        img = Image.fromarray(normalized[:, :, :3], mode="RGB")

    img = img.resize(VISUALIZATION_IMAGE_SIZE, Image.Resampling.LANCZOS)

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
