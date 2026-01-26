"""Utility functions and constants for visualization."""

from datetime import datetime
from io import BytesIO

import numpy as np
from PIL import Image

from rslearn.dataset import Window
from rslearn.log_utils import get_logger
from rslearn.utils.colors import DEFAULT_COLORS
from rslearn.utils.geometry import WGS84_PROJECTION

logger = get_logger(__name__)

# Fixed size for all visualized images (width, height in pixels)
VISUALIZATION_IMAGE_SIZE = (512, 512)


def generate_label_colors(label_classes: set[str]) -> dict[str, tuple[int, int, int]]:
    """Generate distinct colors for label classes.

    Args:
        label_classes: Set or list of label class names

    Returns:
        Dictionary mapping label class names to RGB color tuples
    """
    label_colors = {}

    sorted_labels = sorted(label_classes)
    for color_idx, label in enumerate(sorted_labels):
        label_colors[label] = DEFAULT_COLORS[color_idx % len(DEFAULT_COLORS)]

    return label_colors


def format_window_info(
    window: Window,
) -> tuple[tuple[datetime, datetime] | None, float | None, float | None]:
    """Extract window metadata for display.

    Args:
        window: Window object

    Returns:
        Tuple of (time_range, lat, lon) where time_range is a tuple of (start, end) datetime objects
    """
    lat = None
    lon = None

    geom_wgs84 = window.get_geometry().to_projection(WGS84_PROJECTION)
    centroid = geom_wgs84.shp.centroid
    lon = float(centroid.x)
    lat = float(centroid.y)

    return window.time_range, lat, lon


def array_to_bytes(
    array: np.ndarray, resampling: Image.Resampling = Image.Resampling.LANCZOS
) -> bytes:
    """Convert a numpy array to PNG bytes.

    Args:
        array: Array with shape (height, width, channels) or (height, width) as uint8
        resampling: PIL Image resampling method (default LANCZOS, use NEAREST for labels)

    Returns:
        PNG image bytes
    """
    if array.ndim == 2:
        img = Image.fromarray(array, mode="L")
    elif array.ndim == 3:
        if array.shape[-1] == 1:
            img = Image.fromarray(array[:, :, 0], mode="L")
        elif array.shape[-1] == 3:
            img = Image.fromarray(array, mode="RGB")
        else:
            img = Image.fromarray(array[:, :, :3], mode="RGB")
    else:
        raise ValueError(f"Unsupported array shape: {array.shape}")

    img = img.resize(VISUALIZATION_IMAGE_SIZE, resampling)

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )
