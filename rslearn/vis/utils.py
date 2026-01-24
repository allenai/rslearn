"""Utility functions and constants for visualization."""

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


def format_window_info(window: Window) -> tuple[str, float | None, float | None]:
    """Format window metadata for display.

    Args:
        window: Window object

    Returns:
        Tuple of (formatted info HTML, lat, lon) for Google Maps link
    """
    parts = []
    lat = None
    lon = None

    if window.time_range:
        start = window.time_range[0].isoformat()[:10]
        end = window.time_range[1].isoformat()[:10]
        parts.append(f"Time: {start} to {end}")

    geom_wgs84 = window.get_geometry().to_projection(WGS84_PROJECTION)
    centroid = geom_wgs84.shp.centroid
    lon = float(centroid.x)
    lat = float(centroid.y)
    parts.insert(0, f"Lat: {lat:.4f}, Lon: {lon:.4f}")

    return "<br>".join(parts) if parts else "Unknown", lat, lon


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )
