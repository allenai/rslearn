"""Utility functions and constants for visualization."""

from datetime import datetime
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from rslearn.config import LayerConfig, LayerType
from rslearn.dataset import Window
from rslearn.log_utils import get_logger
from rslearn.utils.colors import DEFAULT_COLORS
from rslearn.utils.geometry import WGS84_PROJECTION
from rslearn.utils.vector_format import VectorFormat

logger = get_logger(__name__)

# Fixed size for all visualized images (width, height in pixels)
VISUALIZATION_IMAGE_SIZE = (512, 512)


def read_vector_layer(
    window: Window,
    layer_name: str,
    layer_config: LayerConfig,
    group_idx: int = 0,
) -> list[Any]:
    """Read a vector layer for visualization.

    Args:
        window: The window to read from
        layer_name: The layer name
        layer_config: The layer configuration
        group_idx: The item group index (default 0)

    Returns:
        List of Feature objects
    """
    if layer_config.type != LayerType.VECTOR:
        raise ValueError(f"Layer {layer_name} is not a vector layer")

    vector_format: VectorFormat = layer_config.instantiate_vector_format()
    layer_dir = window.get_layer_dir(layer_name, group_idx=group_idx)
    logger.info(
        f"Reading vector layer {layer_name} from {layer_dir}, bounds: {window.bounds}, projection: {window.projection}"
    )

    features = vector_format.decode_vector(layer_dir, window.projection, window.bounds)
    logger.info(f"Decoded {len(features)} features from vector layer {layer_name}")
    return features


def generate_label_colors_for_layer(
    layer_config: LayerConfig,
) -> dict[str, tuple[int, int, int]] | None:
    """Generate label colors from a layer config's class_names.

    Returns None if class_names is not set, signaling that the caller should fall back
    to DEFAULT_COLORS by index.

    Args:
        layer_config: LayerConfig to read class_names from

    Returns:
        Dictionary mapping label class names to RGB color tuples, or None
    """
    if not layer_config.class_names:
        return None

    label_colors = {}
    for color_idx, label in enumerate(layer_config.class_names):
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
    array: np.ndarray, resampling: Image.Resampling = Image.Resampling.NEAREST
) -> bytes:
    """Convert a numpy array to PNG bytes.

    Args:
        array: Array with shape (height, width, channels) or (height, width) as uint8
        resampling: PIL Image resampling method (default NEAREST)

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
