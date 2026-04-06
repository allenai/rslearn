"""Functions for reading and rendering raster layers for visualization."""

from collections.abc import Callable
from typing import Any

import numpy as np
from rasterio.warp import Resampling

from rslearn.config import DType, LayerConfig
from rslearn.dataset import Window
from rslearn.log_utils import get_logger
from rslearn.train.dataset import DataInput, read_raster_layer_for_data_input
from rslearn.utils.colors import DEFAULT_COLORS
from rslearn.utils.geometry import PixelBounds, ResolutionFactor

logger = get_logger(__name__)

RASTER_RENDER_SENTINEL2_RGB = "sentinel2_rgb"
RASTER_RENDER_PERCENTILE = "percentile"
RASTER_RENDER_MINMAX = "minmax"
RASTER_RENDER_LINEAR = "linear"
RASTER_RENDER_CLASSES = "classes"


# --- Reading ---


def read_raster_layer(
    window: Window,
    layer_name: str,
    layer_config: LayerConfig,
    band_names: list[str],
    group_idx: int = 0,
    bounds: PixelBounds | None = None,
) -> np.ndarray:
    """Read a raster layer for visualization.

    This reads bands from potentially multiple band sets to get the requested bands.
    Uses read_raster_layer_for_data_input from rslearn.train.dataset.

    Args:
        window: The window to read from
        layer_name: The layer name
        layer_config: The layer configuration
        band_names: List of band names to read (e.g., ["B04", "B03", "B02"])
        group_idx: The item group index (default 0)
        bounds: Optional bounds to read. If None, uses window.bounds

    Returns:
        Array with shape (bands, height, width) as float32
    """
    if bounds is None:
        bounds = window.bounds

    data_input = DataInput(
        data_type="raster",
        layers=[layer_name],
        bands=band_names,
        dtype=DType.FLOAT32,
        resolution_factor=ResolutionFactor(),  # Default 1/1, no scaling
        resampling=Resampling.nearest,
    )

    image_tensor, _ = read_raster_layer_for_data_input(
        window, bounds, layer_name, group_idx, layer_config, data_input
    )

    array = image_tensor.numpy().astype(np.float32)  # (C, T, H, W)
    array = array[:, 0, :, :]
    return array


# --- Raster render functions ---


def render_sentinel2_rgb(
    array: np.ndarray,
    layer_config: LayerConfig,
    label_colors: dict[str, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Render by dividing by 10 and clipping (for Sentinel-2 B04/B03/B02 bands).

    Args:
        array: Input array with shape (channels, height, width)
        layer_config: LayerConfig (unused)
        label_colors: Label colors (unused)

    Returns:
        Array with shape (height, width, channels) as uint8
    """
    array = np.moveaxis(array, 0, -1)
    normalized = np.zeros_like(array, dtype=np.uint8)
    for i in range(array.shape[-1]):
        band = array[..., i] / 10.0
        normalized[..., i] = np.clip(band, 0, 255).astype(np.uint8)
    return normalized


def render_percentile(
    array: np.ndarray,
    layer_config: LayerConfig,
    label_colors: dict[str, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Render using 2-98 percentile clipping per band.

    Args:
        array: Input array with shape (channels, height, width)
        layer_config: LayerConfig (unused)
        label_colors: Label colors (unused)

    Returns:
        Array with shape (height, width, channels) as uint8
    """
    array = np.moveaxis(array, 0, -1)
    normalized = np.zeros_like(array, dtype=np.uint8)
    for i in range(array.shape[-1]):
        band = array[..., i]
        valid_pixels = band[~np.isnan(band)]
        if len(valid_pixels) == 0:
            continue
        vmin, vmax = np.nanpercentile(valid_pixels, (2, 98))
        if vmax == vmin:
            continue
        band = np.clip(band, vmin, vmax)
        normalized[..., i] = ((band - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    return normalized


def render_minmax(
    array: np.ndarray,
    layer_config: LayerConfig,
    label_colors: dict[str, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Render using min-max stretch per band.

    Args:
        array: Input array with shape (channels, height, width)
        layer_config: LayerConfig (unused)
        label_colors: Label colors (unused)

    Returns:
        Array with shape (height, width, channels) as uint8
    """
    array = np.moveaxis(array, 0, -1)
    normalized = np.zeros_like(array, dtype=np.uint8)
    for i in range(array.shape[-1]):
        band = array[..., i]
        vmin, vmax = np.nanmin(band), np.nanmax(band)
        if vmax == vmin:
            continue
        band = np.clip(band, vmin, vmax)
        normalized[..., i] = ((band - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    return normalized


def render_linear(
    array: np.ndarray,
    layer_config: LayerConfig,
    label_colors: dict[str, tuple[int, int, int]] | None = None,
    vmin: float = 0,
    vmax: float = 1,
) -> np.ndarray:
    """Render using user-specified min/max range per band.

    Args:
        array: Input array with shape (channels, height, width)
        layer_config: LayerConfig (unused)
        label_colors: Label colors (unused)
        vmin: Minimum value of the range
        vmax: Maximum value of the range

    Returns:
        Array with shape (height, width, channels) as uint8
    """
    array = np.moveaxis(array, 0, -1)
    normalized = np.zeros_like(array, dtype=np.uint8)
    if vmax == vmin:
        return normalized
    for i in range(array.shape[-1]):
        band = np.clip(array[..., i], vmin, vmax)
        normalized[..., i] = ((band - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    return normalized


def render_classes(
    array: np.ndarray,
    layer_config: LayerConfig,
    label_colors: dict[str, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Render a raster as a colored class map.

    Maps integer pixel values to colors. Uses label_colors when provided,
    otherwise falls back to DEFAULT_COLORS by index.

    Args:
        array: Raster array with shape (bands, height, width) -- uses band 0
        layer_config: LayerConfig with optional class_names
        label_colors: Optional pre-computed mapping of class name -> RGB color

    Returns:
        Array with shape (height, width, 3) as uint8
    """
    if array.ndim == 3:
        label_values = array[0, :, :]
    else:
        label_values = array

    height, width = label_values.shape
    mask_img = np.zeros((height, width, 3), dtype=np.uint8)
    valid_mask = ~np.isnan(label_values)
    label_int = label_values.astype(np.int32)

    if label_colors and layer_config.class_names:
        for idx, class_name in enumerate(layer_config.class_names):
            color = label_colors.get(
                class_name, DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
            )
            mask = (label_int == idx) & valid_mask
            mask_img[mask] = color
    else:
        unique_ids = np.unique(label_int[valid_mask])
        for class_id in unique_ids:
            color = DEFAULT_COLORS[int(class_id) % len(DEFAULT_COLORS)]
            mask = (label_int == class_id) & valid_mask
            mask_img[mask] = color

    return mask_img


RASTER_RENDER_FUNCTIONS: dict[str, Callable[..., np.ndarray]] = {
    RASTER_RENDER_SENTINEL2_RGB: render_sentinel2_rgb,
    RASTER_RENDER_PERCENTILE: render_percentile,
    RASTER_RENDER_MINMAX: render_minmax,
    RASTER_RENDER_LINEAR: render_linear,
    RASTER_RENDER_CLASSES: render_classes,
}


def render_raster(
    array: np.ndarray,
    layer_config: LayerConfig,
    render_spec: dict[str, Any],
    label_colors: dict[str, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Dispatch to the appropriate raster render function.

    Args:
        array: (C, H, W) float32 input from read_raster_layer
        layer_config: LayerConfig for this layer
        render_spec: Dict with "name" key and optional "args" dict,
            e.g. {"name": "linear", "args": {"vmin": 0, "vmax": 3000}}
        label_colors: Optional pre-computed label colors to pass to render functions

    Returns:
        Rendered array as uint8 (H, W, C)
    """
    name = render_spec["name"]
    args = render_spec.get("args", {})
    fn = RASTER_RENDER_FUNCTIONS.get(name)
    if fn is None:
        raise ValueError(f"Unknown raster render method: {name}")

    return fn(array, layer_config, label_colors=label_colors, **args)
