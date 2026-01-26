"""Functions for rendering raster label masks (e.g., segmentation masks)."""

import numpy as np
from PIL import Image
from rasterio.warp import Resampling

from rslearn.config import DType, LayerConfig
from rslearn.dataset import Window
from rslearn.log_utils import get_logger
from rslearn.train.dataset import DataInput, read_raster_layer_for_data_input
from rslearn.utils.geometry import PixelBounds, ResolutionFactor

logger = get_logger(__name__)


def render_raster_label(
    label_array: np.ndarray,
    label_colors: dict[str, tuple[int, int, int]],
    layer_config: LayerConfig,
) -> np.ndarray:
    """Render a raster label array as a colored mask numpy array.

    Args:
        label_array: Raster label array with shape (bands, height, width) - typically single band
        label_colors: Dictionary mapping label class names to RGB color tuples
        layer_config: LayerConfig object (to access class_names if available)

    Returns:
        Array with shape (height, width, 3) as uint8
    """
    if label_array.ndim == 3:
        label_values = label_array[0, :, :]
    else:
        label_values = label_array

    height, width = label_values.shape
    mask_img = np.zeros((height, width, 3), dtype=np.uint8)
    valid_mask = ~np.isnan(label_values)

    if not layer_config.class_names:
        raise ValueError(
            "class_names must be specified in config for raster label layer"
        )

    label_int = label_values.astype(np.int32)
    for idx in range(len(layer_config.class_names)):
        class_name = layer_config.class_names[idx]
        color = label_colors.get(str(class_name), (0, 0, 0))
        mask = (label_int == idx) & valid_mask
        mask_img[mask] = color

    img = Image.fromarray(mask_img, mode="RGB")
    return np.array(img)


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

    image_tensor = read_raster_layer_for_data_input(
        window, bounds, layer_name, group_idx, layer_config, data_input
    )

    return image_tensor.numpy().astype(np.float32)
