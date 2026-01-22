"""Functions for reading raster and vector layers from rslearn datasets."""

from typing import Any

import numpy as np
from rasterio.warp import Resampling

from rslearn.config import DType, LayerConfig, LayerType
from rslearn.dataset import Window
from rslearn.log_utils import get_logger
from rslearn.train.dataset import DataInput, read_raster_layer_for_data_input
from rslearn.utils.geometry import PixelBounds, ResolutionFactor
from rslearn.utils.vector_format import VectorFormat

logger = get_logger(__name__)


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
    if layer_config.type != LayerType.RASTER:
        raise ValueError(f"Layer {layer_name} is not a raster layer")

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
