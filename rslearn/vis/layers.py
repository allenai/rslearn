"""Functions for reading raster and vector layers from rslearn datasets."""

from typing import Any

import numpy as np
from upath import UPath

from rslearn.config import LayerConfig, LayerType
from rslearn.dataset import Window
from rslearn.log_utils import get_logger
from rslearn.utils.geometry import PixelBounds, Projection
from rslearn.utils.raster_format import RasterFormat
from rslearn.utils.vector_format import VectorFormat, GeojsonVectorFormat

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
    Based on read_raster_layer_for_data_input from rslearn.train.dataset.

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

    needed_band_indexes = {band: i for i, band in enumerate(band_names)}
    needed_sets_and_indexes = []
    
    for band_set in layer_config.band_sets:
        needed_src_indexes = []
        needed_dst_indexes = []
        if band_set.bands is None:
            continue
        for i, band in enumerate(band_set.bands):
            if band not in needed_band_indexes:
                continue
            needed_src_indexes.append(i)
            needed_dst_indexes.append(needed_band_indexes[band])
            del needed_band_indexes[band]
        if len(needed_src_indexes) == 0:
            continue
        needed_sets_and_indexes.append(
            (band_set, needed_src_indexes, needed_dst_indexes)
        )
    
    if len(needed_band_indexes) > 0:
        raise ValueError(
            f"could not get all the needed bands from "
            f"window {window.name} layer {layer_name} group {group_idx}. "
            f"Missing bands: {list(needed_band_indexes.keys())}"
        )

    # Initialize output array
    height = bounds[3] - bounds[1]
    width = bounds[2] - bounds[0]
    image = np.zeros((len(band_names), height, width), dtype=np.float32)

    # Read from each band set
    for band_set, src_indexes, dst_indexes in needed_sets_and_indexes:
        if band_set.format is None:
            raise ValueError(f"No format specified for {layer_name}")
        raster_format: RasterFormat = band_set.instantiate_raster_format()
        raster_dir = window.get_raster_dir(
            layer_name, band_set.bands, group_idx=group_idx
        )

        from rasterio.enums import Resampling

        src = raster_format.decode_raster(
            raster_dir, window.projection, bounds, resampling=Resampling.nearest
        )
        image[dst_indexes, :, :] = src[src_indexes, :, :].astype(np.float32)

    return image


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
    logger.info(f"Reading vector layer {layer_name} from {layer_dir}, bounds: {window.bounds}, projection: {window.projection}")
    
    # Use decode_vector which handles projection and bounds filtering
    features = vector_format.decode_vector(
        layer_dir, window.projection, window.bounds
    )
    logger.info(f"Decoded {len(features)} features from vector layer {layer_name}")
    return features

