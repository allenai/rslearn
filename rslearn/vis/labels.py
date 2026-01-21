"""Functions for reading and processing labels from rslearn datasets."""

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from rslearn.config import LayerConfig, LayerType
from rslearn.dataset import Window
from rslearn.log_utils import get_logger
from rslearn.utils.feature import Feature

logger = get_logger(__name__)




def detect_label_classes(windows: list[Window], layer_config: LayerConfig, layer_name: str) -> set[str]:
    """Detect all unique label classes from label layers in windows.
    
    Supports both vector (GeoJSON) and raster label layers.

    Args:
        windows: List of window objects
        layer_config: The label layer configuration
        layer_name: The name of the label layer ('label' or 'labels')

    Returns:
        Set of unique label class names (as strings)
    """
    if layer_config.type == LayerType.VECTOR:
        return _detect_label_classes_from_vector(windows, layer_config, layer_name)
    elif layer_config.type == LayerType.RASTER:
        return _detect_label_classes_from_raster(windows, layer_config, layer_name)
    else:
        logger.warning(f"Unknown layer type {layer_config.type} for label layer {layer_name}")
        return set()


def _detect_label_classes_from_vector(windows: list[Window], layer_config: LayerConfig, layer_name: str) -> set[str]:
    """Detect label classes from vector (GeoJSON) labels.
    
    If class_names is specified in the config, use it directly.
    Otherwise, sample windows to detect classes from the data.
    """
    # If class_names is specified in config, use it directly
    if layer_config.class_names:
        return set(layer_config.class_names)
    
    # Otherwise, detect from data by sampling windows
    label_classes = set()
    property_name = layer_config.class_property_name

    # Try common property names if not specified
    if property_name is None:
        property_names = ["label", "category", "class", "type"]
    else:
        property_names = [property_name]

    for window in windows:
        try:
            from rslearn.vis.layers import read_vector_layer
            features = read_vector_layer(window, layer_name, layer_config)
            for feature in features:
                if feature.properties:
                    for prop_name in property_names:
                        label = feature.properties.get(prop_name)
                        if label:
                            label_classes.add(str(label))
                            break
        except Exception as e:
            logger.debug(f"Failed to read labels from window {window.name}: {e}")
            continue

    return label_classes


def _detect_label_classes_from_raster(windows: list[Window], layer_config: LayerConfig, layer_name: str) -> set[str]:
    """Detect label classes from raster labels by sampling unique values."""
    from rslearn.vis.layers import read_raster_layer
    
    label_values = set()
    
    sample_windows = windows[:min(20, len(windows))]
    
    for window in sample_windows:
        try:
            if not layer_config.band_sets:
                continue
            
            band_set = layer_config.band_sets[0]
            if not band_set.bands:
                continue
            
            band_name = band_set.bands[0]
            label_array = read_raster_layer(
                window, layer_name, layer_config, [band_name], group_idx=0
            )
            
            unique_values = np.unique(label_array[0, :, :])
            unique_values = unique_values[~np.isnan(unique_values)]
            
            for val in unique_values:
                # Convert to integer if it's a whole number, otherwise keep as float string
                if np.isclose(val, int(val)):
                    label_values.add(int(val))
                else:
                    label_values.add(float(val))
                    
        except Exception as e:
            logger.debug(f"Failed to read raster labels from window {window.name}: {e}")
            continue
    
    # Convert to strings for consistency with vector labels
    # If class_names is specified in config, use those for mapping
    if layer_config.class_names:
        # Map numeric values to class names
        label_classes = set()
        for val in label_values:
            idx = int(val)
            if 0 <= idx < len(layer_config.class_names):
                label_classes.add(layer_config.class_names[idx])
            else:
                label_classes.add(str(val))
        return label_classes
    else:
        # Use numeric values as strings, but map 0 to "no_data" if it exists
        label_classes = set()
        for val in label_values:
            if val == 0:
                label_classes.add("no_data")
            else:
                label_classes.add(str(val))
        return label_classes


def generate_label_colors(label_classes: set[str]) -> dict[str, tuple[int, int, int]]:
    """Generate distinct colors for label classes.

    Args:
        label_classes: Set or list of label class names

    Returns:
        Dictionary mapping label class names to RGB color tuples
    """
    label_classes = sorted(label_classes)
    label_colors = {}

    # Hardcoded color for "no_data" (always black - rslearn convention)
    NO_DATA_COLOR = (0, 0, 0)

    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
        (255, 192, 203),  # Pink
        (128, 128, 0),  # Olive
        (0, 128, 128),  # Teal
        (128, 0, 0),  # Maroon
        (0, 128, 0),  # Dark Green
        (0, 0, 128),  # Navy
        (128, 128, 128),  # Gray
        (255, 165, 0),  # Orange
        (255, 20, 147),  # Deep Pink
        (50, 205, 50),  # Lime Green
        (255, 140, 0),  # Dark Orange
        (70, 130, 180),  # Steel Blue
    ]

    # Assign colors to classes
    color_idx = 0
    for label in label_classes:
        if label == "no_data":
            label_colors[label] = NO_DATA_COLOR
        else:
            label_colors[label] = colors[color_idx % len(colors)]
            color_idx += 1

    return label_colors


def get_classification_label(
    window: Window,
    layer_config: LayerConfig,
    layer_name: str,
    group_idx: int = 0,
) -> str | None:
    """Get the classification label for a window.

    Supports both vector (GeoJSON) and raster label layers.
    For vector: extracts label from first feature's properties.
    For raster: finds the single non-zero pixel value.

    Args:
        window: The window to read from
        layer_config: The label layer configuration
        layer_name: The name of the label layer ('label' or 'labels')
        group_idx: The item group index (default 0)

    Returns:
        The label string, or None if not found
    """
    if layer_config.type == LayerType.VECTOR:
        from rslearn.vis.layers import read_vector_layer
        features = read_vector_layer(window, layer_name, layer_config, group_idx=group_idx)
        if not features:
            logger.warning(f"No features in classification label for {window.name}")
            return None

        first_feature = features[0]
        if not first_feature.properties:
            return None

        property_name = layer_config.class_property_name
        if property_name is None:
            property_names = ["label", "category", "class", "type"]
        else:
            property_names = [property_name]

        for prop_name in property_names:
            label = first_feature.properties.get(prop_name)
            if label:
                logger.info(f"Classification label for {window.name}: {label}")
                return str(label)

        logger.warning(f"No label property found in classification GeoJSON for {window.name}")
        return None
    
    elif layer_config.type == LayerType.RASTER:
        from rslearn.vis.layers import read_raster_layer
        
        if not layer_config.band_sets:
            logger.warning(f"No band sets in raster label layer {layer_name} for {window.name}")
            return None
        
        band_set = layer_config.band_sets[0]
        if not band_set.bands:
            logger.warning(f"No bands in raster label layer {layer_name} for {window.name}")
            return None
        
        band_name = band_set.bands[0]
        label_array = read_raster_layer(
            window, layer_name, layer_config, [band_name], group_idx=group_idx
        )
        
        if label_array.ndim == 3:
            label_2d = label_array[0, :, :]
        else:
            label_2d = label_array
        
        unique_vals = np.unique(label_2d)
        non_zero_vals = unique_vals[(unique_vals != 0) & ~np.isnan(unique_vals)]
        
        if len(non_zero_vals) == 0:
            logger.warning(f"No non-zero label value found in raster for {window.name}")
            return None
        
        if len(non_zero_vals) > 1:
            logger.warning(
                f"Multiple non-zero values found in raster label for {window.name}: {non_zero_vals}. "
                f"Using first value: {non_zero_vals[0]}"
            )
        
        label_value = non_zero_vals[0]
        
        if layer_config.class_names:
            idx = int(label_value)
            if 0 <= idx < len(layer_config.class_names):
                class_name = layer_config.class_names[idx]
                logger.info(f"Classification label for {window.name}: {class_name}")
                return str(class_name)
            else:
                logger.warning(
                    f"Label value {label_value} out of range for class_names (length {len(layer_config.class_names)}) "
                    f"for {window.name}"
                )
                return str(int(label_value))
        else:
            if np.isclose(label_value, int(label_value)):
                label_str = str(int(label_value))
            else:
                label_str = str(float(label_value))
            logger.info(f"Classification label for {window.name}: {label_str}")
            return label_str
    
    else:
        logger.warning(f"Unsupported layer type {layer_config.type} for classification label")
        return None

