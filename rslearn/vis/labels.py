"""Functions for reading and processing labels from rslearn datasets."""

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from rslearn.config import LayerConfig, LayerType
from rslearn.dataset import Window
from rslearn.log_utils import get_logger
from rslearn.utils.feature import Feature
from rslearn.utils.colors import DEFAULT_COLORS

logger = get_logger(__name__)



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

    color_idx = 0
    for label in label_classes:
        if label == "no_data":
            label_colors[label] = NO_DATA_COLOR
        else:
            label_colors[label] = DEFAULT_COLORS[color_idx % len(DEFAULT_COLORS)]
            color_idx += 1

    return label_colors


def get_vector_label_by_property(
    window: Window,
    layer_config: LayerConfig,
    layer_name: str,
    group_idx: int = 0,
) -> str | None:
    """Get a label value from a vector layer's first feature property.

    Extracts the label value from the first feature's properties using the property
    name specified in layer_config.class_property_name. This works for both
    classification and segmentation tasks that use vector labels.

    Args:
        window: The window to read from
        layer_config: The label layer configuration (must be vector type)
        layer_name: The name of the label layer
        group_idx: The item group index (default 0)

    Returns:
        The label string, or None if not found
    """
    if layer_config.type != LayerType.VECTOR:
        raise ValueError(
            f"Vector labels must use vector layers. "
            f"Layer '{layer_name}' is of type {layer_config.type}."
        )
    
    from rslearn.vis.layers import read_vector_layer
    features = read_vector_layer(window, layer_name, layer_config, group_idx=group_idx)
    if not features:
        logger.warning(f"No features in vector label layer {layer_name} for {window.name}")
        return None

    first_feature = features[0]
    if not first_feature.properties:
        return None

    if not layer_config.class_property_name:
        raise ValueError(
            f"class_property_name must be specified in the config for vector label layer '{layer_name}'. "
            "Auto-detection of property name is not supported."
        )

    label = first_feature.properties.get(layer_config.class_property_name)
    if label:
        logger.info(f"Label for {window.name}: {label}")
        return str(label)

    logger.warning(
        f"Property '{layer_config.class_property_name}' not found in vector label layer {layer_name} for {window.name}"
    )
    return None

