"""Functions for rendering vector layers as text (text, property)."""

import json
from typing import Any

import shapely.geometry

from rslearn.config import LayerConfig
from rslearn.dataset import Window
from rslearn.log_utils import get_logger
from rslearn.utils.feature import Feature

from .utils import read_vector_layer

logger = get_logger(__name__)

VECTOR_TEXT_RENDER_TEXT = "text"
VECTOR_TEXT_RENDER_PROPERTY = "property"


def render_text(
    features: list[Feature],
    window: Window,
    layer_config: LayerConfig,
) -> str | None:
    """Render all vector features as a GeoJSON FeatureCollection string.

    Args:
        features: List of Feature objects
        window: Window object (unused)
        layer_config: LayerConfig object (unused)

    Returns:
        GeoJSON string, or None if no features
    """
    if not features:
        return None

    geojson_features = []
    for f in features:
        geojson_features.append(
            {
                "type": "Feature",
                "geometry": shapely.geometry.mapping(f.geometry.shp),
                "properties": dict(f.properties) if f.properties else {},
            }
        )
    return json.dumps(
        {"type": "FeatureCollection", "features": geojson_features}, indent=2
    )


def render_property(
    features: list[Feature],
    window: Window,
    layer_config: LayerConfig,
) -> str | None:
    """Get the class_property_name value from the first feature that has it.

    Extracts the label value from features using the property name specified in
    layer_config.class_property_name.

    Args:
        features: List of Feature objects
        window: Window object (unused)
        layer_config: The layer configuration (must have class_property_name set)

    Returns:
        The label string, or None if not found
    """
    if not features:
        logger.warning(f"No features in vector layer for {window.name}")
        return None

    if not layer_config.class_property_name:
        raise ValueError(
            "class_property_name must be specified in the config for "
            "vector layer when using 'property' render method."
        )

    for feature in features:
        if not feature.properties:
            continue
        val = feature.properties.get(layer_config.class_property_name)
        if val is not None:
            logger.info(f"Label for {window.name}: {val}")
            return str(val)

    return None


VECTOR_TEXT_RENDER_FUNCTIONS: dict[str, Any] = {
    VECTOR_TEXT_RENDER_TEXT: render_text,
    VECTOR_TEXT_RENDER_PROPERTY: render_property,
}


def render_vector_text(
    window: Window,
    layer_name: str,
    layer_config: LayerConfig,
    render_spec: dict[str, Any],
    group_idx: int = 0,
) -> str | None:
    """Dispatch to the appropriate vector text render function.

    Reads the vector features and passes them to the render function selected
    by render_spec["name"].

    Args:
        window: Window object
        layer_name: Layer name
        layer_config: LayerConfig object
        render_spec: Dict with "name" key and optional "args" dict
        group_idx: Item group index

    Returns:
        Text string, or None if no features available
    """
    name = render_spec["name"]
    # args contains extra arguments that are expected by the render function.
    args = render_spec.get("args", {})
    fn = VECTOR_TEXT_RENDER_FUNCTIONS.get(name)
    if fn is None:
        raise ValueError(f"Unknown vector text render method: {name}")

    features = read_vector_layer(window, layer_name, layer_config, group_idx=group_idx)

    return fn(features, window, layer_config, **args)
