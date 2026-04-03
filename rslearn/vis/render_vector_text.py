"""Functions for rendering vector layers as text (text, property)."""

import json
from typing import Any

import shapely.geometry

from rslearn.config import LayerConfig, LayerType
from rslearn.dataset import Window
from rslearn.log_utils import get_logger
from rslearn.utils.feature import Feature
from rslearn.utils.vector_format import VectorFormat

logger = get_logger(__name__)

VECTOR_TEXT_RENDER_TEXT = "text"
VECTOR_TEXT_RENDER_PROPERTY = "property"


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
    args = render_spec.get("args", {})
    fn = VECTOR_TEXT_RENDER_FUNCTIONS.get(name)
    if fn is None:
        raise ValueError(f"Unknown vector text render method: {name}")

    features = read_vector_layer(window, layer_name, layer_config, group_idx=group_idx)

    return fn(features, window, layer_config, **args)
