"""Functions for rendering vector label masks (detection and segmentation)."""

from typing import Any

import numpy as np
import shapely
from PIL import Image, ImageDraw

from rslearn.config import LayerConfig, LayerType
from rslearn.dataset import Dataset, Window
from rslearn.log_utils import get_logger
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import PixelBounds, Projection, flatten_shape
from rslearn.utils.vector_format import VectorFormat

from .normalization import normalize_array
from .render_raster_label import read_raster_layer

logger = get_logger(__name__)


def point_to_pixel_coords(
    point: shapely.Point,
    bounds: PixelBounds,
    image_width: int,
    image_height: int,
    actual_width: int,
    actual_height: int,
) -> tuple[int, int]:
    """Convert a point's coordinates to pixel coordinates in the image.

    Args:
        point: Shapely Point object
        bounds: Pixel bounds of the window
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        actual_width: Actual width of the data (bounds[2] - bounds[0])
        actual_height: Actual height of the data (bounds[3] - bounds[1])

    Returns:
        Tuple of (pixel_x, pixel_y) coordinates
    """
    x, y = point.x, point.y
    px = int((x - bounds[0]) * image_width / actual_width)
    py = int((y - bounds[1]) * image_height / actual_height)
    return px, py


def draw_bounding_box_around_point(
    draw: ImageDraw.ImageDraw,
    px: int,
    py: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
    box_size: int = 20,
) -> bool:
    """Draw a bounding box around a point on an image.

    Args:
        draw: PIL ImageDraw object
        px: Pixel x coordinate
        py: Pixel y coordinate
        width: Image width
        height: Image height
        color: RGB color tuple for the bounding box
        box_size: Size of the bounding box in pixels (default 20)

    Returns:
        True if the point was drawn (within bounds), False otherwise
    """
    if 0 <= px < width and 0 <= py < height:
        x1 = max(0, px - box_size // 2)
        y1 = max(0, py - box_size // 2)
        x2 = min(width, px + box_size // 2)
        y2 = min(height, py + box_size // 2)
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=color,
            width=2,
        )
        return True
    return False


def overlay_points_on_image(
    draw: ImageDraw.ImageDraw,
    features: list[Feature],
    bounds: PixelBounds,
    image_width: int,
    image_height: int,
    actual_width: int,
    actual_height: int,
    label_colors: dict[str, tuple[int, int, int]],
    class_property_name: str | None = None,
) -> tuple[int, int]:
    """Overlay point features on an image.

    Args:
        draw: PIL ImageDraw object
        features: List of Feature objects (points)
        bounds: Pixel bounds of the window
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        actual_width: Actual width of the data (bounds[2] - bounds[0])
        actual_height: Actual height of the data (bounds[3] - bounds[1])
        label_colors: Dictionary mapping label class names to RGB colors
        class_property_name: Property name to use for label extraction (from config)

    Returns:
        Tuple of (points_drawn, points_out_of_bounds)
    """
    points_drawn = 0
    points_out_of_bounds = 0

    if not class_property_name:
        raise ValueError(
            "class_property_name must be specified in config for vector label layers"
        )

    for feature in features:
        label = feature.properties.get(class_property_name)
        label = str(label)
        color = label_colors.get(label)
        if color is None:
            raise ValueError(
                f"Label '{label}' not found in label_colors. Available labels: {list(label_colors.keys())}"
            )

        shp = feature.geometry.shp
        flat_shapes = flatten_shape(shp)
        for point in flat_shapes:
            assert isinstance(point, shapely.Point), (
                f"Expected Point, got {type(point)}"
            )
            px, py = point_to_pixel_coords(
                point, bounds, image_width, image_height, actual_width, actual_height
            )
            logger.info(
                f"Point at ({point.x:.2f}, {point.y:.2f}) -> pixel ({px}, {py}), "
                f"bounds: {bounds}, image size: {image_width}x{image_height}, "
                f"actual_size: {actual_width}x{actual_height}"
            )
            if draw_bounding_box_around_point(
                draw, px, py, image_width, image_height, color
            ):
                points_drawn += 1
            else:
                points_out_of_bounds += 1

    return points_drawn, points_out_of_bounds


def render_vector_label_detection(
    features: list[Feature],
    bounds: PixelBounds,
    label_colors: dict[str, tuple[int, int, int]],
    class_property_name: str | None = None,
    reference_array: np.ndarray | None = None,
    normalization_method: str | None = None,
) -> np.ndarray:
    """Render vector labels for detection tasks (overlay points on reference image or blank background).

    Args:
        features: List of Feature objects (points)
        bounds: Pixel bounds of the window
        label_colors: Dictionary mapping label class names to RGB colors
        class_property_name: Property name to use for label extraction (from config)
        reference_array: Optional reference raster array to overlay points on
        normalization_method: Optional normalization method for the reference array

    Returns:
        Array with shape (height, width, 3) as uint8
    """
    actual_width = bounds[2] - bounds[0]
    actual_height = bounds[3] - bounds[1]

    if reference_array is not None and normalization_method is not None:
        normalized = normalize_array(reference_array, normalization_method)
        if normalized.shape[-1] >= 3:
            img = Image.fromarray(normalized[:, :, :3], mode="RGB")
        else:
            img = Image.fromarray(normalized[:, :, 0], mode="L").convert("RGB")
    else:
        img = Image.new("RGB", (actual_width, actual_height), color=(0, 0, 0))

    draw = ImageDraw.Draw(img)

    overlay_points_on_image(
        draw,
        features,
        bounds,
        img.size[0],
        img.size[1],
        actual_width,
        actual_height,
        label_colors,
        class_property_name,
    )
    return np.array(img)


def render_vector_label_segmentation(
    features: list[Feature],
    bounds: PixelBounds,
    projection: Projection,
    label_colors: dict[str, tuple[int, int, int]],
    class_property_name: str | None = None,
) -> np.ndarray:
    """Render vector labels for segmentation tasks (draw polygons on mask).

    Args:
        features: List of Feature objects (polygons)
        bounds: Pixel bounds of the window
        projection: Projection of the window
        label_colors: Dictionary mapping label class names to RGB colors
        class_property_name: Property name to use for label extraction (from config)

    Returns:
        Array with shape (height, width, 3) as uint8
    """
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]

    mask_img = Image.new("RGB", (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(mask_img)

    if not class_property_name:
        raise ValueError(
            "class_property_name must be specified in config for vector label layers"
        )

    def get_label(feat: Any) -> str:
        if not feat.properties:
            return ""
        label = feat.properties.get(class_property_name)
        return str(label) if label else ""

    sorted_features = sorted(features, key=get_label, reverse=True)

    for feature in sorted_features:
        label = feature.properties.get(class_property_name)
        label = str(label)
        color = label_colors.get(label, (255, 255, 255))
        geom_pixel = feature.geometry.to_projection(projection)
        shp = geom_pixel.shp
        if shp.geom_type == "Polygon":
            coords = list(shp.exterior.coords)
            pixel_coords = [(int(x - bounds[0]), int(y - bounds[1])) for x, y in coords]
            if len(pixel_coords) >= 3:
                draw.polygon(pixel_coords, fill=color, outline=color)
                for interior in shp.interiors:
                    hole_coords = [
                        (int(x - bounds[0]), int(y - bounds[1]))
                        for x, y in interior.coords
                    ]
                    if len(hole_coords) >= 3:
                        draw.polygon(hole_coords, fill=(0, 0, 0), outline=color)
        elif shp.geom_type == "MultiPolygon":
            for poly in shp.geoms:
                coords = list(poly.exterior.coords)
                pixel_coords = [
                    (int(x - bounds[0]), int(y - bounds[1])) for x, y in coords
                ]
                if len(pixel_coords) >= 3:
                    draw.polygon(pixel_coords, fill=color, outline=color)

    return np.array(mask_img)


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
    layer_dir = window.get_vector_layer_dir(layer_name, group_idx=group_idx)
    logger.info(
        f"Reading vector layer {layer_name} from {layer_dir}, bounds: {window.bounds}, projection: {window.projection}"
    )

    features = vector_format.decode_vector(layer_dir, window.projection, window.bounds)
    logger.info(f"Decoded {len(features)} features from vector layer {layer_name}")
    return features


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
    features = read_vector_layer(window, layer_name, layer_config, group_idx=group_idx)
    if not features:
        logger.warning(
            f"No features in vector label layer {layer_name} for {window.name}"
        )
        return None

    first_feature = features[0]
    if not first_feature.properties:
        return None

    if not layer_config.class_property_name:
        raise ValueError(
            f"class_property_name must be specified in the config for vector label layer '{layer_name}'. "
        )

    label = first_feature.properties.get(layer_config.class_property_name)
    logger.info(f"Label for {window.name}: {label}")
    return str(label)


def _get_reference_raster_for_detection(
    window: Window,
    dataset: Dataset,
    bands: dict[str, list[str]],
    normalization: dict[str, str],
    label_layers: list[str],
    group_idx: int,
) -> tuple[np.ndarray, str] | None:
    """Get reference raster array for detection tasks.

    Args:
        window: Window object
        dataset: Dataset object
        bands: Dictionary mapping layer_name -> list of band names
        normalization: Dictionary mapping layer_name -> normalization method
        label_layers: List of label layer names
        group_idx: Item group index

    Returns:
        Tuple of (reference_array, normalization_method) or None if no raster layers available
    """
    raster_layers = [name for name in bands.keys() if name not in label_layers]

    ref_layer_name = raster_layers[0]
    ref_layer_config = dataset.layers[ref_layer_name]
    reference_array = read_raster_layer(
        window,
        ref_layer_name,
        ref_layer_config,
        bands[ref_layer_name],
        group_idx=group_idx,
    )
    ref_normalization_method = normalization[ref_layer_name]
    return reference_array, ref_normalization_method


def render_vector_label_image(
    window: Window,
    layer_name: str,
    layer_config: LayerConfig,
    task_type: str,
    label_colors: dict[str, tuple[int, int, int]],
    dataset: Dataset,
    label_layers: list[str],
    group_idx: int,
    bands: dict[str, list[str]] | None = None,
    normalization: dict[str, str] | None = None,
) -> np.ndarray:
    """Render a vector label image (detection or segmentation).

    Args:
        window: Window object
        layer_name: Layer name
        layer_config: LayerConfig object
        task_type: Task type ("detection" or "segmentation")
        label_colors: Dictionary mapping label class names to RGB colors
        dataset: Dataset object
        label_layers: List of label layer names
        group_idx: Item group index
        bands: Optional dictionary mapping layer_name -> list of band names (for detection reference raster)
        normalization: Optional dictionary mapping layer_name -> normalization method (for detection reference raster)

    Returns:
        Array with shape (height, width, 3) as uint8
    """
    if task_type == "classification":
        raise ValueError("Classification labels are text, not images")

    features = read_vector_layer(window, layer_name, layer_config, group_idx=group_idx)

    if task_type == "detection":
        bands = bands or {}
        normalization = normalization or {}
        ref_data = _get_reference_raster_for_detection(
            window, dataset, bands, normalization, label_layers, group_idx
        )
        reference_array = None
        ref_normalization_method = None
        if ref_data is not None:
            reference_array, ref_normalization_method = ref_data
        return render_vector_label_detection(
            features,
            window.bounds,
            label_colors,
            layer_config.class_property_name,
            reference_array,
            ref_normalization_method,
        )
    else:
        return render_vector_label_segmentation(
            features,
            window.bounds,
            window.projection,
            label_colors,
            layer_config.class_property_name,
        )
