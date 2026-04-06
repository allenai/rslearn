"""Functions for rendering vector layers as images (detection, segmentation)."""

from typing import Any

import numpy as np
import shapely
from PIL import Image, ImageDraw

from rslearn.config import LayerConfig
from rslearn.dataset import Dataset, Window
from rslearn.dataset.window import get_layer_and_group_from_dir_name
from rslearn.log_utils import get_logger
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import PixelBounds, flatten_shape

from .render_raster import read_raster_layer, render_raster
from .utils import read_vector_layer

logger = get_logger(__name__)

VECTOR_IMAGE_RENDER_DETECTION = "detection"
VECTOR_IMAGE_RENDER_SEGMENTATION = "segmentation"


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
) -> None:
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
    """
    if not class_property_name:
        raise ValueError(
            "class_property_name must be specified in config for vector label layers"
        )

    for feature in features:
        label = feature.properties.get(class_property_name)
        label = str(label)
        color = label_colors.get(label, (255, 0, 0))

        shp = feature.geometry.shp
        flat_shapes = flatten_shape(shp)
        for point in flat_shapes:
            assert isinstance(point, shapely.Point), (
                f"Expected Point, got {type(point)}"
            )
            px, py = point_to_pixel_coords(
                point, bounds, image_width, image_height, actual_width, actual_height
            )
            logger.debug(
                f"Point at ({point.x:.2f}, {point.y:.2f}) -> pixel ({px}, {py}), "
                f"bounds: {bounds}, image size: {image_width}x{image_height}, "
                f"actual_size: {actual_width}x{actual_height}"
            )
            draw_bounding_box_around_point(
                draw, px, py, image_width, image_height, color
            )


def render_detection(
    features: list[Feature],
    window: Window,
    layer_config: LayerConfig,
    label_colors: dict[str, tuple[int, int, int]],
    dataset: Dataset | None = None,
    bands: dict[str, list[str]] | None = None,
    raster_render: dict[str, dict[str, Any]] | None = None,
) -> np.ndarray:
    """Render vector labels for detection (overlay points on reference raster or blank background).

    Args:
        features: List of Feature objects (points)
        window: Window object
        layer_config: LayerConfig object
        label_colors: Dictionary mapping label class names to RGB colors
        dataset: Dataset object (for reading reference raster)
        bands: Dictionary mapping item_group_name -> list of band names
        raster_render: Dictionary mapping item_group_name -> render spec dict

    Returns:
        Array with shape (height, width, 3) as uint8
    """
    bands = bands or {}
    raster_render = raster_render or {}
    bounds = window.bounds

    actual_width = bounds[2] - bounds[0]
    actual_height = bounds[3] - bounds[1]

    reference_image = None
    raster_group_names = [name for name in bands.keys() if name in raster_render]
    if raster_group_names and dataset is not None:
        ref_item_group_name = raster_group_names[0]
        ref_layer_name, ref_group_idx = get_layer_and_group_from_dir_name(
            ref_item_group_name
        )
        ref_config = dataset.layers[ref_layer_name]
        ref_array = read_raster_layer(
            window,
            ref_layer_name,
            ref_config,
            bands[ref_item_group_name],
            group_idx=ref_group_idx,
        )
        ref_spec = raster_render[ref_item_group_name]
        reference_image = render_raster(ref_array, ref_config, ref_spec)

    if reference_image is not None:
        if reference_image.shape[-1] >= 3:
            img = Image.fromarray(reference_image[:, :, :3], mode="RGB")
        else:
            img = Image.fromarray(reference_image[:, :, 0], mode="L").convert("RGB")
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
        layer_config.class_property_name,
    )
    return np.array(img)


def render_segmentation(
    features: list[Feature],
    window: Window,
    layer_config: LayerConfig,
    label_colors: dict[str, tuple[int, int, int]],
    dataset: Dataset | None = None,
    bands: dict[str, list[str]] | None = None,
    raster_render: dict[str, dict[str, Any]] | None = None,
) -> np.ndarray:
    """Render vector labels for segmentation (draw polygons on mask).

    Args:
        features: List of Feature objects (polygons)
        window: Window object
        layer_config: LayerConfig object
        label_colors: Dictionary mapping label class names to RGB colors
        dataset: Dataset object (unused)
        bands: Dictionary mapping item_group_name -> list of band names (unused)
        raster_render: Dictionary mapping item_group_name -> render spec dict (unused)

    Returns:
        Array with shape (height, width, 3) as uint8
    """
    bounds = window.bounds
    projection = window.projection
    class_property_name = layer_config.class_property_name

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
        color = label_colors.get(label, (255, 0, 0))
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


VECTOR_IMAGE_RENDER_FUNCTIONS: dict[str, Any] = {
    VECTOR_IMAGE_RENDER_DETECTION: render_detection,
    VECTOR_IMAGE_RENDER_SEGMENTATION: render_segmentation,
}


def render_vector_image(
    window: Window,
    layer_name: str,
    layer_config: LayerConfig,
    render_spec: dict[str, Any],
    label_colors: dict[str, tuple[int, int, int]],
    dataset: Dataset | None = None,
    group_idx: int = 0,
    bands: dict[str, list[str]] | None = None,
    raster_render: dict[str, dict[str, Any]] | None = None,
) -> np.ndarray:
    """Dispatch to the appropriate vector image render function.

    Reads the vector features and passes them to the render function selected
    by render_spec["name"].

    Args:
        window: Window object
        layer_name: Layer name
        layer_config: LayerConfig object
        render_spec: Dict with "name" key and optional "args" dict
        label_colors: Dictionary mapping label class names to RGB colors
        dataset: Dataset object (needed for detection reference raster)
        group_idx: Item group index
        bands: Dictionary mapping item_group_name -> list of band names
        raster_render: Dictionary mapping item_group_name -> render spec dict

    Returns:
        Rendered image array (H, W, 3) uint8
    """
    name = render_spec["name"]
    args = render_spec.get("args", {})
    fn = VECTOR_IMAGE_RENDER_FUNCTIONS.get(name)
    if fn is None:
        raise ValueError(f"Unknown vector image render method: {name}")

    features = read_vector_layer(window, layer_name, layer_config, group_idx=group_idx)

    return fn(
        features,
        window,
        layer_config,
        label_colors,
        dataset=dataset,
        bands=bands,
        raster_render=raster_render,
        **args,
    )
