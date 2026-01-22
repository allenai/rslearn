"""Functions for overlaying point features on raster images."""

from typing import Any

import shapely
from PIL import ImageDraw

from rslearn.log_utils import get_logger
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import PixelBounds, flatten_shape

logger = get_logger(__name__)

PROPERTY_NAMES = ["label", "category", "class", "type"]


def extract_label_from_feature(feature: Feature, default: str = "detected") -> str:
    """Extract label from a feature's properties.
    
    Args:
        feature: Feature object
        default: Default label to use if none found
        
    Returns:
        Label string
    """
    if feature.properties:
        for prop_name in PROPERTY_NAMES:
            label = feature.properties.get(prop_name)
            if label:
                return str(label)
    return default


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
        
    Returns:
        Tuple of (points_drawn, points_out_of_bounds)
    """
    points_drawn = 0
    points_out_of_bounds = 0
    
    for feature in features:
        label = extract_label_from_feature(feature)
        color = label_colors.get(label, (255, 0, 0))
        
        shp = feature.geometry.shp
        flat_shapes = flatten_shape(shp)
        for point in flat_shapes:
            assert isinstance(point, shapely.Point), f"Expected Point, got {type(point)}"
            px, py = point_to_pixel_coords(
                point, bounds, image_width, image_height, actual_width, actual_height
            )
            logger.info(
                f"Point at ({point.x:.2f}, {point.y:.2f}) -> pixel ({px}, {py}), "
                f"bounds: {bounds}, image size: {image_width}x{image_height}, "
                f"actual_size: {actual_width}x{actual_height}"
            )
            if draw_bounding_box_around_point(draw, px, py, image_width, image_height, color):
                points_drawn += 1
            else:
                points_out_of_bounds += 1
    
    return points_drawn, points_out_of_bounds

