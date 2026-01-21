"""Visualization functions for converting arrays and features to PNG images."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw
from rasterio.crs import CRS
from rasterio.warp import transform

from rslearn.log_utils import get_logger
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import PixelBounds, Projection

from .normalization import normalize_array

if TYPE_CHECKING:
    from rslearn.config import LayerConfig

logger = get_logger(__name__)

# Fixed size for all visualized images (width, height in pixels)
VISUALIZATION_IMAGE_SIZE = (512, 512)


def array_to_png(
    array: np.ndarray,
    output_path: Path,
    normalization_method: str = "sentinel2_rgb",
) -> None:
    """Convert a numpy array to a PNG image.

    Args:
        array: Array with shape (bands, height, width) or (height, width, bands)
        output_path: Path to save PNG
        normalization_method: Normalization method to apply
    """
    # Handle (bands, height, width) format - convert to (height, width, bands)
    if array.ndim == 3 and array.shape[0] < array.shape[2]:
        array = np.moveaxis(array, 0, -1)

    # Normalize array
    normalized = normalize_array(array, normalization_method)

    # Create PIL Image
    if normalized.shape[-1] == 1:
        img = Image.fromarray(normalized[:, :, 0], mode="L")
    elif normalized.shape[-1] == 3:
        img = Image.fromarray(normalized, mode="RGB")
    else:
        # Take first 3 bands for RGB
        img = Image.fromarray(normalized[:, :, :3], mode="RGB")

    # Resize to fixed visualization size
    img = img.resize(VISUALIZATION_IMAGE_SIZE, Image.Resampling.LANCZOS)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def features_to_mask(
    features: list[Feature],
    bounds: PixelBounds,
    projection: Projection,
    label_colors: dict[str, tuple[int, int, int]],
    output_path: Path,
    reference_raster_array: np.ndarray | None = None,
) -> None:
    """Generate a mask image from vector features.

    For segmentation tasks: draws polygons on a mask.
    For detection tasks: overlays points on a reference raster image.

    Args:
        features: List of Feature objects
        bounds: Pixel bounds of the window
        projection: Projection of the window
        label_colors: Dictionary mapping label class names to RGB colors
        output_path: Path to save PNG
        reference_raster_array: Optional reference raster array for detection tasks
    """
    from collections import Counter

    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]

    # Determine if this is detection (points) or segmentation (polygons)
    geom_types = [f.geometry.shp.geom_type for f in features if f.geometry]
    is_detection = any(gt in ("Point", "MultiPoint") for gt in geom_types)

    if is_detection and reference_raster_array is not None:
        # Detection: overlay points on reference image
        # First, convert reference array to PNG (in memory)
        # Then overlay points
        _overlay_points_on_array(
            reference_raster_array, features, bounds, projection, label_colors, output_path
        )
    else:
        # Segmentation: draw polygons on mask
        _draw_mask_from_features(features, width, height, bounds, projection, label_colors, output_path)


def raster_label_to_mask(
    label_array: np.ndarray,
    label_colors: dict[str, tuple[int, int, int]],
    layer_config: "LayerConfig",
    output_path: Path,
) -> None:
    """Convert a raster label array to a colored mask image.

    Args:
        label_array: Raster label array with shape (bands, height, width) - typically single band
        label_colors: Dictionary mapping label class names to RGB color tuples
        layer_config: LayerConfig object (to access class_names if available)
        output_path: Path to save PNG mask
    """
    # Use first band
    if label_array.ndim == 3:
        label_values = label_array[0, :, :]
    else:
        label_values = label_array

    height, width = label_values.shape
    
    # Create RGB mask image
    mask_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a mask for valid (non-NaN) values
    valid_mask = ~np.isnan(label_values)
    
    # Map label values to colors
    # If class_names is available, map numeric indices to class names
    # Otherwise, use numeric values directly as strings
    if layer_config.class_names:
        # Map numeric values to class names using vectorized operations
        label_int = label_values.astype(np.int32)
        
        # Create mapping array: index -> class name string
        for idx in range(len(layer_config.class_names)):
            class_name = layer_config.class_names[idx]
            color = label_colors.get(str(class_name), (0, 0, 0))
            # Set all pixels with this index to this color
            mask = (label_int == idx) & valid_mask
            mask_img[mask] = color
    else:
        # No class_names - use numeric values directly
        # Get unique values and map them
        unique_vals = np.unique(label_values[valid_mask])
        
        for val in unique_vals:
            # Map 0 to "no_data", otherwise use numeric value as string
            if np.isclose(val, 0):
                val_str = "no_data"
            elif np.isclose(val, int(val)):
                val_str = str(int(val))
            else:
                val_str = str(float(val))
            color = label_colors.get(val_str, (0, 0, 0))
            # Set all pixels with this value to this color
            mask = (label_values == val) & valid_mask
            mask_img[mask] = color
    
    # NaN values are already black (0, 0, 0) from initialization
    
    # Convert to PIL Image and resize
    mask_pil = Image.fromarray(mask_img, mode="RGB")
    mask_pil = mask_pil.resize(VISUALIZATION_IMAGE_SIZE, Image.Resampling.NEAREST)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_pil.save(output_path)


def _draw_mask_from_features(
    features: list[Feature],
    width: int,
    height: int,
    bounds: PixelBounds,
    projection: Projection,
    label_colors: dict[str, tuple[int, int, int]],
    output_path: Path,
) -> None:
    """Draw a mask from polygon features."""
    from collections import Counter

    # Initialize mask with black background
    # "no_data" is always black and represents buffers around other polygons
    background_color = (0, 0, 0)  # Black background
    mask_img = Image.new("RGB", (width, height), color=background_color)
    draw = ImageDraw.Draw(mask_img)

    # Sort features for drawing order:
    # 1. "no_data" first (black buffers)
    # 2. Other labels in reverse alphabetical order (descending)
    #    This ensures labels that come later alphabetically (like "no_landslide") 
    #    are drawn before labels that come earlier (like "landslide"),
    #    so "landslide" appears on top of "no_landslide"
    property_names = ["label", "category", "class", "type"]
    
    # Separate no_data features from others
    no_data_features = []
    other_features = []
    for feat in features:
        if not feat.properties:
            other_features.append(feat)
            continue
        label_found = False
        for prop_name in property_names:
            label = feat.properties.get(prop_name, "")
            if label:
                if str(label) == "no_data":
                    no_data_features.append(feat)
                else:
                    other_features.append(feat)
                label_found = True
                break
        if not label_found:
            other_features.append(feat)
    
    # Sort other features in reverse alphabetical order (descending)
    def get_label(feat: Feature) -> str:
        if not feat.properties:
            return ""
        for prop_name in property_names:
            label = feat.properties.get(prop_name, "")
            if label:
                return str(label)
        return ""
    
    other_features_sorted = sorted(other_features, key=get_label, reverse=True)
    sorted_features = no_data_features + other_features_sorted

    # Draw polygons
    for feature in sorted_features:
        if not feature.properties:
            continue

        # Get label
        label = None
        for prop_name in property_names:
            label = feature.properties.get(prop_name)
            if label:
                label = str(label)
                break

        if not label:
            continue

        color = label_colors.get(label, (255, 255, 255))

        # Get geometry in pixel coordinates
        geom_pixel = feature.geometry.to_projection(projection)
        shp = geom_pixel.shp

        # Draw polygon
        if shp.geom_type == "Polygon":
            coords = list(shp.exterior.coords)
            # Convert to pixel coordinates relative to bounds
            pixel_coords = [
                (int(x - bounds[0]), int(y - bounds[1])) for x, y in coords
            ]
            if len(pixel_coords) >= 3:
                draw.polygon(pixel_coords, fill=color, outline=color)
                # Draw holes
                for interior in shp.interiors:
                    hole_coords = [(int(x - bounds[0]), int(y - bounds[1])) for x, y in interior.coords]
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

    # Resize to fixed visualization size
    mask_img = mask_img.resize(VISUALIZATION_IMAGE_SIZE, Image.Resampling.NEAREST)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_img.save(output_path)


def _overlay_points_on_array(
    array: np.ndarray,
    features: list[Feature],
    bounds: PixelBounds,
    projection: Projection,
    label_colors: dict[str, tuple[int, int, int]],
    output_path: Path,
) -> None:
    """Overlay point features on a raster array."""
    # First convert array to image
    # Handle (bands, height, width) format
    if array.ndim == 3 and array.shape[0] < array.shape[2]:
        array = np.moveaxis(array, 0, -1)

    # Normalize (use sentinel2_rgb as default)
    from .normalization import normalize_array

    normalized = normalize_array(array, "sentinel2_rgb")
    if normalized.shape[-1] >= 3:
        img = Image.fromarray(normalized[:, :, :3], mode="RGB")
    else:
        img = Image.fromarray(normalized[:, :, 0], mode="L").convert("RGB")

    draw = ImageDraw.Draw(img)
    width, height = img.size
    actual_width = bounds[2] - bounds[0]
    actual_height = bounds[3] - bounds[1]

    property_names = ["label", "category", "class", "type"]

    points_drawn = 0
    points_out_of_bounds = 0
    logger.info(f"Processing {len(features)} features for overlay")
    
    for feature in features:
        # Get label (use "detected" as default if no label property found)
        label = None
        if feature.properties:
            for prop_name in property_names:
                label = feature.properties.get(prop_name)
                if label:
                    label = str(label)
                    break
        
        # If no label found, use "detected" as default (for object detection)
        if not label:
            label = "detected"

        color = label_colors.get(label, (255, 0, 0))

        # Get geometry in pixel coordinates
        # The feature geometry is already in the window projection (from read_vector_layer)
        # So we can use it directly
        shp = feature.geometry.shp

        # Draw bounding boxes around points
        if shp.geom_type == "Point":
            x, y = shp.x, shp.y
            px = int((x - bounds[0]) * width / actual_width)
            py = int((y - bounds[1]) * height / actual_height)
            logger.info(f"Point at ({x:.2f}, {y:.2f}) -> pixel ({px}, {py}), bounds: {bounds}, image size: {width}x{height}, actual_size: {actual_width}x{actual_height}")
            if 0 <= px < width and 0 <= py < height:
                # Draw a bounding box (e.g., 20x20 pixels centered on the point)
                box_size = 20
                x1 = max(0, px - box_size // 2)
                y1 = max(0, py - box_size // 2)
                x2 = min(width, px + box_size // 2)
                y2 = min(height, py + box_size // 2)
                draw.rectangle(
                    [x1, y1, x2, y2],
                    outline=color,
                    width=2,
                )
                points_drawn += 1
            else:
                points_out_of_bounds += 1
        elif shp.geom_type == "MultiPoint":
            for point in shp.geoms:
                x, y = point.x, point.y
                px = int((x - bounds[0]) * width / actual_width)
                py = int((y - bounds[1]) * height / actual_height)
                logger.debug(f"MultiPoint at ({x:.2f}, {y:.2f}) -> pixel ({px}, {py}), bounds: {bounds}, image size: {width}x{height}")
                if 0 <= px < width and 0 <= py < height:
                    # Draw a bounding box (e.g., 20x20 pixels centered on the point)
                    box_size = 20
                    x1 = max(0, px - box_size // 2)
                    y1 = max(0, py - box_size // 2)
                    x2 = min(width, px + box_size // 2)
                    y2 = min(height, py + box_size // 2)
                    draw.rectangle(
                        [x1, y1, x2, y2],
                        outline=color,
                        width=2,
                    )
                    points_drawn += 1
                else:
                    points_out_of_bounds += 1

    logger.info(f"Overlaid {points_drawn} points on image ({points_out_of_bounds} out of bounds)")

    # Resize to fixed visualization size
    img = img.resize(VISUALIZATION_IMAGE_SIZE, Image.Resampling.LANCZOS)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)

