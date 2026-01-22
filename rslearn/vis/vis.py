"""Visualization functions for converting arrays and features to PNG images."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from rslearn.config import LayerConfig
from rslearn.log_utils import get_logger
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import PixelBounds, Projection

from .normalization import normalize_array

logger = get_logger(__name__)

# Fixed size for all visualized images (width, height in pixels)
VISUALIZATION_IMAGE_SIZE = (512, 512)


def array_to_png(
    array: np.ndarray,
    output_path: Path,
    normalization_method: str = "sentinel2_rgb",
) -> None:
    """Convert a numpy array to a PNG image - only intended to be used for images with continuous values like satellite images.

    Args:
        array: Array with shape (channels, height, width) from RasterFormat.decode_raster
        output_path: Path to save PNG
        normalization_method: Normalization method to apply
    """
    normalized = normalize_array(array, normalization_method)
    if normalized.shape[-1] == 1:
        img = Image.fromarray(normalized[:, :, 0], mode="L")
    elif normalized.shape[-1] == 3:
        img = Image.fromarray(normalized, mode="RGB")
    else:
        img = Image.fromarray(normalized[:, :, :3], mode="RGB")

    img = img.resize(VISUALIZATION_IMAGE_SIZE, Image.Resampling.LANCZOS)

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
    normalization_method: str = "sentinel2_rgb",
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
        normalization_method: Normalization method for the reference raster array (detection only)
    """
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]

    geom_types = [f.geometry.shp.geom_type for f in features if f.geometry]
    is_detection = any(gt in ("Point", "MultiPoint") for gt in geom_types)

    if is_detection and reference_raster_array is not None:
        _overlay_points_on_array(
            reference_raster_array,
            features,
            bounds,
            projection,
            label_colors,
            output_path,
            normalization_method,
        )
    else:
        _draw_mask_from_features(
            features, width, height, bounds, projection, label_colors, output_path
        )


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
    if label_array.ndim == 3:
        label_values = label_array[0, :, :]
    else:
        label_values = label_array

    height, width = label_values.shape

    mask_img = np.zeros((height, width, 3), dtype=np.uint8)
    valid_mask = ~np.isnan(label_values)

    if layer_config.class_names:
        label_int = label_values.astype(np.int32)
        for idx in range(len(layer_config.class_names)):
            class_name = layer_config.class_names[idx]
            color = label_colors.get(str(class_name), (0, 0, 0))
            mask = (label_int == idx) & valid_mask
            mask_img[mask] = color
    else:
        unique_vals = np.unique(label_values[valid_mask])

        for val in unique_vals:
            if np.isclose(val, 0):
                val_str = "no_data"
            elif np.isclose(val, int(val)):
                val_str = str(int(val))
            else:
                val_str = str(float(val))
            color = label_colors.get(val_str, (0, 0, 0))
            mask = (label_values == val) & valid_mask
            mask_img[mask] = color

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
    background_color = (0, 0, 0)
    mask_img = Image.new("RGB", (width, height), color=background_color)
    draw = ImageDraw.Draw(mask_img)

    property_names = ["label", "category", "class", "type"]
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
    normalization_method: str = "sentinel2_rgb",
) -> None:
    """Overlay point features on a raster array.

    Args:
        array: Raster array to overlay points on
        features: List of Feature objects (points)
        bounds: Pixel bounds of the window
        projection: Projection of the window
        label_colors: Dictionary mapping label class names to RGB colors
        output_path: Path to save PNG
        normalization_method: Normalization method for the array
    """
    from .normalization import normalize_array

    normalized = normalize_array(array, normalization_method)
    if normalized.shape[-1] >= 3:
        img = Image.fromarray(normalized[:, :, :3], mode="RGB")
    else:
        img = Image.fromarray(normalized[:, :, 0], mode="L").convert("RGB")

    draw = ImageDraw.Draw(img)
    width, height = img.size
    actual_width = bounds[2] - bounds[0]
    actual_height = bounds[3] - bounds[1]

    from .overlay import overlay_points_on_image

    logger.info(f"Processing {len(features)} features for overlay")
    points_drawn, points_out_of_bounds = overlay_points_on_image(
        draw, features, bounds, width, height, actual_width, actual_height, label_colors
    )
    logger.info(
        f"Overlaid {points_drawn} points on image ({points_out_of_bounds} out of bounds)"
    )

    img = img.resize(VISUALIZATION_IMAGE_SIZE, Image.Resampling.LANCZOS)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
