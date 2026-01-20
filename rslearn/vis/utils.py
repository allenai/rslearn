"""Utility functions for rslearn dataset visualization."""

import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.warp import transform
from rasterio.crs import CRS
import rasterio.transform

from rslearn.log_utils import get_logger

logger = get_logger(__name__)

# Fixed size for all visualized images (width, height in pixels)
VISUALIZATION_IMAGE_SIZE = (512, 512)


def normalize_band(band, method="sentinel2_rgb"):
    """Normalize band to 0-255 range.

    Args:
        band: Input band data
        method: Normalization method
            - 'sentinel2_rgb': Divide by 10 and clip (for B04/B03/B02)
            - 'percentile': Use 2-98 percentile clipping
            - 'minmax': Use min-max stretch
    """
    if method == "sentinel2_rgb":
        band = band / 10.0
        band = np.clip(band, 0, 255).astype(np.uint8)
    elif method == "percentile":
        valid_pixels = band[~np.isnan(band)]
        if len(valid_pixels) == 0:
            return np.zeros_like(band, dtype=np.uint8)
        vmin, vmax = np.nanpercentile(valid_pixels, (2, 98))
        if vmax == vmin:
            return np.zeros_like(band, dtype=np.uint8)
        band = np.clip(band, vmin, vmax)
        band = ((band - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    elif method == "minmax":
        vmin, vmax = np.nanmin(band), np.nanmax(band)
        if vmax == vmin:
            return np.zeros_like(band, dtype=np.uint8)
        band = np.clip(band, vmin, vmax)
        band = ((band - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return band


def get_rgb_bands_from_config(layer_config):
    """Determine RGB band indices from layer config.
    
    Args:
        layer_config: Layer config dictionary with band_sets
        
    Returns:
        List of band indices (1-indexed) for RGB in order [Red, Green, Blue], or None if not determinable
    """
    band_sets = layer_config.get("band_sets", [])
    if not band_sets:
        return None
    
    band_set = band_sets[0]
    bands = band_set.get("bands", [])
    if not bands:
        return None
    
    # For RGB visualization, we need: Red=B04, Green=B03, Blue=B02
    # Find the indices of these bands in the config
    rgb_band_map = {"B04": None, "B03": None, "B02": None}
    for i, band_name in enumerate(bands, start=1):
        if band_name in rgb_band_map:
            rgb_band_map[band_name] = i
    
    # Return in RGB order: [Red_idx, Green_idx, Blue_idx] = [B04_idx, B03_idx, B02_idx]
    if all(rgb_band_map[bn] is not None for bn in ["B04", "B03", "B02"]):
        return [rgb_band_map["B04"], rgb_band_map["B03"], rgb_band_map["B02"]]
    return None


def band_names_to_indices(band_names, layer_config):
    """Convert band names to 1-indexed band indices based on layer config.
    
    Args:
        band_names: List of band names (e.g., ["B04", "B03", "B02"])
        layer_config: Layer config dictionary with band_sets
        
    Returns:
        List of band indices (1-indexed) in the same order as band_names, or None if not all bands found
    """
    band_sets = layer_config.get("band_sets", [])
    if not band_sets:
        return None
    
    band_set = band_sets[0]
    config_bands = band_set.get("bands", [])
    if not config_bands:
        return None
    
    # Create a mapping from band name to index (1-indexed)
    name_to_index = {name: i for i, name in enumerate(config_bands, start=1)}
    
    # Convert band names to indices
    indices = []
    for band_name in band_names:
        if band_name in name_to_index:
            indices.append(name_to_index[band_name])
        else:
            logger.warning(f"Band name '{band_name}' not found in layer config bands: {config_bands}")
            return None
    
    return indices


def visualize_tif(input_path, output_path, bands=None, normalize_method="sentinel2_rgb", layer_config=None):
    """Convert GeoTIFF to PNG visualization.

    Args:
        input_path: Path to input GeoTIFF
        output_path: Path to output PNG
        bands: List of band indices (1-indexed). If None, will try to determine from layer_config or use defaults
        normalize_method: Method for normalization
        layer_config: Optional layer config dictionary to determine band order
    """
    with rasterio.open(input_path) as src:
        if bands is None and layer_config:
            bands = get_rgb_bands_from_config(layer_config)
            if bands:
                logger.debug(f"Using config bands {bands} for RGB visualization")
        
        if bands is None:
            if src.count >= 4:
                bands = [3, 2, 1]  # Default fallback
            elif src.count >= 3:
                bands = [1, 2, 3]
            elif src.count == 1:
                bands = [1]
            else:
                raise ValueError(f"Unexpected number of bands: {src.count}")
            logger.debug(f"Using default bands {bands} for RGB visualization (band count: {src.count})")

        if len(bands) == 1:
            band_data = src.read(bands[0])
            band_normalized = normalize_band(band_data, method=normalize_method)
            img = Image.fromarray(band_normalized, mode="L")
            # Resize image to fixed visualization size
            img = img.resize(VISUALIZATION_IMAGE_SIZE, Image.Resampling.LANCZOS)
        else:
            # Read bands in order: [Red_band, Green_band, Blue_band]
            # and stack them so rgb_arr[:,:,0]=Red, rgb_arr[:,:,1]=Green, rgb_arr[:,:,2]=Blue
            rgb = []
            for b in bands[:3]:
                band_data = src.read(b)
                rgb.append(normalize_band(band_data, method=normalize_method))
            rgb_arr = np.stack(rgb, axis=-1)
            img = Image.fromarray(rgb_arr, mode="RGB")
        
        # Resize image to fixed visualization size
        img = img.resize(VISUALIZATION_IMAGE_SIZE, Image.Resampling.LANCZOS)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)


def detect_label_classes(window_dirs):
    """Detect all unique label classes from GeoJSON files in windows.
    
    Args:
        window_dirs: List of window directory paths
        
    Returns:
        Set of unique label class names
    """
    label_classes = set()
    
    for window_dir in window_dirs:
        label_dir = Path(window_dir) / "layers" / "label"
        geojson_path = label_dir / "data.geojson"
        
        if not geojson_path.exists():
            continue
            
        try:
            with open(geojson_path, "r") as f:
                geojson_data = json.load(f)
            
            features = geojson_data.get("features", [])
            for feature in features:
                props = feature.get("properties", {})
                label = props.get("label") or props.get("category") or props.get("class") or props.get("type")
                if label:
                    label_classes.add(label)
        except Exception as e:
            logger.debug(f"Failed to read {geojson_path}: {e}")
            continue
    
    return label_classes


def generate_label_colors(label_classes):
    """Generate color mapping for label classes.
    
    Always assigns "no_data" to black. Other classes get distinct colors from palette.
    
    Args:
        label_classes: Set of label class names
        
    Returns:
        Dictionary mapping label class name to RGB color tuple
    """
    NO_DATA_COLOR = (0, 0, 0)  # Black
    
    color_palette = [
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (0, 128, 0),
        (255, 160, 122),
        (139, 69, 19),
        (143, 188, 143),
        (95, 158, 160),
        (255, 200, 0),
        (128, 0, 0),
        (255, 165, 0),
        (128, 0, 128),
        (0, 128, 128),
        (255, 192, 203),
        (160, 82, 45),
        (70, 130, 180),
        (255, 140, 0),
        (255, 20, 147),
        (0, 191, 255),
        (255, 215, 0),
        (50, 205, 50),
        (255, 99, 71),
    ]
    
    label_colors = {}
    color_idx = 0
    
    if "no_data" in label_classes:
        label_colors["no_data"] = NO_DATA_COLOR
    
    sorted_classes = sorted([c for c in label_classes if c != "no_data"])
    
    for label in sorted_classes:
        label_colors[label] = color_palette[color_idx % len(color_palette)]
        color_idx += 1
    
    return label_colors


def overlay_points_on_image(image_path, output_path, geojson_path, reference_tif_path, label_colors, metadata_bounds=None):
    """Overlay point geometries from GeoJSON onto an existing image.
    
    Args:
        image_path: Path to input PNG image
        output_path: Path to save output PNG with overlaid points
        geojson_path: Path to data.geojson file with Point geometries
        reference_tif_path: Path to reference GeoTIFF for coordinate transformation
        label_colors: Dictionary mapping label class names to RGB color tuples
        metadata_bounds: Optional window bounds from metadata.json [min_x, min_y, max_x, max_y].
                        If provided, coordinates are treated as offsets from (min_x, min_y).
    """
    with open(geojson_path, "r") as f:
        geojson_data = json.load(f)

    features = geojson_data.get("features", [])
    if not features:
        logger.warning(f"No features found in {geojson_path}")
        return 0

    # Load the image
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # Get CRS and transform info from reference GeoTIFF
    try:
        with rasterio.open(reference_tif_path) as src:
            bounds = src.bounds
            tif_crs = src.crs
            tif_transform = src.transform
            min_x, min_y, max_x, max_y = bounds.left, bounds.bottom, bounds.right, bounds.top
            logger.info(f"Reference GeoTIFF bounds: [{min_x:.2f}, {min_y:.2f}, {max_x:.2f}, {max_y:.2f}], CRS: {tif_crs}, image size: {width}x{height}")
    except Exception as e:
        logger.error(f"Could not read reference GeoTIFF: {e}")
        return 0

    # Parse GeoJSON CRS (same logic as generate_mask_from_geojson)
    geojson_crs = geojson_data.get("crs", {})
    src_crs = None
    
    if geojson_crs:
        crs_props = geojson_crs.get("properties", {})
        crs_name = crs_props.get("name", "")
        if "EPSG" in crs_name or "epsg" in crs_name:
            import re
            match = re.search(r"EPSG[:/](\d+)", crs_name, re.IGNORECASE)
            if match:
                src_crs = CRS.from_epsg(int(match.group(1)))
    
    if src_crs is None:
        props = geojson_data.get("properties", {})
        crs_str = props.get("crs", "")
        if crs_str and ("EPSG" in crs_str or "epsg" in crs_str):
            import re
            match = re.search(r"EPSG[:/](\d+)", crs_str, re.IGNORECASE)
            if match:
                src_crs = CRS.from_epsg(int(match.group(1)))
    
    if src_crs is None:
        src_crs = CRS.from_epsg(4326)

    # Collect points by label
    points_by_label = defaultdict(list)
    
    for feature in features:
        geometry = feature.get("geometry", {})
        geom_type = geometry.get("type")
        props = feature.get("properties", {})
        label = props.get("label") or props.get("category") or props.get("class") or props.get("type") or "unknown"

        if geom_type == "Point":
            coordinates = geometry.get("coordinates", [])
            if len(coordinates) >= 2:
                points_by_label[label].append((coordinates[0], coordinates[1]))
        elif geom_type == "MultiPoint":
            for point_coords in geometry.get("coordinates", []):
                if len(point_coords) >= 2:
                    points_by_label[label].append((point_coords[0], point_coords[1]))

    if not points_by_label:
        logger.warning(f"No point geometries found in {geojson_path}")
        return 0

    total_points = sum(len(pts) for pts in points_by_label.values())
    logger.info(f"Found {total_points} points in {len(points_by_label)} labels: {list(points_by_label.keys())}")
    logger.info(f"GeoJSON CRS: {src_crs}, GeoTIFF CRS: {tif_crs}")

    def coords_to_pixel(x, y):
        px = int((x - min_x) / (max_x - min_x) * width) if max_x != min_x else width // 2
        py = int((max_y - y) / (max_y - min_y) * height) if max_y != min_y else height // 2
        return (px, py)

    points_drawn = 0
    points_out_of_bounds = 0
    points_transform_failed = 0
    
    for label, point_list in points_by_label.items():
        color = label_colors.get(label, (255, 0, 0))  # Default to red if color not found
        
        for x, y in point_list:
            # Convert coordinates from offsets to absolute if metadata_bounds provided
            if metadata_bounds and len(metadata_bounds) >= 2:
                # Coordinates are stored as offsets from metadata_bounds[0] and metadata_bounds[1]
                x_absolute = metadata_bounds[0] + x
                y_absolute = metadata_bounds[1] + y
            else:
                # Coordinates are already absolute
                x_absolute, y_absolute = x, y
            
            # Transform coordinates if CRS differs
            if src_crs != tif_crs:
                try:
                    xs_transformed, ys_transformed = transform(src_crs, tif_crs, [x_absolute], [y_absolute])
                    x_transformed, y_transformed = xs_transformed[0], ys_transformed[0]
                except Exception as e:
                    logger.warning(f"Failed to transform point coordinates for {label}: {e}")
                    points_transform_failed += 1
                    continue
            else:
                x_transformed, y_transformed = x_absolute, y_absolute

            # Skip points that are out of bounds
            if not (min_x <= x_transformed <= max_x and min_y <= y_transformed <= max_y):
                points_out_of_bounds += 1
                logger.info(f"Point out of bounds: geojson=({x:.2f}, {y:.2f}), absolute=({x_absolute:.2f}, {y_absolute:.2f}), transformed=({x_transformed:.2f}, {y_transformed:.2f}), bounds: [{min_x:.2f}, {min_y:.2f}, {max_x:.2f}, {max_y:.2f}]")
                continue

            px, py = coords_to_pixel(x_transformed, y_transformed)
            # Draw point as a circle with radius 5 pixels
            radius = 5
            draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=color, outline=(255, 255, 255), width=2)
            points_drawn += 1
    
    if points_drawn == 0:
        logger.warning(f"No points drawn: {points_out_of_bounds} out of bounds, {points_transform_failed} transform failed, total points: {total_points}")

    # Resize image to fixed visualization size
    img = img.resize(VISUALIZATION_IMAGE_SIZE, Image.Resampling.LANCZOS)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    logger.info(f"Overlaid {points_drawn} points on image {output_path.name}")
    
    return points_drawn


def generate_mask_from_geojson(geojson_path, output_path, reference_tif_path, label_colors):
    """Generate a multi-class mask image from GeoJSON polygons.

    Args:
        geojson_path: Path to data.geojson file
        output_path: Path to save mask PNG
        reference_tif_path: Path to a reference GeoTIFF to get geographic bounds and size
        label_colors: Dictionary mapping label class names to RGB color tuples
    """
    with open(geojson_path, "r") as f:
        geojson_data = json.load(f)

    features = geojson_data.get("features", [])
    if not features:
        logger.warning(f"No features found in {geojson_path}")
        return

    geojson_crs = geojson_data.get("crs", {})
    src_crs = None
    
    if geojson_crs:
        crs_props = geojson_crs.get("properties", {})
        crs_name = crs_props.get("name", "")
        if "EPSG" in crs_name or "epsg" in crs_name:
            import re
            match = re.search(r"EPSG[:/](\d+)", crs_name, re.IGNORECASE)
            if match:
                src_crs = CRS.from_epsg(int(match.group(1)))
    
    if src_crs is None:
        props = geojson_data.get("properties", {})
        crs_str = props.get("crs", "")
        if crs_str and ("EPSG" in crs_str or "epsg" in crs_str):
            import re
            match = re.search(r"EPSG[:/](\d+)", crs_str, re.IGNORECASE)
            if match:
                src_crs = CRS.from_epsg(int(match.group(1)))
    
    if src_crs is None:
        src_crs = CRS.from_epsg(4326)

    polygons_by_label = defaultdict(list)

    for feature in features:
        geometry = feature.get("geometry", {})
        geom_type = geometry.get("type")
        props = feature.get("properties", {})
        label = props.get("label") or props.get("category") or props.get("class") or props.get("type") or "unknown"

        if geom_type == "Polygon":
            coordinates = geometry.get("coordinates", [[]])[0]
            if coordinates:
                polygons_by_label[label].append(coordinates)
        elif geom_type == "MultiPolygon":
            for polygon_coords in geometry.get("coordinates", []):
                outer_ring = polygon_coords[0]
                if outer_ring:
                    polygons_by_label[label].append(outer_ring)

    if not polygons_by_label:
        logger.warning(f"No valid polygon coordinates found in {geojson_path}")
        return
    
    logger.info(f"Found {len(polygons_by_label)} label types with polygons: {list(polygons_by_label.keys())}")

    try:
        with rasterio.open(reference_tif_path) as src:
            bounds = src.bounds
            width, height = src.width, src.height
            tif_crs = src.crs
            tif_transform = src.transform
            min_x, min_y, max_x, max_y = bounds.left, bounds.bottom, bounds.right, bounds.top
            logger.info(f"Reference GeoTIFF: {width}x{height}, CRS={tif_crs}, bounds=[{min_x:.2f}, {min_y:.2f}, {max_x:.2f}, {max_y:.2f}]")
    except Exception as e:
        logger.error(f"Could not read reference GeoTIFF: {e}")
        return
    
    logger.info(f"GeoJSON CRS: {src_crs}")

    # Determine background color - use background label color if available, otherwise black
    background_labels = {"no_landslide", "background", "unknown"}
    background_color = (0, 0, 0)  # Default to black
    for bg_label in background_labels:
        if bg_label in label_colors:
            background_color = label_colors[bg_label]
            break
    
    img = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(img)

    # Drawing order: background labels first, then no_data (buffer), then all other labels alphabetically
    # This ensures background labels are the base, no_data is the buffer, and other labels appear on top
    all_labels = sorted(polygons_by_label.keys())
    background_class_labels = [label for label in all_labels if label in background_labels]
    no_data_label = ["no_data"] if "no_data" in all_labels else []
    other_labels = [label for label in all_labels if label not in background_labels and label != "no_data"]
    
    draw_order = background_class_labels + no_data_label + sorted(other_labels)
    
    logger.info(f"Drawing order: {draw_order}")
    polygons_drawn = 0

    for label in draw_order:
        if label not in polygons_by_label:
            continue

        color = label_colors.get(label, (128, 128, 128))
        polygon_list = polygons_by_label[label]

        for polygon_coords in polygon_list:
            points = [(x, y) for x, y in polygon_coords]
            src_crs_for_transform = src_crs

            if src_crs_for_transform != tif_crs:
                try:
                    xs, ys = zip(*points)
                    xs_transformed, ys_transformed = transform(src_crs_for_transform, tif_crs, xs, ys)
                    points_transformed = list(zip(xs_transformed, ys_transformed))
                except Exception as e:
                    logger.warning(f"Failed to transform coordinates for {label}: {e}")
                    sample_x, sample_y = points[0]
                    if abs(sample_x) > 180 or abs(sample_y) > 90:
                        points_transformed = points
                    else:
                        continue
            else:
                points_transformed = points

            points_in_bounds = [
                (x, y) for x, y in points_transformed if min_x <= x <= max_x and min_y <= y <= max_y
            ]

            if not points_in_bounds:
                sample_coord = points_transformed[0] if points_transformed else None
                logger.warning(f"Polygon for {label} doesn't overlap bounds (sample coord: {sample_coord}, bounds: [{min_x:.2f}, {min_y:.2f}, {max_x:.2f}, {max_y:.2f}])")
                continue

            def coords_to_pixel(x, y):
                px = int((x - min_x) / (max_x - min_x) * width) if max_x != min_x else width // 2
                py = int((max_y - y) / (max_y - min_y) * height) if max_y != min_y else height // 2
                px = max(0, min(width - 1, px))
                py = max(0, min(height - 1, py))
                return (px, py)

            pixel_points = [coords_to_pixel(x, y) for x, y in points_transformed]
            draw.polygon(pixel_points, fill=color, outline=color)
            polygons_drawn += 1

    # Resize mask to fixed visualization size
    img = img.resize(VISUALIZATION_IMAGE_SIZE, Image.Resampling.NEAREST)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    if polygons_drawn == 0:
        logger.warning(f"No polygons drawn for mask {output_path.name} (had {len(polygons_by_label)} label types)")
    else:
        logger.info(f"Drew {polygons_drawn} polygons for mask {output_path.name}")


def find_geotiff_in_layer(layer_dir):
    """Find the first GeoTIFF file in a layer directory.

    Args:
        layer_dir: Path to layer directory

    Returns:
        Path to GeoTIFF file or None
    """
    layer_dir = Path(layer_dir)
    for item in layer_dir.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            geotiff_path = item / "geotiff.tif"
            if geotiff_path.exists():
                return geotiff_path
    return None


def process_layers(window_dir, layers, output_dir, normalization, bands, window_name):
    """Generic function to process layer images for any task type.
    
    Args:
        window_dir: Path to window directory
        layers: Dictionary of layer configs (excluding label)
        output_dir: Directory to save generated PNGs
        normalization: Dictionary mapping layer_name -> normalization_method
        bands: Dictionary mapping layer_name -> list of band indices (1-indexed)
        window_name: Name of the window (for logging)
    
    Returns:
        Dictionary mapping layer_name -> PNG path
    """
    layers_dir = window_dir / "layers"
    layer_images = {}
    
    for layer_name in layers.keys():
        layer_dir = layers_dir / layer_name
        if not layer_dir.exists():
            logger.debug(f"Layer {layer_name} directory not found for {window_name}")
            continue

        geotiff_path = find_geotiff_in_layer(layer_dir)
        if geotiff_path is None:
            logger.debug(f"No GeoTIFF found in {layer_name} for {window_name}")
            continue

        output_png = output_dir / window_name / f"{layer_name}.png"
        layer_config = layers.get(layer_name)
        normalize_method = normalization[layer_name]
        layer_bands = bands[layer_name] if bands and layer_name in bands else None
        visualize_tif(geotiff_path, output_png, bands=layer_bands, normalize_method=normalize_method, layer_config=layer_config)
        layer_images[layer_name] = output_png
        logger.debug(f"Generated image for {layer_name} in {window_name}")
    
    return layer_images


def process_classification_label(geojson_path, window_name):
    """Process classification label: extract text label from GeoJSON.
    
    Args:
        geojson_path: Path to data.geojson file
        window_name: Name of the window (for logging)
    
    Returns:
        Label text string, or None if not found
    """
    with open(geojson_path, "r") as f:
        geojson_data = json.load(f)
    features = geojson_data.get("features", [])
    if features:
        props = features[0].get("properties", {})
        label_text = props.get("label") or props.get("category") or props.get("class") or props.get("type") or "unknown"
        logger.info(f"Classification label for {window_name}: {label_text}")
        return label_text
    else:
        logger.warning(f"No features found in {geojson_path} for classification task")
        return None


def process_segmentation_label(geojson_path, reference_tif_path, output_path, label_colors, window_name):
    """Process segmentation label: generate mask from GeoJSON polygons.
    
    Args:
        geojson_path: Path to data.geojson file
        reference_tif_path: Path to reference GeoTIFF for coordinate transformation
        output_path: Path to save mask PNG
        label_colors: Dictionary mapping label class names to RGB color tuples
        window_name: Name of the window (for logging)
    
    Returns:
        Path to mask PNG, or None if generation failed
    """
    generate_mask_from_geojson(geojson_path, output_path, reference_tif_path, label_colors)
    return output_path if output_path.exists() else None


def process_detection_label(geojson_path, reference_tif_path, layer_images, output_dir, label_colors, metadata, window_name):
    """Process detection label: overlay points/bounding boxes on layer images.
    
    Args:
        geojson_path: Path to data.geojson file
        reference_tif_path: Path to reference GeoTIFF for coordinate transformation
        layer_images: Dictionary mapping layer_name -> PNG path
        output_dir: Directory to save overlaid images
        label_colors: Dictionary mapping label class names to RGB color tuples
        metadata: Window metadata dictionary (for bounds)
        window_name: Name of the window (for logging)
    
    Returns:
        Tuple of (updated layer_images dict, points_drawn count)
    """
    if not layer_images:
        logger.warning(f"No layer images available for detection overlay in {window_name}")
        return layer_images, 0
    
    first_layer_name = list(layer_images.keys())[0]
    first_layer_image = layer_images[first_layer_name]
    overlay_output = output_dir / window_name / f"{first_layer_name}_with_detections.png"
    
    # Get metadata bounds if available
    metadata_bounds = None
    if metadata and "bounds" in metadata:
        metadata_bounds = metadata["bounds"]
    
    points_drawn = overlay_points_on_image(first_layer_image, overlay_output, geojson_path, reference_tif_path, label_colors, metadata_bounds=metadata_bounds)
    
    # Replace the original image with the overlaid version
    layer_images[first_layer_name] = overlay_output
    logger.info(f"Overlaid detection points on {first_layer_name} for {window_name}")
    
    return layer_images, points_drawn


def process_regression_label(geojson_path, window_name):
    """Process regression label: extract regression value from GeoJSON.
    
    Args:
        geojson_path: Path to data.geojson file
        window_name: Name of the window (for logging)
    
    Returns:
        Regression value (float), or None if not found
    """
    # TODO: Implement regression label extraction when needed
    logger.warning(f"Regression label processing not yet implemented for {window_name}")
    return None


def process_window(window_dir, layers, output_dir, normalization=None, bands=None, label_colors=None, task_type=None):
    """Process a single window: generate PNGs for layers and task-specific label visualization.

    Args:
        window_dir: Path to window directory
        layers: Dictionary of layer configs (excluding label)
        output_dir: Directory to save generated PNGs
        normalization: Dictionary mapping layer_name -> normalization_method
        bands: Dictionary mapping layer_name -> list of band indices (1-indexed)
        label_colors: Dictionary mapping label class names to RGB color tuples
        task_type: Task type (classification, regression, detection, segmentation)

    Returns:
        Dictionary with layer names -> PNG paths, and task-specific label data
    """
    window_dir = Path(window_dir)
    window_name = window_dir.name
    layers_dir = window_dir / "layers"

    result = {
        "window_name": window_name,
        "layer_images": {},
        "mask_path": None,
        "metadata": None,
        "label_text": None,
        "points_drawn": 0,
    }

    # Load metadata
    metadata_path = window_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            result["metadata"] = json.load(f)

    # Process layers (common to all task types)
    result["layer_images"] = process_layers(window_dir, layers, output_dir, normalization, bands, window_name)

    # Process label based on task type
    label_dir = layers_dir / "label"
    if label_dir.exists():
        geojson_path = label_dir / "data.geojson"
        if geojson_path.exists():
            if task_type == "classification":
                result["label_text"] = process_classification_label(geojson_path, window_name)
            elif task_type == "segmentation":
                # Find reference GeoTIFF from layers
                reference_tif = None
                for layer_name in layers.keys():
                    layer_dir = layers_dir / layer_name
                    if layer_dir.exists():
                        ref_tif = find_geotiff_in_layer(layer_dir)
                        if ref_tif:
                            reference_tif = ref_tif
                            break
                if reference_tif and label_colors:
                    mask_output = output_dir / window_name / "label_mask.png"
                    result["mask_path"] = process_segmentation_label(geojson_path, reference_tif, mask_output, label_colors, window_name)
            elif task_type == "detection":
                # Find reference GeoTIFF from layers
                reference_tif = None
                for layer_name in layers.keys():
                    layer_dir = layers_dir / layer_name
                    if layer_dir.exists():
                        ref_tif = find_geotiff_in_layer(layer_dir)
                        if ref_tif:
                            reference_tif = ref_tif
                            break
                if reference_tif and label_colors and result["layer_images"]:
                    updated_images, points_drawn = process_detection_label(
                        geojson_path, reference_tif, result["layer_images"], output_dir, label_colors, result["metadata"], window_name
                    )
                    result["layer_images"] = updated_images
                    result["points_drawn"] = points_drawn
            elif task_type == "regression":
                result["label_text"] = process_regression_label(geojson_path, window_name)

    return result


def save_html_to_outputs(html_path, dataset_path, host="0.0.0.0", port=8000):
    """Save HTML file to outputs directory with dataset name and date.

    Args:
        html_path: Path to the HTML file to copy
        dataset_path: Path to dataset windows directory (used to extract dataset name)
        host: Server host (for log message)
        port: Server port (for log message)

    Returns:
        Path to the saved HTML file
    """
    html_path = Path(html_path)
    dataset_path_obj = Path(dataset_path)
    
    path_parts = dataset_path_obj.parts
    dataset_name_parts = []
    for i, part in enumerate(path_parts):
        if part == "windows" and i > 0:
            dataset_name_parts.append(path_parts[i - 1])
            if i > 1:
                dataset_name_parts.insert(0, path_parts[i - 2])
            break
    if not dataset_name_parts:
        dataset_name_parts = [dataset_path_obj.parent.name]
    dataset_name = "_".join(dataset_name_parts).replace("/", "_")
    
    date_str = datetime.now().strftime("%Y%m%d")
    
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    saved_html_path = outputs_dir / f"{dataset_name}_{date_str}.html"
    
    shutil.copy2(html_path, saved_html_path)
    logger.info(f"Saved HTML to {saved_html_path}")
    logger.info(f"Note: When opening the saved HTML directly, images may not load (they are in the temp directory)")
    logger.info(f"The server at http://{host}:{port} serves the complete visualization with images")
    
    return saved_html_path
