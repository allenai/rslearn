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

from rslearn.log_utils import get_logger

logger = get_logger(__name__)


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
        List of band indices (1-indexed) for RGB, or None if not determinable
    """
    band_sets = layer_config.get("band_sets", [])
    if not band_sets:
        return None
    
    band_set = band_sets[0]
    bands = band_set.get("bands", [])
    if not bands:
        return None
    
    rgb_band_names = ["B04", "B03", "B02"]
    rgb_indices = []
    for band_name in rgb_band_names:
        if band_name in bands:
            rgb_indices.append(bands.index(band_name) + 1)
    
    if len(rgb_indices) == 3:
        return rgb_indices
    return None


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
        
        if bands is None:
            if src.count >= 4:
                bands = [3, 2, 1]
            elif src.count >= 3:
                bands = [1, 2, 3]
            elif src.count == 1:
                bands = [1]
            else:
                raise ValueError(f"Unexpected number of bands: {src.count}")

        if len(bands) == 1:
            band_data = src.read(bands[0])
            band_normalized = normalize_band(band_data, method=normalize_method)
            img = Image.fromarray(band_normalized, mode="L")
        else:
            rgb = []
            for b in bands[:3]:
                band_data = src.read(b)
                rgb.append(normalize_band(band_data, method=normalize_method))
            rgb_arr = np.stack(rgb, axis=-1)
            img = Image.fromarray(rgb_arr, mode="RGB")

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
    
    Always assigns "no_data" to red. Other classes get distinct colors from palette.
    
    Args:
        label_classes: Set of label class names
        
    Returns:
        Dictionary mapping label class name to RGB color tuple
    """
    NO_DATA_COLOR = (255, 0, 0)
    
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

    background_labels = {"no_landslide", "background", "unknown"}
    background_color = (0, 0, 0)
    for bg_label in background_labels:
        if bg_label in label_colors:
            background_color = label_colors[bg_label]
            break
    
    img = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(img)

    draw_order = []
    background_classes = []
    buffer_classes = []
    important_classes = []
    
    buffer_labels = {"no_data"}
    important_labels = {"landslide"}
    
    for label in polygons_by_label.keys():
        if label in background_labels:
            background_classes.append(label)
        elif label in buffer_labels:
            buffer_classes.append(label)
        elif label in important_labels:
            important_classes.append(label)
        else:
            buffer_classes.append(label)
    
    draw_order = sorted(background_classes) + sorted(buffer_classes) + sorted(important_classes)
    
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


def process_window(window_dir, layers, output_dir, normalize_method="sentinel2_rgb", label_colors=None, is_classification_task=False):
    """Process a single window: generate PNGs for layers and mask.

    Args:
        window_dir: Path to window directory
        layers: Dictionary of layer configs (excluding label)
        output_dir: Directory to save generated PNGs
        normalize_method: Normalization method for raster visualization
        label_colors: Dictionary mapping label class names to RGB color tuples
        is_classification_task: If True, extract text label instead of generating mask

    Returns:
        Dictionary with layer names -> PNG paths, mask path (if segmentation), or label_text (if classification)
    """
    window_dir = Path(window_dir)
    window_name = window_dir.name
    layers_dir = window_dir / "layers"

    result = {
        "window_name": window_name,
        "layer_images": {},
        "mask_path": None,
        "metadata": None,
    }

    metadata_path = window_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            result["metadata"] = json.load(f)

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
        try:
            layer_config = layers.get(layer_name)
            visualize_tif(geotiff_path, output_png, normalize_method=normalize_method, layer_config=layer_config)
            result["layer_images"][layer_name] = output_png
            logger.debug(f"Generated image for {layer_name} in {window_name}")
        except Exception as e:
            logger.warning(f"Failed to visualize {layer_name} for {window_name}: {e}")

    label_dir = layers_dir / "label"
    if label_dir.exists():
        geojson_path = label_dir / "data.geojson"
        if geojson_path.exists():
            if is_classification_task:
                # For classification tasks, extract text label
                try:
                    with open(geojson_path, "r") as f:
                        geojson_data = json.load(f)
                    features = geojson_data.get("features", [])
                    if features:
                        props = features[0].get("properties", {})
                        label_text = props.get("label") or props.get("category") or props.get("class") or props.get("type") or "unknown"
                        result["label_text"] = label_text
                        logger.debug(f"Classification label for {window_name}: {label_text}")
                except Exception as e:
                    logger.warning(f"Failed to extract classification label for {window_name}: {e}")
            else:
                # For segmentation tasks, generate mask
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
                    try:
                        generate_mask_from_geojson(geojson_path, mask_output, reference_tif, label_colors)
                        result["mask_path"] = mask_output
                    except Exception as e:
                        logger.warning(f"Failed to generate mask for {window_name}: {e}")

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
