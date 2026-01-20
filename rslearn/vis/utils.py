"""Utility functions for rslearn dataset visualization."""

import json
import shutil
from collections import Counter, defaultdict
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


def detect_task_type_and_crs_from_labels(window_dirs, sample_count=10):
    """Detect task type and CRS from label GeoJSON files.
    
    Args:
        window_dirs: List of window directory paths
        sample_count: Number of windows to sample for detection
        
    Returns:
        Tuple of (task_type, label_crs, is_pixel_coords)
        task_type: "classification", "regression", "detection", or "segmentation"
        label_crs: CRS object or EPSG code, or None if not detected
        is_pixel_coords: Boolean indicating if coordinates are in pixel space
    """
    label_dir = Path(window_dirs[0]) / "layers" / "label"
    geojson_path = label_dir / "data.geojson"
    
    if not geojson_path.exists():
        logger.warning("No label GeoJSON found, cannot auto-detect task type")
        return None, None, False
    
    # Read a sample GeoJSON to detect structure
    with open(geojson_path, "r") as f:
        geojson_data = json.load(f)
    
    features = geojson_data.get("features", [])
    if not features:
        logger.warning("No features in label GeoJSON, cannot auto-detect task type")
        return None, None, False
    
    # Check coordinate mode: pixel coordinates have x_resolution/y_resolution in properties
    props = geojson_data.get("properties", {})
    is_pixel_coords = "x_resolution" in props and "y_resolution" in props
    
    # Detect CRS
    label_crs = None
    if is_pixel_coords:
        # Pixel mode: CRS is in properties.crs
        crs_str = props.get("crs", "")
        if crs_str and ("EPSG" in crs_str or "epsg" in crs_str):
            import re
            match = re.search(r"EPSG[:/](\d+)", crs_str, re.IGNORECASE)
            if match:
                label_crs = int(match.group(1))
    else:
        # Geographic mode: CRS is in crs.properties.name
        crs = geojson_data.get("crs", {})
        if crs:
            crs_props = crs.get("properties", {})
            crs_name = crs_props.get("name", "")
            if "EPSG" in crs_name or "epsg" in crs_name:
                import re
                match = re.search(r"EPSG[:/](\d+)", crs_name, re.IGNORECASE)
                if match:
                    label_crs = int(match.group(1))
    
    # Detect task type from geometry structure
    # Sample multiple windows to be more confident
    geom_type_counts = defaultdict(int)
    total_features = 0
    
    sample_windows = window_dirs[:min(sample_count, len(window_dirs))]
    for window_dir in sample_windows:
        label_dir = Path(window_dir) / "layers" / "label"
        geojson_path = label_dir / "data.geojson"
        if geojson_path.exists():
            try:
                with open(geojson_path, "r") as f:
                    data = json.load(f)
                features = data.get("features", [])
                total_features += len(features)
                for feat in features:
                    geom_type = feat.get("geometry", {}).get("type")
                    geom_type_counts[geom_type] += 1
            except Exception:
                continue
    
    if total_features == 0:
        logger.warning("No features found in sample windows")
        return None, label_crs, is_pixel_coords
    
    # Decision logic:
    # - Single feature per window → classification/regression (text display)
    # - Multiple Point features → detection (rasterize points)
    # - Multiple Polygon features → segmentation (rasterize polygons)
    avg_features_per_window = total_features / len(sample_windows)
    
    if avg_features_per_window <= 1.1:  # Allow small variation
        # Single feature: could be classification or regression
        # For now, default to classification (we could distinguish later if needed)
        task_type = "classification"
        logger.info(f"Detected task type: {task_type} (single feature per window)")
    elif "Point" in geom_type_counts and geom_type_counts["Point"] > geom_type_counts.get("Polygon", 0):
        task_type = "detection"
        logger.info(f"Detected task type: {task_type} (multiple Point features)")
    else:
        task_type = "segmentation"
        logger.info(f"Detected task type: {task_type} (multiple Polygon features)")
    
    return task_type, label_crs, is_pixel_coords


def detect_layers_bands_normalization_from_config(config):
    """Detect layers, bands, and normalization from config.json.
    
    Args:
        config: Config dictionary from config.json
        
    Returns:
        Tuple of (layers, bands, normalization)
        layers: List of layer names to visualize (excluding label)
        bands: Dictionary mapping layer_name -> list of band indices (1-indexed)
        normalization: Dictionary mapping layer_name -> normalization_method
    """
    layers_dict = config.get("layers", {})
    
    # Exclude label layer
    visualization_layers = {k: v for k, v in layers_dict.items() if k != "label"}
    
    layers = sorted(visualization_layers.keys())
    bands = {}
    normalization = {}
    
    # For each layer, detect bands and normalization
    for layer_name, layer_config in visualization_layers.items():
        layer_type = layer_config.get("type")
        
        if layer_type == "raster":
            # Detect RGB bands from config
            rgb_bands = get_rgb_bands_from_config(layer_config)
            if rgb_bands:
                bands[layer_name] = rgb_bands
            else:
                # Default: assume 3+ bands, use [3, 2, 1] for Sentinel-2
                bands[layer_name] = [3, 2, 1]
            
            # Detect normalization (default to sentinel2_rgb for now)
            # Could be enhanced to detect from layer name or config
            normalization[layer_name] = "sentinel2_rgb"
    
    return layers, bands, normalization


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
    """
    
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


def find_geotiff_in_layer(layer_dir):
    """Find GeoTIFF file in a layer directory.
    
    Args:
        layer_dir: Path to layer directory
        
    Returns:
        Path to GeoTIFF file, or None if not found
    """
    layer_dir = Path(layer_dir)
    for item in layer_dir.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            geotiff_path = item / "geotiff.tif"
            if geotiff_path.exists():
                return geotiff_path
    return None


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
    
    # Generate distinct colors for other classes
    # Use a color palette (excluding black, reserved for no_data)
    colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (255, 128, 0),    # Orange
        (128, 0, 255),    # Purple
        (255, 192, 203),  # Pink
        (128, 128, 0),    # Olive
        (0, 128, 128),    # Teal
        (128, 0, 0),      # Maroon
        (0, 128, 0),       # Dark Green
        (0, 0, 128),       # Navy
        (128, 128, 128),   # Gray
        (255, 165, 0),     # Orange
        (255, 20, 147),    # Deep Pink
        (50, 205, 50),     # Lime Green
        (255, 140, 0),     # Dark Orange
        (70, 130, 180),    # Steel Blue
    ]
    
    # Assign colors to classes
    color_idx = 0
    for label in label_classes:
        if label == "no_data":
            label_colors[label] = NO_DATA_COLOR
        else:
            label_colors[label] = colors[color_idx % len(colors)]
            color_idx += 1
    
    return label_colors


def overlay_points_on_image(image_path, output_path, geojson_path, reference_tif_path, label_colors, label_crs, metadata_bounds=None):
    """Overlay point geometries from GeoJSON onto an existing image.
    
    Args:
        image_path: Path to input PNG image
        output_path: Path to save output PNG with overlaid points
        geojson_path: Path to data.geojson file with Point geometries
        reference_tif_path: Path to reference GeoTIFF for coordinate transformation
        label_colors: Dictionary mapping label class names to RGB color tuples
        label_crs: CRS for GeoJSON coordinates (rasterio.crs.CRS object or EPSG code)
        metadata_bounds: Optional window bounds from metadata.json [min_x, min_y, max_x, max_y].
                        If provided, coordinates are treated as offsets from (min_x, min_y).
    """
    with open(geojson_path, "r") as f:
        geojson_data = json.load(f)

    features = geojson_data.get("features", [])
    if not features:
        logger.warning(f"No features found in {geojson_path}")
        return 0

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

    # Try to read CRS from GeoJSON properties first, fall back to label_crs from dataset configuration
    geojson_crs_str = geojson_data.get("properties", {}).get("crs", "")
    if geojson_crs_str and ("EPSG" in geojson_crs_str or "epsg" in geojson_crs_str):
        import re
        match = re.search(r"EPSG[:/](\d+)", geojson_crs_str, re.IGNORECASE)
        if match:
            src_crs = CRS.from_epsg(int(match.group(1)))
        else:
            src_crs = CRS.from_string(geojson_crs_str) if label_crs is None else CRS.from_epsg(label_crs) if isinstance(label_crs, int) else label_crs
    else:
        if label_crs is None:
            raise ValueError("label_crs must be provided when GeoJSON does not specify CRS")
        src_crs = CRS.from_epsg(label_crs) if isinstance(label_crs, int) else label_crs

    logger.info(f"GeoJSON CRS: {src_crs}, GeoTIFF CRS: {tif_crs}")

    points_drawn = 0
    for feature in features:
        geom = feature.get("geometry", {})
        geom_type = geom.get("type")
        
        if geom_type not in ("Point", "MultiPoint"):
            continue
        
        props = feature.get("properties", {})
        label = props.get("label") or props.get("category") or props.get("class") or props.get("type")
        if not label:
            continue
        
        color = label_colors.get(label, (255, 0, 0))  # Default to red if label not found
        
        # Get coordinates
        coords = geom.get("coordinates", [])
        if geom_type == "Point":
            coords_list = [coords]
        else:  # MultiPoint
            coords_list = coords
        
        for coord in coords_list:
            x, y = coord[0], coord[1]
            
            # Transform coordinates from GeoJSON CRS to GeoTIFF CRS
            try:
                xs, ys = transform(src_crs, tif_crs, [x], [y])
                x_transformed, y_transformed = xs[0], ys[0]
            except Exception as e:
                logger.warning(f"Coordinate transformation failed for ({x}, {y}): {e}")
                continue
            
            # Convert geographic coordinates to pixel coordinates using the transform
            # Use ~transform (inverse) to convert (x, y) geographic -> (col, row) pixel
            px, py = ~tif_transform * (x_transformed, y_transformed)
            px = int(round(px))
            py = int(round(py))
            
            # Skip if out of bounds
            if px < 0 or px >= width or py < 0 or py >= height:
                continue
            
            # Draw point as circle
            radius = 5
            draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=color, outline=color)
            points_drawn += 1
            logger.info(f"Point mapped to 512x512 image: pixel=({px}, {py}), geographic=({x_transformed:.2f}, {y_transformed:.2f}), label={label}")

    # Resize image to fixed visualization size
    img = img.resize(VISUALIZATION_IMAGE_SIZE, Image.Resampling.LANCZOS)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    logger.info(f"Overlaid {points_drawn} points on image")
    
    return points_drawn


def generate_mask_from_geojson(geojson_path, output_path, reference_tif_path, label_colors, label_crs):
    """Generate mask image from GeoJSON polygons.
    
    Args:
        geojson_path: Path to data.geojson file
        output_path: Path to save mask PNG
        reference_tif_path: Path to reference GeoTIFF for coordinate transformation
        label_colors: Dictionary mapping label class names to RGB color tuples
        label_crs: CRS for GeoJSON coordinates (rasterio.crs.CRS object or EPSG code)
    """
    if label_crs is None:
        raise ValueError("label_crs must be provided")
    
    with open(geojson_path, "r") as f:
        geojson_data = json.load(f)

    features = geojson_data.get("features", [])
    if not features:
        logger.warning(f"No features found in {geojson_path}")
        return

    # Get reference GeoTIFF bounds and transform
    with rasterio.open(reference_tif_path) as src:
        bounds = src.bounds
        tif_crs = src.crs
        tif_transform = src.transform
        width = src.width
        height = src.height
    
    min_x, min_y, max_x, max_y = bounds.left, bounds.bottom, bounds.right, bounds.top
    
    # Try to read CRS from GeoJSON properties first, fall back to label_crs from dataset configuration
    geojson_crs_str = geojson_data.get("properties", {}).get("crs", "")
    if geojson_crs_str and ("EPSG" in geojson_crs_str or "epsg" in geojson_crs_str):
        import re
        match = re.search(r"EPSG[:/](\d+)", geojson_crs_str, re.IGNORECASE)
        if match:
            src_crs = CRS.from_epsg(int(match.group(1)))
        else:
            src_crs = CRS.from_epsg(label_crs) if isinstance(label_crs, int) else label_crs
    else:
        src_crs = CRS.from_epsg(label_crs) if isinstance(label_crs, int) else label_crs

    # Determine background color from the most common label in the data
    # (excluding "no_data" which should only appear as buffers)
    label_counts = Counter()
    for feat in features:
        props = feat.get("properties", {})
        label = props.get("label") or props.get("category") or props.get("class") or props.get("type")
        if label and label != "no_data":
            label_counts[label] += 1
    
    # Use the most common label as background, or black if no labels found
    background_color = (0, 0, 0)  # Default to black
    if label_counts:
        most_common_label = label_counts.most_common(1)[0][0]
        background_color = label_colors.get(most_common_label, (0, 0, 0))
    
    # Initialize mask with background color
    mask_img = Image.new("RGB", (width, height), color=background_color)
    draw = ImageDraw.Draw(mask_img)

    # Sort features: "no_data" first (buffers), then others alphabetically
    # This ensures buffers are drawn first, then other labels overwrite them
    def sort_key(feat):
        props = feat.get("properties", {})
        label = props.get("label") or props.get("category") or props.get("class") or props.get("type") or ""
        if label == "no_data":
            return (0, label)  # no_data drawn first
        else:
            return (1, label)  # Other labels drawn after, alphabetically
    
    sorted_features = sorted(features, key=sort_key)
    
    # Draw polygons
    for feature in sorted_features:
        geom = feature.get("geometry", {})
        geom_type = geom.get("type")
        
        if geom_type not in ("Polygon", "MultiPolygon"):
            continue
        
        props = feature.get("properties", {})
        label = props.get("label") or props.get("category") or props.get("class") or props.get("type")
        if not label:
            continue
        
        color = label_colors.get(label, (255, 255, 255))  # Default to white if label not found
        
        coords = geom.get("coordinates", [])
        if geom_type == "Polygon":
            polygons = [coords]
        else:  # MultiPolygon
            polygons = coords
        
        for polygon_coords in polygons:
            # Transform coordinates
            polygon_pixels = []
            for ring in polygon_coords:
                ring_pixels = []
                for coord in ring:
                    x, y = coord[0], coord[1]
                    try:
                        xs, ys = transform(src_crs, tif_crs, [x], [y])
                        x_transformed, y_transformed = xs[0], ys[0]
                        
                        # Convert geographic coordinates to pixel coordinates using the transform
                        # Use ~transform (inverse) to convert (x, y) geographic -> (col, row) pixel
                        px, py = ~tif_transform * (x_transformed, y_transformed)
                        px = int(round(px))
                        py = int(round(py))
                        
                        # Skip if out of bounds
                        if px < 0 or px >= width or py < 0 or py >= height:
                            continue
                            
                        ring_pixels.append((px, py))
                    except Exception as e:
                        logger.warning(f"Coordinate transformation failed for ({x}, {y}): {e}")
                        continue
                
                if len(ring_pixels) >= 3:  # Need at least 3 points for a polygon
                    polygon_pixels.append(ring_pixels)
            
            if polygon_pixels:
                # Draw outer ring
                if polygon_pixels[0]:
                    draw.polygon(polygon_pixels[0], fill=color, outline=color)
                    # Draw holes if present (fill with background color - black)
                    for hole in polygon_pixels[1:]:
                        if hole:
                            draw.polygon(hole, fill=(0, 0, 0), outline=color)
    
    # Resize mask to fixed visualization size
    mask_img = mask_img.resize(VISUALIZATION_IMAGE_SIZE, Image.Resampling.NEAREST)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_img.save(output_path)


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
    if not features:
        logger.warning(f"No features in classification label for {window_name}")
        return None
    
    # Get label from first feature (as per boss's suggestion: "first feature")
    first_feature = features[0]
    props = first_feature.get("properties", {})
    
    # Try common property names
    label = props.get("label") or props.get("category") or props.get("class") or props.get("type")
    
    if label:
        logger.info(f"Classification label for {window_name}: {label}")
        return str(label)
    
    logger.warning(f"No label property found in classification GeoJSON for {window_name}")
    return None


def process_segmentation_label(geojson_path, reference_tif_path, output_path, label_colors, label_crs, window_name):
    """Process segmentation label: generate mask from GeoJSON polygons.
    
    Args:
        geojson_path: Path to data.geojson file
        reference_tif_path: Path to reference GeoTIFF for coordinate transformation
        output_path: Path to save mask PNG
        label_colors: Dictionary mapping label class names to RGB color tuples
        label_crs: CRS for GeoJSON coordinates (rasterio.crs.CRS object or EPSG code)
        window_name: Name of the window (for logging)
    
    Returns:
        Path to mask PNG, or None if generation failed
    """
    generate_mask_from_geojson(geojson_path, output_path, reference_tif_path, label_colors, label_crs)
    return output_path if output_path.exists() else None


def process_detection_label(geojson_path, reference_tif_path, layer_images, output_dir, label_colors, label_crs, metadata, window_name):
    """Process detection label: overlay points/bounding boxes on layer images.
    
    Args:
        geojson_path: Path to data.geojson file
        reference_tif_path: Path to reference GeoTIFF for coordinate transformation
        layer_images: Dictionary mapping layer_name -> PNG path
        output_dir: Directory to save overlaid images
        label_colors: Dictionary mapping label class names to RGB color tuples
        label_crs: CRS for GeoJSON coordinates (rasterio.crs.CRS object or EPSG code)
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
    
    # Coordinates are always absolute, metadata_bounds not needed
    points_drawn = overlay_points_on_image(first_layer_image, overlay_output, geojson_path, reference_tif_path, label_colors, label_crs, metadata_bounds=None)
    
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


def process_layers(window_dir, layers, output_dir, normalization, bands, window_name):
    """Process raster layers: generate PNG images for each layer.
    
    Args:
        window_dir: Path to window directory
        layers: Dictionary of layer configs
        output_dir: Directory to save PNGs
        normalization: Dictionary mapping layer_name -> normalization_method
        bands: Dictionary mapping layer_name -> list of band indices (1-indexed)
        window_name: Name of the window (for logging)
    
    Returns:
        Dictionary mapping layer_name -> PNG path
    """
    window_dir = Path(window_dir)
    layers_dir = window_dir / "layers"
    layer_images = {}
    
    for layer_name, layer_config in layers.items():
        layer_dir = layers_dir / layer_name
        if not layer_dir.exists():
            continue
        
        geotiff_path = find_geotiff_in_layer(layer_dir)
        if not geotiff_path:
            logger.debug(f"No GeoTIFF found for layer {layer_name} in {window_name}")
            continue
        
        output_png = output_dir / window_name / f"{layer_name}.png"
        normalize_method = normalization.get(layer_name, "sentinel2_rgb")
        layer_bands = bands.get(layer_name)
        
        visualize_tif(geotiff_path, output_png, bands=layer_bands, normalize_method=normalize_method, layer_config=layer_config)
        layer_images[layer_name] = output_png
    
    return layer_images


def process_window(window_dir, layers, output_dir, normalization=None, bands=None, label_colors=None, task_type=None, label_crs=None):
    """Process a single window: generate PNGs for layers and task-specific label visualization.

    Args:
        window_dir: Path to window directory
        layers: Dictionary of layer configs (excluding label)
        output_dir: Directory to save generated PNGs
        normalization: Dictionary mapping layer_name -> normalization_method
        bands: Dictionary mapping layer_name -> list of band indices (1-indexed)
        label_colors: Dictionary mapping label class names to RGB color tuples
        task_type: Task type (classification, regression, detection, segmentation)
        label_crs: CRS for GeoJSON coordinates (rasterio.crs.CRS object or EPSG code)

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
                    result["mask_path"] = process_segmentation_label(geojson_path, reference_tif, mask_output, label_colors, label_crs, window_name)
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
                        geojson_path, reference_tif, result["layer_images"], output_dir, label_colors, label_crs, result["metadata"], window_name
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
