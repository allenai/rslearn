"""
Visualization server for rslearn datasets.

This module provides a web server to visualize rslearn datasets using the Dataset/Window APIs.
"""

import argparse
import atexit
import http.server
import importlib.resources
import random
import shutil
import socketserver
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from jinja2 import Environment, FileSystemLoader
from upath import UPath

from rslearn.config import LayerConfig, LayerType
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.log_utils import get_logger

from .labels import generate_label_colors, get_vector_label_by_property
from .layers import read_raster_layer, read_vector_layer
from .vis import VISUALIZATION_IMAGE_SIZE, array_to_png, features_to_mask, raster_label_to_mask

logger = get_logger(__name__)


class VisualizationServer:
    """Visualization server for rslearn datasets using Dataset/Window APIs."""

    def __init__(self):
        """Initialize the visualization server."""
        self.windows: list[Window] = []
        self.dataset: Dataset | None = None
        self.layers: list[str] = []
        self.bands: dict[str, list[str]] = {}
        self.normalization: dict[str, str] = {}
        self.task_type: str = ""
        self.label_colors_dict: dict[str, dict[str, tuple[int, int, int]]] = {}
        self.group_idx: int = 0

    def format_window_info(self, window: Window) -> tuple[str, float | None, float | None]:
        """Format window metadata for display.

        Args:
            window: Window object

        Returns:
            Tuple of (formatted info HTML, lat, lon) for Google Maps link
        """
        parts = []
        lat = None
        lon = None

        if window.time_range:
            start = window.time_range[0].isoformat()[:10]
            end = window.time_range[1].isoformat()[:10]
            parts.append(f"Time: {start} to {end}")

        geom_wgs84 = window.get_geometry().to_projection(WGS84_PROJECTION)
        centroid = geom_wgs84.shp.centroid
        lon = float(centroid.x)
        lat = float(centroid.y)
        parts.insert(0, f"Lat: {lat:.4f}, Lon: {lon:.4f}")

        return "<br>".join(parts) if parts else "Unknown", lat, lon

    def _generate_image_as_bytes(
        self,
        window: Window,
        layer_name: str,
        dataset: Dataset,
        bands: dict[str, list[str]],
        normalization: dict[str, str],
        task_type: str,
        label_colors: dict[str, tuple[int, int, int]] | None,
        label_colors_dict: dict[str, dict[str, tuple[int, int, int]]] | None = None,
        group_idx: int = 0,
    ) -> bytes:
        """Generate an image for a window/layer combination as PNG bytes.

        Args:
            window: Window object
            layer_name: Layer name to visualize
            dataset: Dataset object
            bands: Dictionary mapping layer_name -> list of band names
            normalization: Dictionary mapping layer_name -> normalization method
            task_type: Task type
            label_colors: Dictionary mapping label class names to RGB colors
            label_colors_dict: Dictionary mapping layer_name -> label_colors
            group_idx: Item group index

        Returns:
            PNG image bytes
        """
        from PIL import Image

        # Helper to check if a layer is a label layer
        def is_label_layer(name: str) -> bool:
            return (
                name in ("label", "labels", "label_raster") or
                name.endswith("_label") or
                name.endswith("_label_raster") or
                name.startswith("label_")
            )

        if layer_name not in dataset.layers:
            layer_dir = window.get_layer_dir(layer_name, group_idx=group_idx)
            layer_dir_path = UPath(layer_dir)
            data_geojson = layer_dir_path / "data.geojson"
            if data_geojson.exists():
                layer_config = LayerConfig(type=LayerType.VECTOR)
            else:
                raise ValueError(f"Layer {layer_name} not in config and cannot infer type")
        else:
            layer_config = dataset.layers[layer_name]

        if layer_config.type == LayerType.RASTER and not is_label_layer(layer_name):
            if layer_name not in bands:
                raise ValueError(f"Bands not specified for layer {layer_name}")
            if layer_name not in normalization:
                raise ValueError(f"Normalization not specified for layer {layer_name}")
            
            array = read_raster_layer(window, layer_name, layer_config, bands[layer_name], group_idx=group_idx)
            from .normalization import normalize_array
            normalized = normalize_array(array, normalization[layer_name])
            
            if normalized.shape[-1] == 1:
                img = Image.fromarray(normalized[:, :, 0], mode="L")
            elif normalized.shape[-1] == 3:
                img = Image.fromarray(normalized, mode="RGB")
            else:
                img = Image.fromarray(normalized[:, :, :3], mode="RGB")
            
            img = img.resize(VISUALIZATION_IMAGE_SIZE, Image.Resampling.LANCZOS)
            
        elif is_label_layer(layer_name):
            if layer_config.type == LayerType.RASTER:
                if not layer_config.band_sets:
                    raise ValueError(f"No band sets in raster label layer {layer_name}")
                band_set = layer_config.band_sets[0]
                if not band_set.bands:
                    raise ValueError(f"No bands in raster label layer {layer_name}")
                
                label_array = read_raster_layer(window, layer_name, layer_config, [band_set.bands[0]], group_idx=group_idx)
                layer_label_colors = label_colors_dict.get(layer_name) if label_colors_dict else label_colors
                if not layer_label_colors:
                    raise ValueError(f"No label colors available for layer {layer_name}")
                
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
                        color = layer_label_colors.get(str(class_name), (0, 0, 0))
                        mask = (label_int == idx) & valid_mask
                        mask_img[mask] = color
                else:
                    unique_vals = np.unique(label_values[valid_mask])
                    for val in unique_vals:
                        val_str = "no_data" if np.isclose(val, 0) else str(int(val) if np.isclose(val, int(val)) else float(val))
                        color = layer_label_colors.get(val_str, (0, 0, 0))
                        mask = (label_values == val) & valid_mask
                        mask_img[mask] = color
                
                img = Image.fromarray(mask_img, mode="RGB")
                img = img.resize(VISUALIZATION_IMAGE_SIZE, Image.Resampling.NEAREST)
                
            elif layer_config.type == LayerType.VECTOR:
                if task_type == "classification":
                    raise ValueError("Classification labels are text, not images")
                
                features = read_vector_layer(window, layer_name, layer_config, group_idx=group_idx)
                if not features:
                    logger.info(f"No features found in vector label layer {layer_name} for window {window.name}")
                    raise FileNotFoundError(f"No features in vector label layer {layer_name}")
                
                reference_array = None
                ref_normalization_method = None
                if task_type == "detection":
                    raster_layers = [name for name in bands.keys() if not is_label_layer(name)]
                    if raster_layers:
                        ref_layer_name = raster_layers[0]
                        ref_layer_config = dataset.layers[ref_layer_name]
                        reference_array = read_raster_layer(
                            window, ref_layer_name, ref_layer_config, bands[ref_layer_name], group_idx=group_idx
                        )
                        ref_normalization_method = normalization[ref_layer_name]
                
                width = window.bounds[2] - window.bounds[0]
                height = window.bounds[3] - window.bounds[1]
                
                if reference_array is not None:
                    from .normalization import normalize_array
                    from .overlay import overlay_points_on_image
                    from PIL import ImageDraw
                    
                    normalized = normalize_array(reference_array, ref_normalization_method)
                    if normalized.shape[-1] >= 3:
                        img = Image.fromarray(normalized[:, :, :3], mode="RGB")
                    else:
                        img = Image.fromarray(normalized[:, :, 0], mode="L").convert("RGB")
                    
                    draw = ImageDraw.Draw(img)
                    actual_width = width
                    actual_height = height
                    overlay_points_on_image(
                        draw, features, window.bounds, img.size[0], img.size[1], actual_width, actual_height, label_colors
                    )
                    img = img.resize(VISUALIZATION_IMAGE_SIZE, Image.Resampling.LANCZOS)
                else:
                    from PIL import ImageDraw
                    mask_img = Image.new("RGB", (width, height), color=(0, 0, 0))
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
                    
                    def get_label(feat) -> str:
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
                        label = None
                        for prop_name in property_names:
                            label = feature.properties.get(prop_name)
                            if label:
                                label = str(label)
                                break
                        if not label:
                            continue
                        color = label_colors.get(label, (255, 255, 255))
                        geom_pixel = feature.geometry.to_projection(window.projection)
                        shp = geom_pixel.shp
                        if shp.geom_type == "Polygon":
                            coords = list(shp.exterior.coords)
                            pixel_coords = [(int(x - window.bounds[0]), int(y - window.bounds[1])) for x, y in coords]
                            if len(pixel_coords) >= 3:
                                draw.polygon(pixel_coords, fill=color, outline=color)
                                for interior in shp.interiors:
                                    hole_coords = [(int(x - window.bounds[0]), int(y - window.bounds[1])) for x, y in interior.coords]
                                    if len(hole_coords) >= 3:
                                        draw.polygon(hole_coords, fill=(0, 0, 0), outline=color)
                        elif shp.geom_type == "MultiPolygon":
                            for poly in shp.geoms:
                                coords = list(poly.exterior.coords)
                                pixel_coords = [(int(x - window.bounds[0]), int(y - window.bounds[1])) for x, y in coords]
                                if len(pixel_coords) >= 3:
                                    draw.polygon(pixel_coords, fill=color, outline=color)
                    
                    img = mask_img.resize(VISUALIZATION_IMAGE_SIZE, Image.Resampling.NEAREST)
        else:
            raise ValueError(f"Unsupported layer type for {layer_name}")

        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def process_window(
        self,
        window: Window,
        dataset: Dataset,
        layer_names: list[str],
        bands: dict[str, list[str]],
        normalization: dict[str, str],
        output_dir: Path,
        task_type: str,
        label_colors: dict[str, tuple[int, int, int]] | None,
        label_colors_dict: dict[str, dict[str, tuple[int, int, int]]] | None = None,
        group_idx: int = 0,
    ) -> dict[str, Any]:
        """Process a single window and generate visualizations.

        Args:
            window: Window object
            dataset: Dataset object
            layer_names: List of layer names to visualize
            bands: Dictionary mapping layer_name -> list of band names
            normalization: Dictionary mapping layer_name -> normalization method
            output_dir: Directory to save output images
            task_type: Task type (classification, regression, detection, segmentation)
            label_colors: Dictionary mapping label class names to RGB colors (for segmentation/detection)
            group_idx: Item group index (default 0)

        Returns:
            Dictionary with window processing results
        """
        window_dir = output_dir / window.name
        window_dir.mkdir(parents=True, exist_ok=True)

        result = {
            "window_name": window.name,
            "window": window,
            "layer_images": {},
            "mask_path": None,
            "label_text": None,
        }

        # Helper to check if a layer is a label layer
        def is_label_layer(name: str) -> bool:
            return (
                name in ("label", "labels", "label_raster") or
                name.endswith("_label") or
                name.endswith("_label_raster") or
                name.startswith("label_")
            )

        # Process all layers (both raster image layers and label layers)
        for layer_name in layer_names:
            if layer_name not in dataset.layers:
                layer_dir = window.get_layer_dir(layer_name, group_idx=group_idx)
                layer_dir_path = UPath(layer_dir)
                data_geojson = layer_dir_path / "data.geojson"
                
                if data_geojson.exists():
                    layer_config = LayerConfig(type=LayerType.VECTOR)
                    logger.debug(f"Creating minimal vector LayerConfig for {layer_name} (not in config.json)")
                else:
                    logger.warning(f"Layer {layer_name} not in config.json and cannot infer type - skipping")
                    continue
            else:
                layer_config = dataset.layers[layer_name]
            
            if not window.is_layer_completed(layer_name, group_idx=group_idx):
                logger.debug(f"Layer {layer_name} not marked as completed for window {window.name}, attempting to read anyway")

            if layer_config.type == LayerType.RASTER and not is_label_layer(layer_name):
                try:
                    if layer_name not in bands:
                        raise ValueError(f"Bands not specified for layer {layer_name}. Please provide --bands {layer_name}:band1,band2,band3")
                    band_names = bands[layer_name]
                    array = read_raster_layer(
                        window, layer_name, layer_config, band_names, group_idx=group_idx
                    )

                    if layer_name not in normalization:
                        raise ValueError(f"Normalization not specified for layer {layer_name}. Please provide --normalization {layer_name}:method")
                    norm_method = normalization[layer_name]
                    img_path = window_dir / f"{layer_name}.png"
                    array_to_png(array, img_path, norm_method)

                    result["layer_images"][layer_name] = img_path

                except Exception as e:
                    logger.error(f"Failed to process layer {layer_name} for window {window.name}: {e}")
                    continue
            
            elif is_label_layer(layer_name):
                try:
                    if layer_config.type == LayerType.RASTER:
                        if not layer_config.band_sets:
                            logger.debug(f"No band sets in raster label layer {layer_name}")
                            continue
                        band_set = layer_config.band_sets[0]
                        if not band_set.bands:
                            logger.debug(f"No bands in raster label layer {layer_name}")
                            continue
                        
                        band_name = band_set.bands[0]
                        label_array = read_raster_layer(
                            window, layer_name, layer_config, [band_name], group_idx=group_idx
                        )
                        
                        mask_path = window_dir / f"{layer_name}.png"
                        layer_label_colors = None
                        if label_colors_dict and layer_name in label_colors_dict:
                            layer_label_colors = label_colors_dict[layer_name]
                        elif label_colors:
                            layer_label_colors = label_colors
                        
                        if layer_label_colors:
                            raster_label_to_mask(label_array, layer_label_colors, layer_config, mask_path)
                            result["layer_images"][layer_name] = mask_path
                        else:
                            logger.warning(f"No label colors available for raster label layer {layer_name}")
                        
                    elif layer_config.type == LayerType.VECTOR:
                        if task_type == "classification":
                            if result["label_text"] is None:
                                label_text = get_vector_label_by_property(window, layer_config, layer_name, group_idx=group_idx)
                                result["label_text"] = label_text
                        elif task_type in ("segmentation", "detection") and label_colors:
                            features = read_vector_layer(window, layer_name, layer_config, group_idx=group_idx)
                            logger.info(f"Window {window.name}: Found {len(features)} label features in {layer_name}")
                            if features:
                                mask_path = window_dir / f"{layer_name}.png"

                                reference_array = None
                                ref_normalization_method = None
                                if task_type == "detection" and result["layer_images"]:
                                    raster_layers = [name for name in layer_names if not is_label_layer(name) and name in result["layer_images"]]
                                    if raster_layers:
                                        ref_layer_name = raster_layers[0]
                                        ref_layer_config = dataset.layers[ref_layer_name]
                                        if ref_layer_name not in bands:
                                            raise ValueError(f"Bands not specified for reference layer {ref_layer_name}. Please provide --bands {ref_layer_name}:band1,band2,band3")
                                        ref_band_names = bands[ref_layer_name]
                                        reference_array = read_raster_layer(
                                            window,
                                            ref_layer_name,
                                            ref_layer_config,
                                            ref_band_names,
                                            group_idx=group_idx,
                                        )
                                        logger.info(f"Window {window.name}: Read reference array with shape {reference_array.shape}, bounds: {window.bounds}")
                                        
                                        if ref_layer_name not in normalization:
                                            raise ValueError(f"Normalization not specified for reference layer {ref_layer_name}. Please provide --normalization {ref_layer_name}:method")
                                        ref_normalization_method = normalization[ref_layer_name]

                                if reference_array is not None:
                                    features_to_mask(
                                        features,
                                        window.bounds,
                                        window.projection,
                                        label_colors,
                                        mask_path,
                                        reference_raster_array=reference_array,
                                        normalization_method=ref_normalization_method,
                                    )
                                else:
                                    features_to_mask(
                                        features,
                                        window.bounds,
                                        window.projection,
                                        label_colors,
                                        mask_path,
                                        reference_raster_array=None,
                                    )
                                result["layer_images"][layer_name] = mask_path
                except Exception as e:
                    logger.error(f"Failed to process label layer {layer_name} for window {window.name}: {e}")
                    continue

        return result

    def _render_template(self, sampled_windows: list[Window]) -> str:
        """Render the HTML template with window data.

        Args:
            sampled_windows: List of windows to display

        Returns:
            Rendered HTML as string
        """
        template_dir = Path(__file__).parent / "templates"
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        template = env.get_template("viewer.html")

        window_data = []
        for idx, window in enumerate(sampled_windows):
            info_html, lat, lon = self.format_window_info(window)
            maps_link = f"https://www.google.com/maps?q={lat},{lon}" if lat is not None and lon is not None else None

            available_layers = set()
            mask_layer = None
            label_text = None

            def is_label_layer(name: str) -> bool:
                return (
                    name in ("label", "labels", "label_raster") or
                    name.endswith("_label") or
                    name.endswith("_label_raster") or
                    name.startswith("label_")
                )

            for layer_name in self.layers:
                if layer_name not in self.dataset.layers:
                    continue
                layer_config = self.dataset.layers[layer_name]
                try:
                    if layer_config.type == LayerType.RASTER and not is_label_layer(layer_name):
                        if window.is_layer_completed(layer_name, group_idx=self.group_idx):
                            available_layers.add(layer_name)
                    elif is_label_layer(layer_name):
                        if layer_config.type == LayerType.VECTOR:
                            if self.task_type == "classification":
                                if window.is_layer_completed(layer_name, group_idx=self.group_idx):
                                    label_text = get_vector_label_by_property(window, layer_config, layer_name, group_idx=self.group_idx)
                            else:
                                if window.is_layer_completed(layer_name, group_idx=self.group_idx):
                                    try:
                                        features = read_vector_layer(window, layer_name, layer_config, group_idx=self.group_idx)
                                        if features:
                                            mask_layer = layer_name
                                    except Exception:
                                        pass
                        elif layer_config.type == LayerType.RASTER:
                            if window.is_layer_completed(layer_name, group_idx=self.group_idx):
                                mask_layer = layer_name
                except Exception:
                    continue

            window_data.append({
                "idx": idx,
                "name": window.name,
                "info_html": info_html,
                "maps_link": maps_link,
                "available_layers": available_layers,
                "mask_layer": mask_layer,
                "label_text": label_text,
            })

        label_colors = None
        if self.label_colors_dict:
            first_label_layer = list(self.label_colors_dict.keys())[0]
            label_colors = self.label_colors_dict[first_label_layer]

        # Render template
        html = template.render(
            windows=window_data,
            layer_names=self.layers,
            label_colors=label_colors,
            task_type=self.task_type,
        )
        return html

    def run(
        self,
        dataset_path: str | Path | UPath,
        layers: list[str] | None = None,
        bands: dict[str, list[str]] | None = None,
        normalization: dict[str, str] | None = None,
        task_type: str | None = None,
        max_samples: int = 100,
        port: int = 8000,
        host: str = "0.0.0.0",
        save_html: bool = False,
        group_idx: int = 0,
    ) -> None:
        """Run the visualization server.

        Args:
            dataset_path: Path to dataset directory (containing config.json)
            layers: List of layer names to visualize (if None, uses all raster layers except "label")
            bands: Dictionary mapping layer_name -> list of band names (if None, uses first 3 bands)
            normalization: Dictionary mapping layer_name -> normalization method (if None, uses defaults)
            task_type: Task type - "classification", "regression", "detection", or "segmentation" (if None, auto-detects)
            max_samples: Maximum number of windows to sample
            port: Port to serve on
            host: Host to bind to
            save_html: Whether to save HTML file to outputs directory
            group_idx: Item group index (default 0)
        """
        dataset_path = UPath(dataset_path)
        dataset = Dataset(dataset_path)

        # Determine non-label layers to visualize
        if layers is None:
            layers = [
                name
                for name, config in dataset.layers.items()
                if config.type == LayerType.RASTER
                and not (name in ("label", "labels", "label_raster") or name.endswith("_label") or name.endswith("_label_raster"))
            ]
            layers = sorted(layers)

        for layer_name in layers:
            if layer_name not in dataset.layers:
                logger.warning(
                    f"Layer {layer_name} not found in dataset config.json. "
                    f"Will attempt to visualize if it exists in window directories."
                )

        def is_label_layer(name: str) -> bool:
            return (
                name in ("label", "labels", "label_raster") or
                name.endswith("_label") or
                name.endswith("_label_raster") or
                name.startswith("label_")
            )
        
        # Separate raster image layers from label layers
        raster_image_layers = [name for name in layers if not is_label_layer(name)]
        label_layers_in_list = [name for name in layers if is_label_layer(name)]
        
        bands = bands or {}
        normalization = normalization or {}
        for layer_name in raster_image_layers:
            if layer_name not in bands:
                raise ValueError(f"Bands not specified for layer {layer_name}. Please provide --bands {layer_name}:band1,band2,band3")
            if layer_name not in normalization:
                raise ValueError(f"Normalization not specified for layer {layer_name}. Please provide --normalization {layer_name}:method")

        if task_type is None:
            raise ValueError("--task_type must be specified. Choose from: classification, regression, detection, segmentation")

        # Get label colors for segmentation/detection
        # Get classes from config for each label layer individually
        label_colors_dict = {}  # Map layer_name -> label_colors
        if task_type in ("segmentation", "detection"):
            for label_layer_name in label_layers_in_list:
                label_config = dataset.layers[label_layer_name]
                if not label_config.class_names:
                    # For detection tasks, use default 'detected' class if no classes found
                    if task_type == "detection":
                        label_classes = {"detected"}
                        label_colors = generate_label_colors(label_classes)
                        label_colors_dict[label_layer_name] = label_colors
                        logger.info(f"No label classes in config for {label_layer_name} (detection task) - using default 'detected' class")
                    else:
                        raise ValueError(
                            f"class_names must be specified in the config for label layer '{label_layer_name}'. "
                            "Auto-detection of class names is not supported."
                        )
                else:
                    label_classes = set(label_config.class_names)
                    if label_classes:
                        label_colors = generate_label_colors(label_classes)
                        label_colors_dict[label_layer_name] = label_colors
                        logger.info(f"Found {len(label_classes)} label classes for {label_layer_name}: {sorted(label_classes)}")
                    else:
                        # For detection tasks, use default 'detected' class if class_names is empty list
                        if task_type == "detection":
                            label_classes = {"detected"}
                            label_colors = generate_label_colors(label_classes)
                            label_colors_dict[label_layer_name] = label_colors
                            logger.info(f"Empty class_names for {label_layer_name} (detection task) - using default 'detected' class")
                        else:
                            logger.warning(f"No label classes in config for {label_layer_name} - masks will not be generated")
        
        # Store state
        self.dataset = dataset
        self.layers = layers
        self.bands = bands
        self.normalization = normalization
        self.task_type = task_type
        self.label_colors_dict = label_colors_dict
        self.group_idx = group_idx

        # Load all windows on startup
        logger.info(f"Loading all windows from dataset {dataset_path}")
        self.windows = dataset.load_windows()
        logger.info(f"Loaded {len(self.windows)} windows from dataset")
        logger.info(f"Layers: {layers}")
        logger.info(f"Bands: {bands}")
        logger.info(f"Normalization: {normalization}")
        logger.info(f"Task type: {task_type}")

        # Create custom HTTP handler
        server_instance = self

        class VisualizationHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.strip("/")
                
                # Route: /images/<window_idx>/<layer>
                if path.startswith("images/"):
                    parts = path.split("/")
                    if len(parts) == 3 and parts[0] == "images":
                        try:
                            window_idx = int(parts[1])
                            layer_name = parts[2]
                            
                            if window_idx < 0 or window_idx >= len(server_instance.windows):
                                self.send_response(404)
                                self.send_header("Content-type", "text/plain")
                                self.end_headers()
                                self.wfile.write(b"Window index out of range")
                                return
                            
                            window = server_instance.windows[window_idx]
                            
                            # Generate image on-demand
                            try:
                                # Get label colors for this layer
                                layer_label_colors = None
                                if server_instance.label_colors_dict and layer_name in server_instance.label_colors_dict:
                                    layer_label_colors = server_instance.label_colors_dict[layer_name]
                                elif server_instance.label_colors_dict:
                                    # Use first label layer's colors as fallback
                                    first_label_layer = list(server_instance.label_colors_dict.keys())[0]
                                    layer_label_colors = server_instance.label_colors_dict[first_label_layer]
                                
                                image_bytes = server_instance._generate_image_as_bytes(
                                    window,
                                    layer_name,
                                    server_instance.dataset,
                                    server_instance.bands,
                                    server_instance.normalization,
                                    server_instance.task_type,
                                    layer_label_colors,
                                    server_instance.label_colors_dict,
                                    server_instance.group_idx,
                                )
                                
                                self.send_response(200)
                                self.send_header("Content-type", "image/png")
                                self.send_header("Content-Length", str(len(image_bytes)))
                                self.end_headers()
                                self.wfile.write(image_bytes)
                            except FileNotFoundError:
                                # No data available (e.g., no features in label layer) - return 404
                                self.send_response(404)
                                self.send_header("Content-type", "text/plain")
                                self.end_headers()
                                self.wfile.write(b"Image not available")
                            except Exception as e:
                                logger.error(f"Failed to generate image for window {window_idx}, layer {layer_name}: {e}")
                                self.send_response(500)
                                self.send_header("Content-type", "text/plain")
                                self.end_headers()
                                self.wfile.write(f"Error generating image: {e}".encode())
                        except (ValueError, IndexError) as e:
                            self.send_response(400)
                            self.send_header("Content-type", "text/plain")
                            self.end_headers()
                            self.wfile.write(b"Invalid image URL format")
                    else:
                        self.send_response(404)
                        self.send_header("Content-type", "text/plain")
                        self.end_headers()
                        self.wfile.write(b"Invalid image URL")
                
                # Route: / (main page)
                else:
                    # Sample windows
                    if len(server_instance.windows) > max_samples:
                        sampled_windows = random.sample(server_instance.windows, max_samples)
                    else:
                        sampled_windows = server_instance.windows
                    
                    # Render template
                    html = server_instance._render_template(sampled_windows)
                    
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(html.encode("utf-8"))
            
            def log_message(self, format, *args):
                # Suppress default logging
                pass

        try:
            with socketserver.TCPServer((host, port), VisualizationHandler) as httpd:
                logger.info(f"Serving on http://{host}:{port}")
                logger.info(f"Open http://localhost:{port} in your browser")
                logger.info(f"Loaded {len(self.windows)} windows - refreshing the page will show a different random sample")
                
                # Save HTML if requested (render template and save)
                if save_html:
                    from datetime import datetime
                    
                    if len(self.windows) > max_samples:
                        sampled_windows = random.sample(self.windows, max_samples)
                    else:
                        sampled_windows = self.windows
                    
                    html = self._render_template(sampled_windows)
                    dataset_name = dataset_path.name
                    date_str = datetime.now().strftime("%Y%m%d")
                    outputs_dir = dataset_path.parent / "outputs"
                    outputs_dir.mkdir(exist_ok=True)
                    saved_html_path = outputs_dir / f"{dataset_name}_{date_str}.html"
                    with open(saved_html_path, "w", encoding="utf-8") as f:
                        f.write(html)
                    logger.info(f"Saved HTML to {saved_html_path}")
                
                httpd.serve_forever()
        except OSError as e:
            if e.errno == 48:  # Address already in use
                logger.error(f"Port {port} is already in use. Try a different port with --port")
                raise
            raise


def parse_bands_arg(bands_str: str | None) -> dict[str, list[str]]:
    """Parse --bands argument.

    Format: "layer1:band1,band2,band3;layer2:band1,band2" or just "band1,band2,band3" for single layer
    """
    if not bands_str:
        return {}

    bands_dict = {}
    # Split by semicolon for multiple layers
    parts = bands_str.split(";")
    for part in parts:
        if ":" in part:
            layer_name, band_str = part.split(":", 1)
            band_names = [b.strip() for b in band_str.split(",")]
            bands_dict[layer_name.strip()] = band_names
        else:
            # Single layer format - use first layer (user should specify layer name)
            band_names = [b.strip() for b in part.split(",")]
            if bands_dict:
                raise ValueError("Cannot use single-layer format with multiple layers. Use layer:band1,band2 format")
            bands_dict["_default"] = band_names
    return bands_dict


def parse_normalization_arg(norm_str: str | None) -> dict[str, str]:
    """Parse --normalization argument.

    Format: "layer1:method1;layer2:method2" or just "method" for single layer
    """
    if not norm_str:
        return {}

    norm_dict = {}
    # Split by semicolon for multiple layers
    parts = norm_str.split(";")
    for part in parts:
        if ":" in part:
            layer_name, method = part.split(":", 1)
            norm_dict[layer_name.strip()] = method.strip()
        else:
            # Single layer format
            if norm_dict:
                raise ValueError("Cannot use single-layer format with multiple layers. Use layer:method format")
            norm_dict["_default"] = part.strip()
    return norm_dict


def main():
    parser = argparse.ArgumentParser(description="Visualize rslearn dataset in a web browser")
    parser.add_argument("dataset_path", type=str, help="Path to dataset directory (containing config.json)")
    parser.add_argument(
        "--layers",
        type=str,
        nargs="+",
        help="List of layer names to visualize, including label layers (default: all raster layers except label layers)",
    )
    parser.add_argument(
        "--bands",
        type=str,
        help='Bands to visualize per layer. Format: "layer1:band1,band2,band3;layer2:band1,band2" '
        "or just 'band1,band2,band3' for single layer (default: first 3 bands)",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        help='Normalization method per layer. Format: "layer1:method1;layer2:method2" '
        'or just "method" for single layer (default: sentinel2_rgb)',
    )
    parser.add_argument(
        "--task_type",
        type=str,
        choices=["classification", "regression", "detection", "segmentation"],
        help="Task type (default: auto-detect, but user should specify)",
    )
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum number of windows to sample")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument(
        "--save_html", action="store_true", help="Save HTML file to outputs/{dataset_name}_{YYYYMMDD}.html"
    )
    parser.add_argument("--group_idx", type=int, default=0, help="Item group index (default: 0)")

    args = parser.parse_args()

    # Parse bands and normalization
    bands_dict = parse_bands_arg(args.bands)
    normalization_dict = parse_normalization_arg(args.normalization)

    server = VisualizationServer()
    server.run(
        dataset_path=args.dataset_path,
        layers=args.layers,
        bands=bands_dict if bands_dict else None,
        normalization=normalization_dict if normalization_dict else None,
        task_type=args.task_type,
        max_samples=args.max_samples,
        port=args.port,
        host=args.host,
        save_html=args.save_html,
        group_idx=args.group_idx,
    )


if __name__ == "__main__":
    main()
