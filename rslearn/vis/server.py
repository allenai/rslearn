"""Visualization server for rslearn datasets.

This module provides a web server to visualize rslearn datasets using the Dataset/Window APIs.
"""

import argparse
import atexit
import base64
import http.server
import random
import shutil
import socketserver
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from upath import UPath

from rslearn.config import LayerType
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.log_utils import get_logger

from .labels import detect_label_classes, generate_label_colors, get_classification_label, read_label_features
from .layers import read_raster_layer, read_vector_layer
from .vis import VISUALIZATION_IMAGE_SIZE, array_to_png, features_to_mask, raster_label_to_mask

logger = get_logger(__name__)


def _get_label_layer_name(dataset: Dataset) -> str | None:
    """Get the label layer name from dataset config.
    
    Checks for 'label', 'labels', or 'label_raster' in order of preference.
    
    Args:
        dataset: Dataset object
        
    Returns:
        Label layer name or None if none exists
    """
    if "label" in dataset.layers:
        return "label"
    elif "labels" in dataset.layers:
        return "labels"
    elif "label_raster" in dataset.layers:
        return "label_raster"
    return None


def get_default_bands_for_layer(layer_config) -> list[str]:
    """Get default bands for a layer (first 3 bands from first band set).

    Args:
        layer_config: LayerConfig object

    Returns:
        List of band names (first 3, or all if fewer than 3)
    """
    if not layer_config.band_sets:
        return []
    band_set = layer_config.band_sets[0]
    bands = band_set.bands
    if not bands:
        return []
    # Return first 3 bands, or all if fewer than 3
    return bands[:3] if len(bands) >= 3 else bands


def get_default_normalization_for_layer(layer_config) -> str:
    """Get default normalization method for a layer.

    Args:
        layer_config: LayerConfig object

    Returns:
        Normalization method name (default: "sentinel2_rgb")
    """
    # If data_source is set, it's likely an image - use sentinel2_rgb
    # If data_source is not set, it might be a mask - but for now, default to sentinel2_rgb
    # The user can override with --normalization
    return "sentinel2_rgb"


class VisualizationServer:
    """Visualization server for rslearn datasets using Dataset/Window APIs."""

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

        # Get time range
        if window.time_range:
            start = window.time_range[0].isoformat()[:10]
            end = window.time_range[1].isoformat()[:10]
            parts.append(f"Time: {start} to {end}")

        # Get lat/lon from geometry centroid
        try:
            geom_wgs84 = window.get_geometry().to_projection(WGS84_PROJECTION)
            centroid = geom_wgs84.shp.centroid
            lon = float(centroid.x)
            lat = float(centroid.y)
            parts.insert(0, f"Lat: {lat:.4f}, Lon: {lon:.4f}")
        except Exception as e:
            logger.warning(f"Failed to get centroid for window {window.name}: {e}")

        return "<br>".join(parts) if parts else "Unknown", lat, lon

    def _encode_image_to_base64(self, image_path: Path) -> str:
        """Encode an image file to base64 data URI.

        Args:
            image_path: Path to the image file (PNG)

        Returns:
            Base64 data URI string for use in HTML img src attribute
        """
        try:
            with open(image_path, "rb") as img_file:
                img_data = img_file.read()
                encoded = base64.b64encode(img_data).decode("utf-8")
                return f"data:image/png;base64,{encoded}"
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return ""

    def generate_html(
        self,
        window_results: list[dict[str, Any]],
        layer_names: list[str],
        output_dir: Path,
        html_output_path: Path,
        label_colors: dict[str, tuple[int, int, int]] | None = None,
        task_type: str | None = None,
    ) -> None:
        """Generate HTML gallery for visualization.

        Args:
            window_results: List of dictionaries with window processing results
            layer_names: List of layer names to display (in order)
            output_dir: Directory where PNGs are stored
            html_output_path: Path to save HTML file
            label_colors: Dictionary mapping label class names to RGB color tuples
            task_type: Task type (classification, regression, detection, segmentation)
        """
        output_dir = Path(output_dir)
        html_output_path = Path(html_output_path)
        html_dir = html_output_path.parent

        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>rslearn Dataset Visualization</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 40px;
        }
        .legend {
            background: white;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .legend h3 {
            margin: 0 0 10px 0;
            color: #555;
        }
        .legend-items {
            display: flex;
            justify-content: center;
            gap: 30px;
            flex-wrap: wrap;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .legend-color {
            width: 30px;
            height: 20px;
            border: 1px solid #999;
        }
        .window-section {
            background: white;
            margin-bottom: 40px;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .window-header {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 3px solid #3498db;
        }
        .window-info {
            color: #666;
            font-size: 14px;
            margin-bottom: 20px;
        }
        .window-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .image-container {
            border-radius: 4px;
            overflow: hidden;
            background: #fff;
            display: inline-block;
            width: fit-content;
        }
        .image-container img {
            max-width: 100%;
            height: auto;
            display: block;
            background: #fff;
        }
        .image-label {
            padding: 8px;
            font-size: 14px;
            color: #333;
            text-align: center;
            background: #f8f9fa;
            border-top: 1px solid #e0e0e0;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <h1>rslearn Dataset Visualization</h1>
"""

        # Generate legend dynamically from label colors (skip for classification tasks)
        if label_colors and task_type != "classification":
            html_content += """
    <div class="legend">
        <h3>Mask Color Legend</h3>
        <div class="legend-items">
"""
            # Sort classes for consistent display
            sorted_labels = sorted(label_colors.keys())
            for label in sorted_labels:
                color = label_colors[label]
                color_hex = f"rgb({color[0]}, {color[1]}, {color[2]})"
                html_content += f"""
            <div class="legend-item">
                <div class="legend-color" style="background: {color_hex};"></div>
                <span>{label}</span>
            </div>
"""
            html_content += """
        </div>
    </div>
"""

        html_content += """
"""

        # Add each window section
        for result in window_results:
            window_name = result["window_name"]
            window_info, lat, lon = self.format_window_info(result["window"])

            # Add Google Maps link if lat/lon available
            maps_link_html = ""
            if lat is not None and lon is not None:
                maps_url = f"https://www.google.com/maps?q={lat},{lon}"
                maps_link_html = f' <a href="{maps_url}" target="_blank" style="color: #3498db; text-decoration: none; margin-left: 10px;">🗺️ View on Google Maps</a>'

            html_content += f"""
    <div class="window-section">
        <div class="window-header">{window_name}{maps_link_html}</div>
        <div class="window-info">{window_info}</div>
        <div class="window-grid">
"""

            # Add layer images
            for layer_name in layer_names:
                if layer_name in result["layer_images"]:
                    img_path = Path(result["layer_images"][layer_name])
                    # Ensure we have an absolute path
                    if not img_path.is_absolute():
                        img_path = output_dir / img_path
                    img_data_uri = self._encode_image_to_base64(img_path)
                    if img_data_uri:
                        html_content += f"""
            <div class="image-container">
                <img src="{img_data_uri}" alt="{layer_name}">
                <div class="image-label">{layer_name}</div>
            </div>
"""

            # Add mask or label text if available (once per window, not per layer)
            if result.get("mask_path"):
                mask_path = Path(result["mask_path"])
                # Ensure we have an absolute path
                if not mask_path.is_absolute():
                    mask_path = output_dir / mask_path
                mask_data_uri = self._encode_image_to_base64(mask_path)
                if mask_data_uri:
                    html_content += f"""
            <div class="image-container">
                <img src="{mask_data_uri}" alt="Label Mask">
                <div class="image-label">Label Mask</div>
            </div>
"""
            elif result.get("label_text"):
                html_content += f"""
            <div class="image-container">
                <div style="padding: 20px; font-size: 18px; font-weight: bold;">Label: {result["label_text"]}</div>
            </div>
"""

            html_content += """
        </div>
    </div>
"""

        html_content += """
</body>
</html>
"""

        html_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(html_output_path, "w") as f:
            f.write(html_content)

        logger.info(f"Generated HTML visualization at {html_output_path}")

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

        # Process raster layers
        for layer_name in layer_names:
            if layer_name not in dataset.layers:
                logger.warning(f"Layer {layer_name} not found in dataset config")
                continue

            layer_config = dataset.layers[layer_name]
            if layer_config.type != LayerType.RASTER:
                continue

            # For visualization, try to read the layer even if not marked as completed
            # (the data files might still exist)
            if not window.is_layer_completed(layer_name, group_idx=group_idx):
                logger.debug(f"Layer {layer_name} not marked as completed for window {window.name}, attempting to read anyway")

            try:
                # Read raster array
                band_names = bands.get(layer_name, get_default_bands_for_layer(layer_config))
                array = read_raster_layer(
                    window, layer_name, layer_config, band_names, group_idx=group_idx
                )

                # Convert to PNG
                norm_method = normalization.get(
                    layer_name, get_default_normalization_for_layer(layer_config)
                )
                img_path = window_dir / f"{layer_name}.png"
                array_to_png(array, img_path, norm_method)

                result["layer_images"][layer_name] = img_path

            except Exception as e:
                logger.error(f"Failed to process layer {layer_name} for window {window.name}: {e}")
                continue

        # Process label layer
        label_layer_name = _get_label_layer_name(dataset)
        if label_layer_name is None:
            logger.debug(f"No 'label' or 'labels' layer found in dataset config (window {window.name})")
            return result

        label_config = dataset.layers[label_layer_name]
        # For visualization, try to read the label layer even if not marked as completed
        if not window.is_layer_completed(label_layer_name, group_idx=group_idx):
            logger.debug(f"Label layer {label_layer_name} not marked as completed for window {window.name}, attempting to read anyway")

        try:
            if task_type == "classification":
                # Get text label (works with both vector and raster labels)
                label_text = get_classification_label(window, label_config, label_layer_name, group_idx=group_idx)
                result["label_text"] = label_text
            elif task_type in ("segmentation", "detection") and label_colors:
                if label_config.type == LayerType.VECTOR:
                    # Generate mask from vector features
                    features = read_label_features(window, label_config, label_layer_name, group_idx=group_idx)
                    logger.info(f"Window {window.name}: Found {len(features)} label features")
                    if features:
                        mask_path = window_dir / "label_mask.png"

                        # For detection, we need a reference raster image
                        reference_array = None
                        if task_type == "detection" and result["layer_images"]:
                            # Use first available layer image as reference
                            first_layer_name = layer_names[0]
                            if first_layer_name in result["layer_images"]:
                                # Re-read the array for overlay
                                first_layer_config = dataset.layers[first_layer_name]
                                first_band_names = bands.get(
                                    first_layer_name, get_default_bands_for_layer(first_layer_config)
                                )
                                reference_array = read_raster_layer(
                                    window,
                                    first_layer_name,
                                    first_layer_config,
                                    first_band_names,
                                    group_idx=group_idx,
                                )
                                logger.info(f"Window {window.name}: Read reference array with shape {reference_array.shape}, bounds: {window.bounds}")
                            else:
                                logger.warning(f"Window {window.name}: No layer images available for detection overlay")
                        else:
                            if task_type == "detection":
                                logger.warning(f"Window {window.name}: No reference array available for detection (task_type={task_type}, layer_images={result['layer_images']})")

                        features_to_mask(
                            features,
                            window.bounds,
                            window.projection,
                            label_colors,
                            mask_path,
                            reference_raster_array=reference_array,
                        )
                        result["mask_path"] = mask_path
                    else:
                        logger.debug(f"Window {window.name}: No features found in label layer")
                elif label_config.type == LayerType.RASTER:
                    # Generate mask from raster labels
                    if task_type == "segmentation":
                        # Read raster label layer
                        if not label_config.band_sets:
                            logger.debug(f"No band sets in raster label layer {label_layer_name}")
                            return result
                        band_set = label_config.band_sets[0]
                        if not band_set.bands:
                            logger.debug(f"No bands in raster label layer {label_layer_name}")
                            return result
                        
                        # Read first band
                        band_name = band_set.bands[0]
                        label_array = read_raster_layer(
                            window, label_layer_name, label_config, [band_name], group_idx=group_idx
                        )
                        
                        # Log array info for debugging
                        if label_array.ndim == 3:
                            label_2d = label_array[0, :, :]
                        else:
                            label_2d = label_array
                        unique_vals, counts = np.unique(label_2d, return_counts=True)
                        logger.info(
                            f"Window {window.name}: Label array shape {label_array.shape}, "
                            f"bounds: {window.bounds}, "
                            f"unique values (top 5 by count): {sorted(zip(counts, unique_vals), reverse=True)[:5]}"
                        )
                        
                        mask_path = window_dir / "label_mask.png"
                        raster_label_to_mask(label_array, label_colors, label_config, mask_path)
                        result["mask_path"] = mask_path
                    else:
                        logger.debug(f"Raster labels not supported for task type {task_type}")

        except Exception as e:
            logger.error(f"Failed to process label for window {window.name}: {e}")

        return result

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

        # Determine layers to visualize
        if layers is None:
            layers = [
                name
                for name, config in dataset.layers.items()
                if name != "label" and config.type == LayerType.RASTER
            ]
            layers = sorted(layers)

        # Validate layers
        for layer_name in layers:
            if layer_name not in dataset.layers:
                raise ValueError(f"Layer {layer_name} not found in dataset config")
            if dataset.layers[layer_name].type != LayerType.RASTER:
                raise ValueError(f"Layer {layer_name} is not a raster layer")

        # Get default bands and normalization if not provided
        bands = bands or {}
        normalization = normalization or {}
        for layer_name in layers:
            layer_config = dataset.layers[layer_name]
            if layer_name not in bands:
                bands[layer_name] = get_default_bands_for_layer(layer_config)
            if layer_name not in normalization:
                normalization[layer_name] = get_default_normalization_for_layer(layer_config)

        # Auto-detect task type if not provided
        if task_type is None:
            # Sample a few windows to detect task type
            all_windows = dataset.load_windows()
            sample_windows = all_windows[: min(10, len(all_windows))]
            label_layer_name = _get_label_layer_name(dataset)
            if sample_windows and label_layer_name:
                label_config = dataset.layers[label_layer_name]
                label_classes = detect_label_classes(sample_windows, label_config, label_layer_name)
                # Simple heuristic: if only 1 class, likely classification; if many polygons/points, segmentation/detection
                # This is a simplified heuristic - user should specify task_type
                task_type = "classification"  # Default, user should specify
                logger.warning(
                    "task_type not specified, defaulting to 'classification'. "
                    "Please specify --task_type for accurate visualization."
                )
            else:
                task_type = "classification"

        # Get label colors for segmentation/detection
        label_colors = None
        label_layer_name = _get_label_layer_name(dataset)
        if task_type in ("segmentation", "detection") and label_layer_name:
            all_windows = dataset.load_windows()
            label_config = dataset.layers[label_layer_name]
            label_classes = detect_label_classes(all_windows, label_config, label_layer_name)
            if label_classes:
                label_colors = generate_label_colors(label_classes)
                logger.info(f"Detected {len(label_classes)} label classes: {sorted(label_classes)}")
            else:
                # For detection tasks, use default 'detected' class if no classes found
                if task_type == "detection":
                    label_classes = {"detected"}
                    label_colors = generate_label_colors(label_classes)
                    logger.info("No label classes detected for detection task - using default 'detected' class")
                else:
                    logger.warning("No label classes detected - masks will not be generated")

        # Create temporary output directory
        output_dir = Path(tempfile.mkdtemp(prefix="rslearn_vis_"))
        atexit.register(shutil.rmtree, output_dir, ignore_errors=True)

        logger.info(f"Processing up to {max_samples} windows from dataset {dataset_path}")
        logger.info(f"Layers: {layers}")
        logger.info(f"Bands: {bands}")
        logger.info(f"Normalization: {normalization}")
        logger.info(f"Task type: {task_type}")

        # Load and sample windows
        all_windows = dataset.load_windows()
        logger.info(f"Loaded {len(all_windows)} windows from dataset")
        if len(all_windows) > max_samples:
            sampled_windows = random.sample(all_windows, max_samples)
        else:
            sampled_windows = all_windows

        # Process windows
        window_results = []
        windows_skipped_no_layers = 0
        for window in sampled_windows:
            try:
                result = self.process_window(
                    window,
                    dataset,
                    layers,
                    bands,
                    normalization,
                    output_dir,
                    task_type,
                    label_colors,
                    group_idx=group_idx,
                )
                # Only include windows that have at least one layer image
                if result["layer_images"]:
                    window_results.append(result)
                else:
                    windows_skipped_no_layers += 1
            except Exception as e:
                logger.error(f"Failed to process window {window.name}: {e}")
                continue

        if windows_skipped_no_layers > 0:
            logger.warning(
                f"Skipped {windows_skipped_no_layers} windows with no layer images "
                f"(layers may not be completed or processing failed silently)"
            )
        logger.info(f"Processed {len(window_results)} windows successfully")

        # Generate HTML
        html_path = output_dir / "index.html"
        self.generate_html(window_results, layers, output_dir, html_path, label_colors, task_type)

        # Save HTML to outputs directory if requested
        if save_html:
            from datetime import datetime

            dataset_name = dataset_path.name
            date_str = datetime.now().strftime("%Y%m%d")
            outputs_dir = dataset_path.parent / "outputs"
            outputs_dir.mkdir(exist_ok=True)
            saved_html_path = outputs_dir / f"{dataset_name}_{date_str}.html"
            shutil.copy(html_path, saved_html_path)
            logger.info(f"Saved HTML to {saved_html_path}")

        # Start HTTP server
        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(output_dir), **kwargs)

        try:
            with socketserver.TCPServer((host, port), Handler) as httpd:
                logger.info(f"Serving on http://{host}:{port}")
                logger.info(f"Open http://localhost:{port} in your browser")
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
        help="List of layer names to visualize (default: all raster layers except 'label')",
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
