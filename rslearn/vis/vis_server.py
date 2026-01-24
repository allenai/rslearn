"""Visualization server for rslearn datasets.

This module provides a web server to visualize rslearn datasets using the Dataset/Window APIs.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any

from flask import Flask, Response
from upath import UPath

from rslearn.config import LayerType
from rslearn.dataset import Dataset, Window
from rslearn.log_utils import get_logger

from .render_raster_label import read_raster_layer, render_raster_label_as_bytes
from .render_sensor_image import render_sensor_image_as_bytes
from .render_vector_label import get_vector_label_by_property, render_vector_label_image
from .utils import _escape_html, format_window_info, generate_label_colors

logger = get_logger(__name__)


def generate_image_as_bytes(
    window: Window,
    layer_name: str,
    dataset: Dataset,
    bands: dict[str, list[str]],
    normalization: dict[str, str],
    task_type: str,
    label_colors: dict[str, tuple[int, int, int]] | None,
    label_colors_dict: dict[str, dict[str, tuple[int, int, int]]] | None = None,
    group_idx: int = 0,
    label_layers: list[str] | None = None,
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
        label_layers: List of layer names that are labels

    Returns:
        PNG image bytes
    """
    label_layers = label_layers or []
    layer_config = dataset.layers[layer_name]

    # Render raster sensor image
    if layer_config.type == LayerType.RASTER and layer_name not in label_layers:
        if layer_name in bands and layer_name in normalization:
            array = read_raster_layer(
                window, layer_name, layer_config, bands[layer_name], group_idx=group_idx
            )
            return render_sensor_image_as_bytes(array, normalization[layer_name])
        else:
            raise ValueError(
                f"Bands or normalization not specified for layer {layer_name}"
            )

    # Render labels
    elif layer_name in label_layers:
        layer_label_colors = (
            label_colors_dict.get(layer_name) if label_colors_dict else label_colors
        )
        if not layer_label_colors:
            raise ValueError(f"No label colors available for layer {layer_name}")

        # Render raster label
        if layer_config.type == LayerType.RASTER:
            band_set = layer_config.band_sets[0]
            label_array = read_raster_layer(
                window,
                layer_name,
                layer_config,
                [band_set.bands[0]],
                group_idx=group_idx,
            )
            return render_raster_label_as_bytes(
                label_array, layer_label_colors, layer_config
            )

        # Render vector label
        elif layer_config.type == LayerType.VECTOR:
            return render_vector_label_image(
                window,
                layer_name,
                layer_config,
                task_type,
                layer_label_colors,
                dataset,
                label_layers,
                group_idx,
                bands,
                normalization,
            )

    raise ValueError(f"Layer {layer_name} is not a raster sensor image or label layer")


# Global state (set during initialization)
_app_state: dict[str, Any] = {}


def render_template(
    sampled_windows: list[Window],
    dataset: Dataset,
    layers: list[str],
    label_layers: list[str],
    task_type: str,
    label_colors_dict: dict[str, dict[str, tuple[int, int, int]]],
    group_idx: int,
) -> str:
    """Render the HTML template with window data.

    Args:
        sampled_windows: List of windows to display
        dataset: Dataset object
        layers: List of all layers
        label_layers: List of label layer names
        task_type: Task type
        label_colors_dict: Dictionary mapping layer_name -> label_colors
        group_idx: Item group index

    Returns:
        Rendered HTML as string
    """
    window_data: list[dict[str, Any]] = []
    for idx, window in enumerate(sampled_windows):
        info_html, lat, lon = format_window_info(window)
        maps_link = (
            f"https://www.google.com/maps?q={lat},{lon}"
            if lat is not None and lon is not None
            else None
        )

        available_layers = set()
        mask_layers = []
        label_texts = {}

        for layer_name in layers:
            if layer_name not in dataset.layers:
                continue
            layer_config = dataset.layers[layer_name]
            try:
                if (
                    layer_config.type == LayerType.RASTER
                    and layer_name not in label_layers
                ):
                    if window.is_layer_completed(layer_name, group_idx=group_idx):
                        available_layers.add(layer_name)
                elif layer_name in label_layers:
                    if layer_config.type == LayerType.VECTOR:
                        if task_type == "classification":
                            try:
                                if not window.is_layer_completed(
                                    layer_name, group_idx=group_idx
                                ):
                                    logger.debug(
                                        f"Layer {layer_name} not marked as completed for window {window.name}, attempting to read anyway"
                                    )
                                label_text = get_vector_label_by_property(
                                    window,
                                    layer_config,
                                    layer_name,
                                    group_idx=group_idx,
                                )
                                if label_text is not None:
                                    label_texts[layer_name] = label_text
                            except Exception as e:
                                logger.debug(
                                    f"Failed to get label text for {layer_name} in window {window.name}: {e}"
                                )
                        else:
                            if window.is_layer_completed(
                                layer_name, group_idx=group_idx
                            ):
                                mask_layers.append(layer_name)
                    elif layer_config.type == LayerType.RASTER:
                        if window.is_layer_completed(layer_name, group_idx=group_idx):
                            mask_layers.append(layer_name)
            except Exception as e:
                logger.debug(
                    f"Error processing layer {layer_name} for window {window.name}: {e}"
                )
                continue

        window_data.append(
            {
                "idx": idx,
                "name": window.name,
                "info_html": info_html,
                "maps_link": maps_link,
                "available_layers": available_layers,
                "mask_layers": mask_layers,
                "label_texts": label_texts,
            }
        )

    label_colors = None
    if label_colors_dict:
        first_label_layer = list(label_colors_dict.keys())[0]
        label_colors = label_colors_dict[first_label_layer]

    # Build HTML
    html_parts = [
        """<!DOCTYPE html>
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
    ]

    # Legend
    if label_colors and task_type != "classification":
        html_parts.append('    <div class="legend">\n')
        html_parts.append("        <h3>Mask Color Legend</h3>\n")
        html_parts.append('        <div class="legend-items">\n')
        for label in sorted(label_colors.keys()):
            r, g, b = label_colors[label]
            html_parts.append(
                f'            <div class="legend-item">\n'
                f'                <div class="legend-color" style="background: rgb({r}, {g}, {b});"></div>\n'
                f"                <span>{_escape_html(label)}</span>\n"
                f"            </div>\n"
            )
        html_parts.append("        </div>\n")
        html_parts.append("    </div>\n")

    # Windows
    for window_dict in window_data:
        html_parts.append('    <div class="window-section">\n')
        html_parts.append('        <div class="window-header">\n')
        html_parts.append(f"            {_escape_html(window_dict['name'])}\n")
        if window_dict["maps_link"]:
            html_parts.append(
                f'            <a href="{_escape_html(window_dict["maps_link"])}" target="_blank" style="color: #3498db; text-decoration: none; margin-left: 10px;">🗺️ View on Google Maps</a>\n'
            )
        html_parts.append("        </div>\n")
        html_parts.append(
            f'        <div class="window-info">{window_dict["info_html"]}</div>\n'
        )
        html_parts.append('        <div class="window-grid">\n')

        # Regular layers
        for layer_name in layers:
            if layer_name in window_dict["available_layers"]:
                html_parts.append('            <div class="image-container">\n')
                html_parts.append(
                    f'                <img src="/images/{window_dict["idx"]}/{_escape_html(layer_name)}" alt="{_escape_html(layer_name)}">\n'
                )
                html_parts.append(
                    f'                <div class="image-label">{_escape_html(layer_name)}</div>\n'
                )
                html_parts.append("            </div>\n")

        # Mask layers
        for mask_layer in window_dict["mask_layers"]:
            html_parts.append(
                f'            <div class="image-container" id="mask-{window_dict["idx"]}-{_escape_html(mask_layer)}">\n'
            )
            html_parts.append(
                f'                <img src="/images/{window_dict["idx"]}/{_escape_html(mask_layer)}" alt="Label Mask: {_escape_html(mask_layer)}" onerror="this.closest(\'.image-container\').remove()">\n'
            )
            html_parts.append(
                f'                <div class="image-label">Label Mask: {_escape_html(mask_layer)}</div>\n'
            )
            html_parts.append("            </div>\n")

        # Label texts (classification)
        for layer_name, label_text in window_dict["label_texts"].items():
            html_parts.append('            <div class="image-container">\n')
            html_parts.append(
                f'                <div style="padding: 20px; font-size: 18px; font-weight: bold;">{_escape_html(layer_name)}: {_escape_html(label_text)}</div>\n'
            )
            html_parts.append("            </div>\n")

        html_parts.append("        </div>\n")
        html_parts.append("    </div>\n")

    html_parts.append("</body>\n</html>")
    return "".join(html_parts)


def create_app(
    dataset: Dataset,
    windows: list[Window],
    layers: list[str],
    bands: dict[str, list[str]],
    normalization: dict[str, str],
    task_type: str,
    label_colors_dict: dict[str, dict[str, tuple[int, int, int]]],
    group_idx: int,
    label_layers: list[str],
    max_samples: int,
) -> Flask:
    """Create and configure Flask app.

    Args:
        dataset: Dataset object
        windows: List of all windows
        layers: List of all layers
        bands: Dictionary mapping layer_name -> list of band names
        normalization: Dictionary mapping layer_name -> normalization method
        task_type: Task type
        label_colors_dict: Dictionary mapping layer_name -> label_colors
        group_idx: Item group index
        label_layers: List of label layer names
        max_samples: Maximum number of windows to sample

    Returns:
        Configured Flask app
    """
    app = Flask(__name__)

    @app.route("/")
    def index() -> str:
        """Render the main visualization page."""
        if len(windows) > max_samples:
            sampled_windows = random.sample(windows, max_samples)
        else:
            sampled_windows = windows

        return render_template(
            sampled_windows,
            dataset,
            layers,
            label_layers,
            task_type,
            label_colors_dict,
            group_idx,
        )

    @app.route("/images/<int:window_idx>/<layer_name>")
    def get_image(window_idx: int, layer_name: str) -> Response:
        """Generate and serve an image for a specific window/layer.

        Args:
            window_idx: Index of the window in the windows list
            layer_name: Name of the layer to visualize

        Returns:
            PNG image response or error response
        """
        if window_idx < 0 or window_idx >= len(windows):
            return Response(
                "Window index out of range", status=404, mimetype="text/plain"
            )

        window = windows[window_idx]

        layer_label_colors = None
        if label_colors_dict and layer_name in label_colors_dict:
            layer_label_colors = label_colors_dict[layer_name]
        elif label_colors_dict:
            first_label_layer = list(label_colors_dict.keys())[0]
            layer_label_colors = label_colors_dict[first_label_layer]

        image_bytes = generate_image_as_bytes(
            window,
            layer_name,
            dataset,
            bands,
            normalization,
            task_type,
            layer_label_colors,
            label_colors_dict,
            group_idx,
            label_layers,
        )

        return Response(
            image_bytes,
            mimetype="image/png",
            headers={"Content-Length": str(len(image_bytes))},
        )

    return app


def run(
    dataset_path: str | Path | UPath,
    layers: list[str] | None = None,
    bands: dict[str, list[str]] | None = None,
    normalization: dict[str, str] | None = None,
    task_type: str | None = None,
    max_samples: int = 100,
    port: int = 8000,
    host: str = "0.0.0.0",
    group_idx: int = 0,
    label_layers: list[str] | None = None,
) -> None:
    """Run the visualization server.

    Args:
        dataset_path: Path to dataset directory (containing config.json)
        layers: List of layer names to visualize
        bands: Dictionary mapping layer_name -> list of band names
        normalization: Dictionary mapping layer_name -> normalization method
        task_type: Task type - "classification", "regression", "detection", or "segmentation"
        max_samples: Maximum number of windows to sample
        port: Port to serve on
        host: Host to bind to
        group_idx: Item group index (default 0)
        label_layers: List of layer names that are labels
    """
    dataset_path = UPath(dataset_path)
    dataset = Dataset(dataset_path)

    label_layers = label_layers or []

    if layers is None:
        raise ValueError("--layers is required")
    all_layers = list(set(layers + label_layers))
    raster_image_layers = [name for name in all_layers if name not in label_layers]
    label_layers_in_list = [name for name in all_layers if name in label_layers]

    bands = bands or {}
    normalization = normalization or {}
    for layer_name in raster_image_layers:
        if layer_name not in bands:
            raise ValueError(
                f"Bands not specified for layer {layer_name}. Please provide --bands {layer_name}:band1,band2,band3"
            )
        if layer_name not in normalization:
            raise ValueError(
                f"Normalization not specified for layer {layer_name}. Please provide --normalization {layer_name}:method"
            )

    label_colors_dict = {}
    if task_type in ("segmentation", "detection"):
        for label_layer_name in label_layers_in_list:
            label_config = dataset.layers[label_layer_name]
            if not label_config.class_names:
                raise ValueError(
                    f"class_names must be specified in the config for label layer '{label_layer_name}'. "
                    "Auto-detection of class names is not supported."
                )
            else:
                label_classes = set(label_config.class_names)
                label_colors = generate_label_colors(label_classes)
                label_colors_dict[label_layer_name] = label_colors
                logger.info(
                    f"Found {len(label_classes)} label classes for {label_layer_name}: {sorted(label_classes)}"
                )

    logger.info(f"Loading all windows from dataset {dataset_path}")
    windows = dataset.load_windows()
    logger.info(f"Loaded {len(windows)} windows from dataset")
    logger.info(f"Layers: {all_layers}")
    logger.info(f"Bands: {bands}")
    logger.info(f"Normalization: {normalization}")
    logger.info(f"Task type: {task_type}")

    if task_type is None:
        raise ValueError("--task_type is required")

    app = create_app(
        dataset,
        windows,
        all_layers,
        bands,
        normalization,
        task_type,
        label_colors_dict,
        group_idx,
        label_layers,
        max_samples,
    )

    logger.info(f"Serving on http://{host}:{port}")
    logger.info(f"Open http://localhost:{port} in your browser")
    logger.info(
        f"Loaded {len(windows)} windows - refreshing the page will show a different random sample"
    )

    app.run(host=host, port=port, debug=False)


def parse_bands_arg(bands_str: str | None) -> dict[str, list[str]]:
    """Parse --bands argument as JSON.

    Args:
        bands_str: JSON string mapping layer_name -> list of band names, e.g. '{"sentinel2": ["B04", "B03", "B02"]}'

    Returns:
        Dictionary mapping layer_name -> list of band names
    """
    if not bands_str:
        return {}
    try:
        bands_dict = json.loads(bands_str)
        if not isinstance(bands_dict, dict):
            raise ValueError("Bands must be a JSON object/dictionary")
        for layer_name, band_list in bands_dict.items():
            if not isinstance(band_list, list):
                raise ValueError(f"Bands for layer '{layer_name}' must be a list")
            if not all(isinstance(band, str) for band in band_list):
                raise ValueError(f"All bands for layer '{layer_name}' must be strings")
        return bands_dict
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for bands: {e}") from e


def parse_normalization_arg(norm_str: str | None) -> dict[str, str]:
    """Parse --normalization argument as JSON.

    Args:
        norm_str: JSON string mapping layer_name -> normalization method, e.g. '{"sentinel2": "sentinel2_rgb"}'

    Returns:
        Dictionary mapping layer_name -> normalization method
    """
    if not norm_str:
        return {}
    try:
        norm_dict = json.loads(norm_str)
        if not isinstance(norm_dict, dict):
            raise ValueError("Normalization must be a JSON object/dictionary")
        for layer_name, method in norm_dict.items():
            if not isinstance(method, str):
                raise ValueError(
                    f"Normalization method for layer '{layer_name}' must be a string"
                )
        return norm_dict
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for normalization: {e}") from e


def main() -> None:
    """Main entry point for the visualization server CLI."""
    parser = argparse.ArgumentParser(
        description="Visualize rslearn dataset in a web browser"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to dataset directory (containing config.json)",
    )
    parser.add_argument(
        "--layers",
        type=str,
        required=True,
        nargs="+",
        help="List of layer names to visualize (default: all raster layers)",
    )
    parser.add_argument(
        "--label_layers",
        type=str,
        nargs="+",
        help="List of layer names that are labels (same format as --layers)",
    )
    parser.add_argument(
        "--bands",
        type=str,
        required=True,
        help='Bands to visualize per layer as JSON. Example: \'{"sentinel2": ["B04", "B03", "B02"]}\'',
    )
    parser.add_argument(
        "--normalization",
        type=str,
        required=True,
        help='Normalization method per layer as JSON. Example: \'{"sentinel2": "sentinel2_rgb"}\'',
    )
    parser.add_argument(
        "--task_type",
        type=str,
        required=True,
        choices=["classification", "regression", "detection", "segmentation"],
        help="Task type (default: auto-detect, but user should specify)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Maximum number of windows to sample",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to serve on (default: 8000)"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--group_idx", type=int, default=0, help="Item group index (default: 0)"
    )

    args = parser.parse_args()

    bands_dict = parse_bands_arg(args.bands)
    normalization_dict = parse_normalization_arg(args.normalization)

    run(
        dataset_path=args.dataset_path,
        layers=args.layers,
        bands=bands_dict if bands_dict else None,
        normalization=normalization_dict if normalization_dict else None,
        task_type=args.task_type,
        max_samples=args.max_samples,
        port=args.port,
        host=args.host,
        group_idx=args.group_idx,
        label_layers=args.label_layers,
    )


if __name__ == "__main__":
    main()
