"""Visualization server for rslearn datasets.

This module provides a web server to visualize rslearn datasets using the Dataset/Window APIs.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any

from flask import Flask, Response
from flask import render_template as flask_render_template
from PIL import Image
from upath import UPath

from rslearn.dataset import Dataset, Window
from rslearn.dataset.window import get_layer_and_group_from_dir_name
from rslearn.log_utils import get_logger

from .render_raster import (
    read_raster_layer,
    render_raster,
)
from .render_vector_image import render_vector_image
from .render_vector_text import render_vector_text
from .utils import (
    array_to_bytes,
    format_window_info,
    generate_label_colors_for_layer,
)

logger = get_logger(__name__)


def prepare_visualization_data(
    sampled_windows: list[Window],
    dataset: Dataset,
    raster_groups: list[str],
    vector_text_groups: list[str],
    vector_text_render: dict[str, dict[str, Any]],
    vector_image_groups: list[str],
    label_colors_dict: dict[str, dict[str, tuple[int, int, int]]],
) -> dict[str, Any]:
    """Prepare data for visualization template.

    Args:
        sampled_windows: List of windows to display
        dataset: Dataset object
        raster_groups: List of raster item group names
        vector_text_groups: List of vector text item group names
        vector_text_render: Dictionary mapping item_group_name -> render spec dict
        vector_image_groups: List of vector image item group names
        label_colors_dict: Dictionary mapping item_group_name -> label_colors

    Returns:
        Dictionary with template context data
    """
    window_data: list[dict[str, Any]] = []
    for idx, window in enumerate(sampled_windows):
        time_range, lat, lon = format_window_info(window)
        maps_link = (
            f"https://www.google.com/maps?q={lat},{lon}"
            if lat is not None and lon is not None
            else None
        )

        available_raster_groups: set[str] = set()
        available_vector_image_groups: list[str] = []
        label_texts: dict[str, str] = {}

        for item_group_name in raster_groups:
            layer_name, group_idx = get_layer_and_group_from_dir_name(item_group_name)
            if layer_name not in dataset.layers:
                continue
            if window.is_layer_completed(layer_name, group_idx=group_idx):
                available_raster_groups.add(item_group_name)

        for item_group_name in vector_text_groups:
            layer_name, group_idx = get_layer_and_group_from_dir_name(item_group_name)
            if layer_name not in dataset.layers:
                continue
            layer_config = dataset.layers[layer_name]
            render_spec = vector_text_render[item_group_name]
            try:
                result = render_vector_text(
                    window,
                    layer_name,
                    layer_config,
                    render_spec,
                    group_idx=group_idx,
                )
                if result is not None:
                    label_texts[item_group_name] = result
            except Exception as e:
                logger.debug(
                    f"Error processing group {item_group_name} for window {window.name}: {e}"
                )
                continue

        for item_group_name in vector_image_groups:
            layer_name, group_idx = get_layer_and_group_from_dir_name(item_group_name)
            if layer_name not in dataset.layers:
                continue
            if window.is_layer_completed(layer_name, group_idx=group_idx):
                available_vector_image_groups.append(item_group_name)

        # Format time range for template
        time_range_formatted = None
        if time_range:
            time_range_formatted = (
                time_range[0].isoformat()[:10],
                time_range[1].isoformat()[:10],
            )

        window_data.append(
            {
                "idx": idx,
                "name": window.name,
                "time_range": time_range,
                "time_range_formatted": time_range_formatted,
                "lat": lat,
                "lon": lon,
                "maps_link": maps_link,
                "available_raster_groups": available_raster_groups,
                "available_vector_image_groups": available_vector_image_groups,
                "label_texts": label_texts,
            }
        )

    per_layer_legends: dict[str, dict[str, tuple[int, int, int]]] = {}
    for item_group_name, colors in label_colors_dict.items():
        if colors:
            per_layer_legends[item_group_name] = colors

    return {
        "windows": window_data,
        "raster_groups": raster_groups,
        "per_layer_legends": per_layer_legends,
    }


def create_app(
    dataset: Dataset,
    windows: list[Window],
    raster_groups: list[str],
    vector_text_groups: list[str],
    vector_image_groups: list[str],
    bands: dict[str, list[str]],
    raster_render: dict[str, dict[str, Any]],
    vector_text_render: dict[str, dict[str, Any]],
    vector_image_render: dict[str, dict[str, Any]],
    label_colors_dict: dict[str, dict[str, tuple[int, int, int]]],
    resampling: dict[str, Image.Resampling],
    max_samples: int,
) -> Flask:
    """Create and configure Flask app.

    Args:
        dataset: Dataset object
        windows: List of all windows
        raster_groups: List of raster item group names
        vector_text_groups: List of vector text item group names
        vector_image_groups: List of vector image item group names
        bands: Dictionary mapping item_group_name -> list of band names
        raster_render: Dictionary mapping item_group_name -> render spec dict
        vector_text_render: Dictionary mapping item_group_name -> render spec dict
        vector_image_render: Dictionary mapping item_group_name -> render spec dict
        label_colors_dict: Dictionary mapping item_group_name -> label_colors
        resampling: Dictionary mapping item_group_name -> PIL resampling method
        max_samples: Maximum number of windows to sample

    Returns:
        Configured Flask app
    """
    # Set template folder explicitly to ensure Flask can find templates
    template_folder = Path(__file__).parent / "templates"
    app = Flask(__name__, template_folder=str(template_folder))

    if len(windows) > max_samples:
        sampled_windows = random.sample(windows, max_samples)
    else:
        sampled_windows = list(windows)

    @app.route("/")
    def index() -> str:
        """Render the main visualization page."""
        template_data = prepare_visualization_data(
            sampled_windows,
            dataset,
            raster_groups,
            vector_text_groups,
            vector_text_render,
            vector_image_groups,
            label_colors_dict,
        )
        return flask_render_template("visualization.html", **template_data)

    @app.route("/images/<int:window_idx>/<item_group_name>")
    def get_image(window_idx: int, item_group_name: str) -> Response:
        """Generate and serve an image for a specific window/group.

        Args:
            window_idx: Index of the window in the sampled windows list
            item_group_name: Item group name to visualize

        Returns:
            PNG image response or error response
        """
        if window_idx < 0 or window_idx >= len(sampled_windows):
            return Response(
                "Window index out of range", status=404, mimetype="text/plain"
            )

        window = sampled_windows[window_idx]

        layer_name, group_idx = get_layer_and_group_from_dir_name(item_group_name)
        if layer_name not in dataset.layers:
            return Response(
                f"Unknown layer: {layer_name}", status=404, mimetype="text/plain"
            )
        layer_config = dataset.layers[layer_name]

        if item_group_name in raster_groups:
            array = read_raster_layer(
                window,
                layer_name,
                layer_config,
                bands[item_group_name],
                group_idx=group_idx,
            )
            image_array = render_raster(
                array,
                layer_config,
                raster_render[item_group_name],
                label_colors=label_colors_dict.get(item_group_name),
            )
            layer_resampling = resampling.get(item_group_name, Image.Resampling.NEAREST)
            image_bytes = array_to_bytes(image_array, resampling=layer_resampling)
        elif item_group_name in vector_image_groups:
            image_array = render_vector_image(
                window,
                layer_name,
                layer_config,
                vector_image_render[item_group_name],
                label_colors_dict.get(item_group_name, {}),
                dataset=dataset,
                group_idx=group_idx,
                bands=bands,
                raster_render=raster_render,
            )
            layer_resampling = resampling.get(item_group_name, Image.Resampling.NEAREST)
            image_bytes = array_to_bytes(image_array, resampling=layer_resampling)
        else:
            return Response(
                f"Unknown item group name: {item_group_name}",
                status=404,
                mimetype="text/plain",
            )

        return Response(
            image_bytes,
            mimetype="image/png",
            headers={"Content-Length": str(len(image_bytes))},
        )

    return app


def run(
    dataset_path: str | Path | UPath,
    raster_groups: list[str] | None = None,
    bands: dict[str, list[str]] | None = None,
    raster_render: dict[str, dict[str, Any]] | None = None,
    vector_text_groups: list[str] | None = None,
    vector_text_render: dict[str, dict[str, Any]] | None = None,
    vector_image_groups: list[str] | None = None,
    vector_image_render: dict[str, dict[str, Any]] | None = None,
    resampling: dict[str, str] | None = None,
    max_samples: int = 100,
    port: int = 8000,
    host: str = "0.0.0.0",
    groups: list[str] | None = None,
) -> None:
    """Run the visualization server.

    Args:
        dataset_path: Path to dataset directory (containing config.json)
        raster_groups: List of raster item group names to visualize
            (e.g. ["sentinel1", "sentinel1.1", "label"])
        bands: Dictionary mapping item_group_name -> list of band names
        raster_render: Dictionary mapping item_group_name -> render spec dict
        vector_text_groups: List of vector text item group names to visualize
        vector_text_render: Dictionary mapping item_group_name -> render spec dict
        vector_image_groups: List of vector image item group names to visualize
        vector_image_render: Dictionary mapping item_group_name -> render spec dict
        resampling: Optional dictionary mapping item_group_name -> resampling method
            name (e.g. "nearest", "bilinear", "bicubic"). Defaults to "nearest".
        max_samples: Maximum number of windows to sample
        port: Port to serve on
        host: Host to bind to
        groups: Optional list of window group names to load (e.g. ["predict"]).
            If not set, all groups under windows/ are loaded.
    """
    dataset_path = UPath(dataset_path)
    dataset = Dataset(dataset_path)

    raster_groups = raster_groups or []
    vector_text_groups = vector_text_groups or []
    vector_image_groups = vector_image_groups or []
    bands = bands or {}
    raster_render = raster_render or {}
    vector_text_render = vector_text_render or {}
    vector_image_render = vector_image_render or {}

    resampling_str_map = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    resampling_resolved: dict[str, Image.Resampling] = {}
    for name, method_str in (resampling or {}).items():
        method = resampling_str_map.get(method_str)
        if method is None:
            raise ValueError(
                f"Unknown resampling method '{method_str}' for '{name}'. "
                f"Valid options: {', '.join(resampling_str_map)}"
            )
        resampling_resolved[name] = method

    for item_group_name in raster_groups:
        if item_group_name not in bands:
            raise ValueError(
                f"Bands not specified for raster group '{item_group_name}'. "
                f"Provide --bands with an entry for '{item_group_name}'."
            )
        if item_group_name not in raster_render:
            raise ValueError(
                f"Render method not specified for raster group '{item_group_name}'. "
                f"Provide --raster_render with an entry for '{item_group_name}'."
            )

    for item_group_name in vector_text_groups:
        if item_group_name not in vector_text_render:
            raise ValueError(
                f"Render method not specified for vector text group '{item_group_name}'. "
                f"Provide --vector_text_render with an entry for '{item_group_name}'."
            )

    for item_group_name in vector_image_groups:
        if item_group_name not in vector_image_render:
            raise ValueError(
                f"Render method not specified for vector image group '{item_group_name}'. "
                f"Provide --vector_image_render with an entry for '{item_group_name}'."
            )

    if not raster_groups and not vector_text_groups and not vector_image_groups:
        raise ValueError(
            "At least one of --raster_groups, --vector_text_groups, "
            "or --vector_image_groups is required"
        )

    label_colors_dict: dict[str, dict[str, tuple[int, int, int]]] = {}

    for item_group_name in raster_groups + vector_image_groups:
        layer_name, _ = get_layer_and_group_from_dir_name(item_group_name)
        layer_config = dataset.layers[layer_name]
        colors = generate_label_colors_for_layer(layer_config)
        if colors:
            label_colors_dict[item_group_name] = colors

    logger.info(f"Loading all windows from dataset {dataset_path}")
    windows = dataset.load_windows(groups=groups, workers=128, show_progress=True)
    logger.info(f"Loaded {len(windows)} windows from dataset")
    logger.info(f"Raster groups: {raster_groups}")
    logger.info(f"Vector text groups: {vector_text_groups}")
    logger.info(f"Vector image groups: {vector_image_groups}")
    logger.info(f"Bands: {bands}")
    logger.info(f"Raster render: {raster_render}")
    logger.info(f"Vector text render: {vector_text_render}")
    logger.info(f"Vector image render: {vector_image_render}")

    app = create_app(
        dataset,
        windows,
        raster_groups,
        vector_text_groups,
        vector_image_groups,
        bands,
        raster_render,
        vector_text_render,
        vector_image_render,
        label_colors_dict,
        resampling_resolved,
        max_samples,
    )

    logger.info(f"Serving on http://{host}:{port}")
    logger.info(f"Open http://localhost:{port} in your browser")
    logger.info(
        f"Loaded {len(windows)} windows, sampled {min(len(windows), max_samples)} for display"
    )

    app.run(host=host, port=port, debug=False)


def parse_json_dict_arg(arg_str: str | None, arg_name: str) -> dict:
    """Parse a JSON dict CLI argument.

    Args:
        arg_str: JSON string to parse
        arg_name: Name of the argument (for error messages)

    Returns:
        Parsed dictionary
    """
    if not arg_str:
        return {}
    try:
        d = json.loads(arg_str)
        if not isinstance(d, dict):
            raise ValueError(f"{arg_name} must be a JSON object/dictionary")
        return d
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for {arg_name}: {e}") from e


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
        "--raster_groups",
        type=str,
        nargs="+",
        help=(
            "List of raster item group names to visualize. "
            "E.g. 'sentinel1' (group 0) or 'sentinel1.1' (group 1)."
        ),
    )
    parser.add_argument(
        "--bands",
        type=str,
        help=(
            "Bands per raster group as JSON. "
            'Example: \'{"sentinel2": ["B04", "B03", "B02"], "sentinel2.1": ["B04", "B03", "B02"]}\''
        ),
    )
    parser.add_argument(
        "--raster_render",
        type=str,
        help=(
            "Render method per raster group as JSON. Each value is a dict with 'name' and optional 'args'. "
            "Methods: sentinel2_rgb, percentile, minmax, linear, classes. "
            'Example: \'{"sentinel2": {"name": "sentinel2_rgb"}, "elevation": {"name": "linear", "args": {"vmin": 0, "vmax": 3000}}}\''
        ),
    )
    parser.add_argument(
        "--vector_text_groups",
        type=str,
        nargs="+",
        help="List of vector item group names to render as text",
    )
    parser.add_argument(
        "--vector_text_render",
        type=str,
        help=(
            "Render method per vector text group as JSON. "
            "Methods: text, property. "
            'Example: \'{"labels": {"name": "text"}}\''
        ),
    )
    parser.add_argument(
        "--vector_image_groups",
        type=str,
        nargs="+",
        help="List of vector item group names to render as images",
    )
    parser.add_argument(
        "--vector_image_render",
        type=str,
        help=(
            "Render method per vector image group as JSON. "
            "Methods: detection, segmentation. "
            'Example: \'{"annotations": {"name": "detection"}}\''
        ),
    )
    parser.add_argument(
        "--resampling",
        type=str,
        help=(
            "Resampling method per group as JSON. Optional per-layer, defaults to nearest. "
            "Methods: nearest, bilinear, bicubic, lanczos. "
            'Example: \'{"sentinel2": "bilinear", "label": "nearest"}\''
        ),
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
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--groups",
        type=str,
        nargs="+",
        default=None,
        help="Window group(s) to load (e.g. predict). If not set, all groups under windows/ are loaded.",
    )

    args = parser.parse_args()

    bands_dict = parse_json_dict_arg(args.bands, "--bands")
    raster_render_dict = parse_json_dict_arg(args.raster_render, "--raster_render")
    vector_text_render_dict = parse_json_dict_arg(
        args.vector_text_render, "--vector_text_render"
    )
    vector_image_render_dict = parse_json_dict_arg(
        args.vector_image_render, "--vector_image_render"
    )
    resampling_dict = parse_json_dict_arg(args.resampling, "--resampling")

    run(
        dataset_path=args.dataset_path,
        raster_groups=args.raster_groups,
        bands=bands_dict or None,
        raster_render=raster_render_dict or None,
        vector_text_groups=args.vector_text_groups,
        vector_text_render=vector_text_render_dict or None,
        vector_image_groups=args.vector_image_groups,
        vector_image_render=vector_image_render_dict or None,
        resampling=resampling_dict or None,
        max_samples=args.max_samples,
        port=args.port,
        host=args.host,
        groups=args.groups,
    )


if __name__ == "__main__":
    main()
