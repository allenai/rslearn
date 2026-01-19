"""Visualization server for rslearn datasets.

This module provides a web server to visualize rslearn datasets, displaying
raster layers and ground truth labels for sampled windows.
"""

import argparse
import atexit
import http.server
import json
import random
import shutil
import socketserver
import tempfile
from datetime import datetime
from pathlib import Path

from rslearn.log_utils import get_logger
from rslearn.vis.utils import (
    detect_label_classes,
    find_geotiff_in_layer,
    generate_label_colors,
    generate_mask_from_geojson,
    process_window,
    visualize_tif,
)

logger = get_logger(__name__)

def get_window_dirs(dataset_path):
    """Get list of window directories from dataset path.

    Args:
        dataset_path: Path to windows directory (e.g., windows/sen12_landslides/)

    Returns:
        List of window directory paths
    """
    dataset_path = Path(dataset_path)
    window_dirs = []

    # Find all subdirectories that contain metadata.json
    for window_dir in dataset_path.iterdir():
        if window_dir.is_dir():
            metadata_path = window_dir / "metadata.json"
            if metadata_path.exists():
                window_dirs.append(window_dir)

    return sorted(window_dirs)


def sample_windows(window_dirs, max_samples):
    """Randomly sample windows.

    Args:
        window_dirs: List of window directory paths
        max_samples: Maximum number of samples to return

    Returns:
        List of sampled window directory paths
    """
    if max_samples is None or len(window_dirs) <= max_samples:
        return window_dirs

    return sorted(random.sample(window_dirs, max_samples))


def format_window_info(metadata):
    """Format window metadata for display.

    Args:
        metadata: Dictionary with window metadata

    Returns:
        Tuple of (formatted info HTML, lat, lon) for Google Maps link
    """
    if not metadata:
        return "Unknown", None, None
    parts = []
    lat = None
    lon = None
    if "options" in metadata:
        opts = metadata["options"]
        if "latitude" in opts and "longitude" in opts:
            lat = opts["latitude"]
            lon = opts["longitude"]
            parts.append(f"Lat: {lat:.4f}, Lon: {lon:.4f}")
        if "time_range_start" in opts and "time_range_end" in opts:
            start = opts["time_range_start"][:10]  # Just date part
            end = opts["time_range_end"][:10]
            parts.append(f"Time: {start} to {end}")
    return "<br>".join(parts), lat, lon


def generate_html(window_results, layers, output_dir, html_output_path, layer_order=None, label_colors=None):
    """Generate HTML gallery for visualization.

    Args:
        window_results: List of dictionaries with window processing results
        layers: Dictionary of layer configs (already filtered if --layers was specified)
        output_dir: Directory where PNGs are stored
        html_output_path: Path to save HTML file
        layer_order: Optional list of layer names specifying which layers to display and in what order
        label_colors: Dictionary mapping label class names to RGB color tuples
    """
    output_dir = Path(output_dir)
    html_output_path = Path(html_output_path)
    html_dir = html_output_path.parent

    # Get layer names from the provided layers dict (already filtered if --layers was specified)
    all_layer_names = list(layers.keys())
    
    if layer_order:
        # Use only the specified layers in the specified order
        # Only include layers that exist in the layers dict
        layer_names = [name for name in layer_order if name in all_layer_names]
    else:
        # Default: alphabetical order of all available layers
        layer_names = sorted(all_layer_names)

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
            border: 1px solid #ddd;
            border-radius: 4px;
            overflow: hidden;
            background: #fafafa;
        }
        .image-container img {
            width: 100%;
            height: auto;
            display: block;
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
    
    # Generate legend dynamically from label colors
    if label_colors:
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
        window_info, lat, lon = format_window_info(result.get("metadata"))
        
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
                # Calculate relative path from HTML file's directory
                try:
                    rel_path = img_path.relative_to(html_dir)
                    # Convert to string with forward slashes for URLs
                    rel_path_str = rel_path.as_posix()
                except ValueError:
                    # If paths are on different drives (Windows), fall back to absolute
                    # This shouldn't happen in normal usage
                    logger.warning(f"Could not compute relative path for {img_path} from {html_dir}")
                    rel_path_str = str(img_path)
                html_content += f"""
            <div class="image-container">
                <img src="{rel_path_str}" alt="{layer_name}">
                <div class="image-label">{layer_name}</div>
            </div>
"""
        # Add mask if available
        if result["mask_path"]:
            mask_path = Path(result["mask_path"])
            # Calculate relative path from HTML file's directory
            try:
                rel_path = mask_path.relative_to(html_dir)
                # Convert to string with forward slashes for URLs
                rel_path_str = rel_path.as_posix()
            except ValueError:
                # If paths are on different drives (Windows), fall back to absolute
                logger.warning(f"Could not compute relative path for {mask_path} from {html_dir}")
                rel_path_str = str(mask_path)
            html_content += f"""
            <div class="image-container">
                <img src="{rel_path_str}" alt="Label Mask">
                <div class="image-label">Label Mask</div>
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

    # Write HTML file
    html_output_path.parent.mkdir(parents=True, exist_ok=True)
    html_output_path.write_text(html_content, encoding="utf-8")
    logger.info(f"HTML gallery saved to {html_output_path}")


def main():
    """Main entry point for visualization server."""
    parser = argparse.ArgumentParser(description="Visualize rslearn dataset in a web browser")
    parser.add_argument("dataset_path", type=str, help="Path to dataset windows directory")
    parser.add_argument("config_path", type=str, help="Path to config.json file")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum number of windows to sample")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--normalize", type=str, default="sentinel2_rgb", choices=["sentinel2_rgb", "percentile", "minmax"], help="Normalization method")
    parser.add_argument("--layers", type=str, nargs="+", default=None, help="Specific layers to include in visualization, in the provided order (e.g., --layers pre_sentinel2 post_sentinel2). If not specified, all layers are shown.")
    parser.add_argument("--save_html", action="store_true", help="Save HTML file to outputs/{dataset_name}_{YYYYMMDD}.html")

    args = parser.parse_args()

    # Load config
    with open(args.config_path, "r") as f:
        config = json.load(f)

    layers = config.get("layers", {})
    # Exclude label layer from visualization layers
    all_visualization_layers = {k: v for k, v in layers.items() if k != "label"}
    
    # Filter to only specified layers if --layers is provided
    if args.layers:
        # Validate that all specified layers exist
        invalid_layers = [layer for layer in args.layers if layer not in all_visualization_layers]
        if invalid_layers:
            logger.error(f"Invalid layer names specified: {invalid_layers}")
            logger.error(f"Available layers: {sorted(all_visualization_layers.keys())}")
            raise ValueError(f"Invalid layer names: {invalid_layers}")
        
        # Filter to only specified layers, preserving the order
        visualization_layers = {k: all_visualization_layers[k] for k in args.layers if k in all_visualization_layers}
        logger.info(f"Filtering to {len(visualization_layers)} specified layers: {list(visualization_layers.keys())}")
    else:
        visualization_layers = all_visualization_layers

    # Get window directories
    window_dirs = get_window_dirs(args.dataset_path)
    logger.info(f"Found {len(window_dirs)} windows")

    # Detect label classes from GeoJSON files
    label_classes = detect_label_classes(window_dirs)
    label_colors = generate_label_colors(label_classes)
    if label_classes:
        logger.info(f"Detected {len(label_classes)} label classes: {sorted(label_classes)}")
        logger.info(f"Generated color mapping:")
        for label, color in sorted(label_colors.items()):
            logger.info(f"  {label}: RGB{color}")
    else:
        logger.info("No label classes detected")

    # Create output directory (persistent, not temporary, since server runs indefinitely)
    output_dir = Path(tempfile.mkdtemp(prefix="rslearn_vis_"))
    logger.info(f"Output directory: {output_dir}")

    # Register cleanup function
    def cleanup():
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
            logger.info(f"Cleaned up {output_dir}")

    atexit.register(cleanup)

    # Get expected layer names (only specified layers if --layers was provided)
    expected_layer_names = set(visualization_layers.keys())
    logger.info(f"Expecting {len(expected_layer_names)} layers: {sorted(expected_layer_names)}")

    # Process windows and filter to only keep those with all required layers
    window_results = []
    processed_count = 0
    skipped_count = 0
    
    # Shuffle windows for random sampling
    window_dirs_shuffled = list(window_dirs)
    random.shuffle(window_dirs_shuffled)
    
    for window_dir in window_dirs_shuffled:
        if len(window_results) >= args.max_samples:
            break
            
        try:
            result = process_window(window_dir, visualization_layers, output_dir, args.normalize, label_colors)
            processed_count += 1
            
            # Check if window has all required layers
            found_layer_names = set(result["layer_images"].keys())
            if found_layer_names == expected_layer_names:
                window_results.append(result)
                logger.debug(f"Accepted window {result['window_name']} with all {len(expected_layer_names)} layers")
            else:
                skipped_count += 1
                missing_layers = expected_layer_names - found_layer_names
                logger.debug(f"Skipped window {result['window_name']} - missing layers: {sorted(missing_layers)}")
        except Exception as e:
            processed_count += 1
            skipped_count += 1
            logger.error(f"Failed to process window {window_dir}: {e}")

    logger.info(f"Processed {processed_count} windows, kept {len(window_results)} with all layers, skipped {skipped_count}")
    
    if len(window_results) < args.max_samples:
        logger.warning(f"Only found {len(window_results)} windows with all required layers (requested {args.max_samples})")
    
    # Log image generation results for debugging
    total_images = sum(len(r.get("layer_images", {})) for r in window_results)
    total_masks = sum(1 for r in window_results if r.get("mask_path"))
    logger.info(f"Generated {total_images} layer images and {total_masks} masks")

    # Generate HTML
    html_path = output_dir / "index.html"
    # Pass the layer order (which layers to display and in what order)
    layer_order = args.layers if args.layers else None
    generate_html(window_results, visualization_layers, output_dir, html_path, layer_order=layer_order, label_colors=label_colors)
    
    # Save HTML to outputs directory if requested
    if args.save_html:
        # Extract dataset name from dataset path
        # The dataset_path points to the windows directory, so we need to go up to get dataset name
        dataset_path_obj = Path(args.dataset_path)
        # Extract meaningful name: use parent directory name or last part of path
        # e.g., .../sample_landslide/20260114_positives/windows/... -> sample_landslide_20260114_positives
        path_parts = dataset_path_obj.parts
        # Find "windows" in path and use the part before it
        dataset_name_parts = []
        for i, part in enumerate(path_parts):
            if part == "windows" and i > 0:
                # Get the directory before "windows"
                dataset_name_parts.append(path_parts[i - 1])
                if i > 1:
                    dataset_name_parts.insert(0, path_parts[i - 2])
                break
        if not dataset_name_parts:
            # Fallback: use last directory name before windows directory
            dataset_name_parts = [dataset_path_obj.parent.name]
        dataset_name = "_".join(dataset_name_parts).replace("/", "_")
        
        # Get current date in YYYYMMDD format
        date_str = datetime.now().strftime("%Y%m%d")
        
        # Create outputs directory if it doesn't exist
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)
        
        # Save HTML file
        saved_html_path = outputs_dir / f"{dataset_name}_{date_str}.html"
        
        # Copy the HTML file
        shutil.copy2(html_path, saved_html_path)
        logger.info(f"Saved HTML to {saved_html_path}")
        logger.info(f"Note: When opening the saved HTML directly, images may not load (they are in the temp directory)")
        logger.info(f"The server at http://{args.host}:{args.port} serves the complete visualization with images")

    # Start HTTP server
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(output_dir), **kwargs)

    try:
        with socketserver.TCPServer((args.host, args.port), Handler) as httpd:
            logger.info(f"Serving on http://{args.host}:{args.port}")
            logger.info(f"Open http://localhost:{args.port} in your browser")
            logger.info(f"Press Ctrl+C to stop the server")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                logger.info("\nShutting down server")
                cleanup()
    except OSError as e:
        if e.errno == 48:  # Address already in use
            logger.error(f"Port {args.port} is already in use. Please:")
            logger.error(f"  1. Stop the existing server on that port, or")
            logger.error(f"  2. Use a different port with --port <port_number>")
            cleanup()
            raise
        else:
            cleanup()
            raise


if __name__ == "__main__":
    main()


