"""Dataset-specific visualization server implementations.

This module provides example implementations of dataset-specific visualization
servers that inherit from VisualizationServer and customize behavior for
specific datasets.
"""

from rslearn.log_utils import get_logger
from rslearn.vis.server import VisualizationServer

logger = get_logger(__name__)


class KenyaCropVisualizationServer(VisualizationServer):
    """Visualization server customized for Kenya/Nandi datasets.
    
    This class handles the Kenya dataset's metadata format where coordinates
    are embedded in the window name.
    """
    
    layers = ["sentinel2"]    
    task_type = "classification"    
    normalization = {"sentinel2": "sentinel2_rgb"}
    bands = {"sentinel2": [3, 2, 1]}  # B04, B03, B02 for RGB (Kenya dataset has bands in order: B02, B03, B04, B08)
    
    def format_window_info(self, metadata):
        """Format window metadata for Kenya dataset.
        
        Kenya dataset metadata has:
        - time_range at top level
        - Coordinates in window name (format: ID_LON_LAT)
        """
        if not metadata:
            return "Unknown", None, None
        parts = []
        lat = None
        lon = None
        
        # Handle standard rslearn format: time_range at top level
        if "time_range" in metadata and metadata["time_range"]:
            start = metadata["time_range"][0][:10]  # Just date part
            end = metadata["time_range"][1][:10]
            parts.append(f"Time: {start} to {end}")
        
        # Extract lat/lon from window name (format: ID_LON_LAT)
        if "name" in metadata:
            name = metadata["name"]
            # Try to extract coordinates from name (e.g., "15_0.3788750002417029_35.034375000140685")
            parts_split = name.split("_")
            if len(parts_split) >= 3:
                try:
                    # Kenya format: ID_LON_LAT
                    potential_lon = float(parts_split[-2])
                    potential_lat = float(parts_split[-1])
                    # Validate reasonable lat/lon ranges
                    if -180 <= potential_lon <= 180 and -90 <= potential_lat <= 90:
                        lon = potential_lon
                        lat = potential_lat
                        parts.insert(0, f"Lat: {lat:.4f}, Lon: {lon:.4f}")
                except (ValueError, IndexError):
                    pass
        
        return "<br>".join(parts) if parts else "Unknown", lat, lon


class LandslideVisualizationServer(VisualizationServer):
    """Visualization server customized for landslide datasets.
    
    This class handles the landslide dataset's metadata format where coordinates
    and time ranges are in the options dictionary.
    """
    
    layers = ["pre_sentinel2", "post_sentinel2"]    
    task_type = "segmentation"    
    normalization = {"pre_sentinel2": "sentinel2_rgb", "post_sentinel2": "sentinel2_rgb"}
    bands = {"pre_sentinel2": [4, 3, 2], "post_sentinel2": [4, 3, 2]}  # B04, B03, B02 for RGB
    
    def format_window_info(self, metadata):
        """Format window metadata for landslide dataset.
        
        Landslide dataset metadata has:
        - latitude/longitude in options
        - time_range_start/time_range_end in options
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
        
        return "<br>".join(parts) if parts else "Unknown", lat, lon

