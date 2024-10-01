"""Data source for raster data in ESA Copernicus API."""

import functools
import io
import json
import shutil
import urllib
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.utils import STGeometry
from rslearn.utils.grid_index import GridIndex

SENTINEL2_TILE_URL = "https://sentiwiki.copernicus.eu/__attachments/1692737/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.zip"


def get_harmonize_callback(
    tree: ET.ElementTree,
) -> Callable[[npt.NDArray], npt.NDArray] | None:
    """Gets the harmonization callback based on the metadata XML.

    Harmonization ensures that scenes before and after processing baseline 04.00
    are comparable. 04.00 introduces +1000 offset to the pixel values to include
    more information about dark areas.

    Args:
        tree: the parsed XML tree

    Returns:
        None if no callback is needed, or the callback to subtract the new offset
    """
    offset = None
    for el in tree.iter("RADIO_ADD_OFFSET"):
        value = int(el.text)
        if offset is None:
            offset = value
            assert offset <= 0
            # For now assert the offset is always -1000.
            assert offset == -1000
        else:
            assert offset == value

    if offset is None or offset == 0:
        return None

    def callback(array):
        return np.clip(array, -offset, None) + offset

    return callback


@functools.cache
def load_sentinel2_tile_index(cache_dir: UPath) -> GridIndex:
    """Load a GridIndex over Sentinel-2 tiles."""
    json_fname = cache_dir / "tile_index.json"

    if not json_fname.exists():
        # Identify the Sentinel-2 tile names and bounds using the KML file.
        # First, download the zip file and extract and parse the KML.
        buf = io.BytesIO()
        with urllib.request.urlopen(SENTINEL2_TILE_URL) as response:
            shutil.copyfileobj(response, buf)
        buf.seek(0)
        with zipfile.ZipFile(buf, "r") as zipf:
            member_names = zipf.namelist()
            assert len(member_names) == 1
            with zipf.open(member_names[0]) as memberf:
                tree = ET.parse(memberf)

        # The KML is list of Placemark so iterate over those.
        namespace = "{http://www.opengis.net/kml/2.2}"
        json_data: dict[str, tuple[float, float, float, float]] = {}
        for placemark_node in tree.iter(namespace + "Placemark"):
            tile_name = placemark_node.find(namespace + "name").text
            bounds = None
            for coord_node in placemark_node.iter(namespace + "coordinates"):
                # It is list of space-separated coordinates like:
                #   180,-73.0597374076,0 176.8646237862,-72.9914734628,0 ...
                point_strs = coord_node.text.strip().split()
                for point_str in point_strs:
                    parts = point_str.split(",")
                    if len(parts) != 2 and len(parts) != 3:
                        continue
                    lon = float(parts[0])
                    lat = float(parts[1])
                    if bounds is None:
                        bounds = (lon, lat, lon, lat)
                    else:
                        bounds = (
                            min(bounds[0], lon),
                            min(bounds[1], lat),
                            max(bounds[2], lon),
                            max(bounds[3], lat),
                        )

            json_data[tile_name] = bounds

        with json_fname.open("w") as f:
            json.dump(json_data, f)

    else:
        with json_fname.open() as f:
            json_data = json.load(f)

    # Now we can populate the grid index.
    grid_index = GridIndex(0.5)
    for tile_name, bounds in json_data.items():
        grid_index.insert(bounds, tile_name)

    return grid_index


def get_sentinel2_tiles(geometry: STGeometry, cache_dir: UPath) -> list[str]:
    """Get all Sentinel-2 tiles (like 01CCV) intersecting the given geometry.

    Args:
        geometry: the geometry to check.
        cache_dir: directory to cache the tiles.

    Returns:
        list of Sentinel-2 tile names that intersect the geometry.
    """
    tile_index = load_sentinel2_tile_index(cache_dir)
    wgs84_bounds = geometry.to_projection(WGS84_PROJECTION).shp.bounds
    return tile_index.query(wgs84_bounds)
