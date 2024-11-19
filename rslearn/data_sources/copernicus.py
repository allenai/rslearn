"""Data source for raster data in ESA Copernicus API."""

import functools
import io
import json
import shutil
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.log_utils import get_logger
from rslearn.utils.fsspec import open_atomic
from rslearn.utils.geometry import STGeometry, flatten_shape
from rslearn.utils.grid_index import GridIndex

SENTINEL2_TILE_URL = "https://sentiwiki.copernicus.eu/__attachments/1692737/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.zip"
SENTINEL2_KML_NAMESPACE = "{http://www.opengis.net/kml/2.2}"

logger = get_logger(__name__)


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
        if el.text is None:
            raise ValueError(f"text is missing in {el}")
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

    def callback(array: npt.NDArray) -> npt.NDArray:
        return np.clip(array, -offset, None) + offset  # type: ignore

    return callback


def _cache_sentinel2_tile_index(cache_dir: UPath) -> None:
    """Cache the tiles from SENTINEL2_TILE_URL.

    This way we just need to download it once.
    """
    json_fname = cache_dir / "tile_index.json"

    if json_fname.exists():
        return

    logger.info(f"caching list of Sentinel-2 tiles to {json_fname}")

    # Identify the Sentinel-2 tile names and bounds using the KML file.
    # First, download the zip file and extract and parse the KML.
    buf = io.BytesIO()
    with urllib.request.urlopen(SENTINEL2_TILE_URL) as response:
        shutil.copyfileobj(response, buf)
    buf.seek(0)
    with zipfile.ZipFile(buf, "r") as zipf:
        member_names = zipf.namelist()
        if len(member_names) != 1:
            raise ValueError(
                "Sentinel-2 tile zip file unexpectedly contains more than one file"
            )

        with zipf.open(member_names[0]) as memberf:
            tree = ET.parse(memberf)

    # Map from the tile name to the longitude/latitude bounds.
    json_data: dict[str, tuple[float, float, float, float]] = {}

    # The KML is list of Placemark so iterate over those.
    for placemark_node in tree.iter(SENTINEL2_KML_NAMESPACE + "Placemark"):
        # The <name> node specifies the Sentinel-2 tile name.
        name_node = placemark_node.find(SENTINEL2_KML_NAMESPACE + "name")
        if name_node is None or name_node.text is None:
            raise ValueError("Sentinel-2 KML has Placemark without valid name node")

        tile_name = name_node.text

        # There may be one or more <coordinates> nodes depending on whether it is a
        # MultiGeometry. Here we just iterate over all of the coordinates since we are
        # only interested in the bounds in WGS-84 coordinates.
        lons = []
        lats = []
        for coord_node in placemark_node.iter(SENTINEL2_KML_NAMESPACE + "coordinates"):
            # It is list of space-separated coordinates like:
            #   180,-73.0597374076,0 176.8646237862,-72.9914734628,0 ...
            if coord_node.text is None:
                raise ValueError("Sentinel-2 KML has coordinates node missing text")

            point_strs = coord_node.text.strip().split()
            for point_str in point_strs:
                parts = point_str.split(",")
                if len(parts) != 2 and len(parts) != 3:
                    continue

                lon = float(parts[0])
                lat = float(parts[1])
                lons.append(lon)
                lats.append(lat)

        if len(lons) == 0 or len(lats) == 0:
            raise ValueError("Sentinel-2 KML has Placemark with no coordinates")

        bounds = (
            min(lons),
            min(lats),
            max(lons),
            max(lats),
        )
        json_data[tile_name] = bounds

    with open_atomic(json_fname, "w") as f:
        json.dump(json_data, f)


@functools.cache
def load_sentinel2_tile_index(cache_dir: UPath) -> GridIndex:
    """Load a GridIndex over Sentinel-2 tiles.

    This function is cached so the GridIndex only needs to be constructed once (per
    process).

    Args:
        cache_dir: the directory to cache the list of Sentinel-2 tiles.

    Returns:
        GridIndex over the tile names
    """
    _cache_sentinel2_tile_index(cache_dir)
    json_fname = cache_dir / "tile_index.json"
    with json_fname.open() as f:
        json_data = json.load(f)

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
    wgs84_geometry = geometry.to_projection(WGS84_PROJECTION)
    # If the shape is a collection, it could be cutting across prime meridian.
    # So we query each component shape separately and collect the results to avoid
    # issues.
    # We assume the caller has already applied split_at_prime_meridian.
    results = set()
    for shp in flatten_shape(wgs84_geometry.shp):
        for result in tile_index.query(shp.bounds):
            assert isinstance(result, str)
            results.add(result)
    return list(results)
