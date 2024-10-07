"""Utilities for working with UTM/UPS projections."""

import pyproj.aoi
import pyproj.database
import shapely
from rasterio.crs import CRS

from rslearn.utils import Projection, STGeometry

UPS_NORTH_EPSG = 5041
"""EPSG code for the UPS North CRS."""

UPS_SOUTH_EPSG = 5042
"""EPSG code for the UPS South CRS."""

EPSILON = 1e-4

UPS_NORTH_THRESHOLD = 84 - EPSILON
"""Use UPS North for latitudes north of this threshold."""

UPS_SOUTH_THRESHOLD = -80 + EPSILON
"""Use UPS South for latitudes south of this threshold."""


def get_utm_ups_crs(lon: float, lat: float) -> CRS:
    """Get the appropriate UTM or UPS CRS for a given lon/lat.

    Args:
        lon: longitude in degrees
        lat: latitude in degrees

    Returns:
        the rasterio CRS for the appropriate UTM or UPS zone
    """
    if lat > UPS_NORTH_THRESHOLD:
        return CRS.from_epsg(UPS_NORTH_EPSG)
    if lat < UPS_SOUTH_THRESHOLD:
        return CRS.from_epsg(UPS_SOUTH_EPSG)

    utm_crs_list = pyproj.database.query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=pyproj.aoi.AreaOfInterest(
            west_lon_degree=lon,
            south_lat_degree=lat,
            east_lon_degree=lon,
            north_lat_degree=lat,
        ),
    )
    if len(utm_crs_list) == 0:
        raise ValueError(f"Could not find UTM zone for lon={lon}, lat={lat}")
    utm_crs = utm_crs_list[0]
    return CRS.from_epsg(utm_crs.code)


def get_utm_ups_projection(
    lon: float, lat: float, x_resolution: float, y_resolution: float
) -> Projection:
    """Get the appropriate UTM or UPS Projection for a given lon/lat.

    Args:
        lon: longitude in degrees
        lat: latitude in degrees
        x_resolution: desired x resolution in meters per pixel
        y_resolution: desired y resolution in meters per pixel

    Returns:
    the Projection object
    """
    crs = get_utm_ups_crs(lon, lat)
    return Projection(crs, x_resolution, y_resolution)


def get_utm_zone_info(utm_crs: CRS) -> tuple[int, str]:
    """Get UTM zone number (1 to 60) and S/N from CRS.

    Args:
        utm_crs: the UTM CRS.

    Returns:
        tuple of (utm_zone, "S" or "N")
    """
    assert utm_crs.is_epsg_code
    epsg_code = utm_crs.to_epsg()
    if epsg_code > 32600 and epsg_code <= 32660:
        return (epsg_code - 32600, "N")
    elif epsg_code > 32700 and epsg_code <= 32760:
        return (epsg_code - 32700, "S")
    raise ValueError(f"EPSG code {epsg_code} is not UTM")


def get_wgs84_bounds(utm_crs: CRS) -> tuple[int, int, int, int]:
    """Returns the bounding longitude and latitude of this UTM zone.

    Args:
        utm_crs: the UTM CRS.

    Returns:
        tuple (min_lon, min_lat, max_lon, max_lat)
    """
    utm_zone, hemisphere = get_utm_zone_info(utm_crs)
    if hemisphere == "S":
        min_lat = -80
        max_lat = 0
    elif hemisphere == "N":
        min_lat = 0
        max_lat = 84
    # 1N/S is -180 to -174 and goes up from there
    min_lon = -180 + 6 * (utm_zone - 1)
    max_lon = -180 + 6 * utm_zone
    return (min_lon, min_lat, max_lon, max_lat)


def get_proj_bounds(utm_crs: CRS) -> tuple[float, float, float, float]:
    """Returns projection bounds of a UTM zone.

    Args:
        utm_crs: the UTM CRS.

    Returns:
        tuple (min_x, min_y, max_x, max_y)
    """
    bounds = get_wgs84_bounds(utm_crs)
    # Convert from WGS84 to the UTM zone.
    src_proj = Projection(CRS.from_epsg(4326), 1, 1)
    dst_proj = Projection(utm_crs, 1, 1)
    shp = shapely.box(*bounds)
    result = STGeometry(src_proj, shp, None).to_projection(dst_proj).shp
    return result.bounds
