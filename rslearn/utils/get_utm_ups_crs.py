"""Utilities for working with UTM/UPS projections."""

import pyproj.aoi
import pyproj.database
from rasterio.crs import CRS

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
