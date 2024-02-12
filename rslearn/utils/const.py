from rasterio.crs import CRS

from .geometry import Projection

WGS84_EPSG = 4326
WGS84_PROJECTION = Projection(CRS.from_epsg(WGS84_EPSG), 1)
