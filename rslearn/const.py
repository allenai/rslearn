from rasterio.crs import CRS

from rslearn.utils import PixelBounds, Projection

WGS84_EPSG = 4326
WGS84_PROJECTION = Projection(CRS.from_epsg(WGS84_EPSG), 1, 1)
WGS84_BOUNDS: PixelBounds = (-180, -90, 180, 90)

TILE_SIZE = 512
