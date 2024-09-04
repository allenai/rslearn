"""Constants."""

from rasterio.crs import CRS

from rslearn.utils import PixelBounds, Projection

WGS84_EPSG = 4326
"""The EPSG code for WGS-84."""

WGS84_PROJECTION = Projection(CRS.from_epsg(WGS84_EPSG), 1, 1)
"""The Projection for WGS-84 assuming 1 degree per pixel.

This can be used to create STGeometry with shapes in longitude/latitude coordinates.
"""

WGS84_BOUNDS: PixelBounds = (-180, -90, 180, 90)
"""The bounds of the WGS-84 projection."""

TILE_SIZE = 512
"""Default tile size. TODO: remove this or move it elsewhere."""

SHAPEFILE_AUX_EXTENSIONS = [".cpg", ".dbf", ".prj", ".sbn", ".sbx", ".shx", ".txt"]
"""Extensions of potential auxiliary files to .shp file."""
