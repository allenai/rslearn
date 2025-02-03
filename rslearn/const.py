"""Constants."""

from rslearn.utils.geometry import WGS84_BOUNDS, WGS84_EPSG, WGS84_PROJECTION

TILE_SIZE = 512
"""Default tile size. TODO: remove this or move it elsewhere."""

SHAPEFILE_AUX_EXTENSIONS = [".cpg", ".dbf", ".prj", ".sbn", ".sbx", ".shx", ".txt"]
"""Extensions of potential auxiliary files to .shp file."""

__all__ = (
    "WGS84_PROJECTION",
    "WGS84_EPSG",
    "WGS84_BOUNDS",
    "TILE_SIZE",
    "SHAPEFILE_AUX_EXTENSIONS",
)
