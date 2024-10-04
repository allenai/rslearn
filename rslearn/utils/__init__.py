"""rslearn utilities."""

import logging
import os

from .feature import Feature
from .geometry import (
    PixelBounds,
    Projection,
    STGeometry,
    is_same_resolution,
    shp_intersects,
)
from .get_utm_ups_crs import get_utm_ups_crs
from .grid_index import GridIndex
from .time import daterange
from .utils import open_atomic

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("RSLEARN_LOGLEVEL", "INFO").upper())

__all__ = (
    "Feature",
    "GridIndex",
    "PixelBounds",
    "Projection",
    "STGeometry",
    "daterange",
    "get_utm_ups_crs",
    "is_same_resolution",
    "logger",
    "open_atomic",
    "shp_intersects",
)
