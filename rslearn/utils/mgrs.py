"""Utilities related to MGRS cells."""

from collections.abc import Generator

import numpy as np
from mgrs import MGRS


def for_each_cell(
    bounds: tuple[float, float, float, float],
) -> Generator[str, None, None]:
    """Yields each MGRS cell in the specified WGS84 bounds.

    Args:
        bounds: (minx, miny, maxx, maxy) in WGS84 coordinates
    """
    # Determine MGRS cell by converting to WGS84 and iterating over several
    # longitude/latitudes.
    # MGRS is at most 0.9-ish degrees (at equator) so we can iterate in 0.5
    # degree increments.
    mgrs_db = MGRS()
    for lon in np.arange(bounds[0], bounds[2], 0.5):
        for lat in np.arange(bounds[1], bounds[3], 0.5):
            yield mgrs_db.toMGRS(lat, lon, MGRSPrecision=0)
