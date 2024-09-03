from datetime import datetime, timedelta, timezone

import pytest
import shapely
from rasterio import CRS

from rslearn.const import WGS84_PROJECTION
from rslearn.utils import Projection, STGeometry


@pytest.fixture
def seattle2020() -> STGeometry:
    ts = datetime(2020, 7, 15, tzinfo=timezone.utc)
    time_range = (ts, ts + timedelta(days=30))
    wgs84_shp = shapely.Point(-122.33, 47.61)
    wgs84_geom = STGeometry(WGS84_PROJECTION, wgs84_shp, time_range)
    dst_projection = Projection(CRS.from_epsg(32610), 10, -10)
    dst_geom = wgs84_geom.to_projection(dst_projection)
    point = dst_geom.shp
    size = 64
    box = shapely.box(
        point.x - size // 2,
        point.y - size // 2,
        point.x + size // 2,
        point.y + size // 2,
    )
    return STGeometry(dst_projection, box, time_range)
