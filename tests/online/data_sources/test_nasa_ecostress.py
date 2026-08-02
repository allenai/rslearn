"""Online test for the NASA ECOSTRESS L2T LSTE data source (hits real LP DAAC).

Requires an Earthdata bearer token in the ``EARTHDATA_TOKEN`` environment variable;
the test is skipped otherwise.
"""

import os
from datetime import UTC, datetime

import numpy as np
import pytest
import shapely
from rasterio import CRS

from rslearn.config import QueryConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.nasa_ecostress import EcostressLSTE
from rslearn.utils import Projection, STGeometry

pytestmark = pytest.mark.skipif(
    os.environ.get("EARTHDATA_TOKEN") is None,
    reason="EARTHDATA_TOKEN is required for ECOSTRESS online test",
)


def _boulder_geometry() -> STGeometry:
    """A small window over Boulder, CO in UTM 13N."""
    wgs84_geom = STGeometry(WGS84_PROJECTION, shapely.Point(-105.24, 40.02), None)
    dst_projection = Projection(CRS.from_epsg(32613), 70, -70)
    point = wgs84_geom.to_projection(dst_projection).shp
    size = 32
    box = shapely.box(
        point.x - size // 2,
        point.y - size // 2,
        point.x + size // 2,
        point.y + size // 2,
    )
    time_range = (
        datetime(2023, 7, 1, tzinfo=UTC),
        datetime(2023, 8, 1, tzinfo=UTC),
    )
    return STGeometry(dst_projection, box, time_range)


def test_read_lst() -> None:
    geometry = _boulder_geometry()
    data_source = EcostressLSTE(band_names=["LST"])
    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS, max_matches=1)
    item_groups = data_source.get_items([geometry], query_config)[0]
    assert len(item_groups) >= 1
    item = item_groups[0].items[0]

    bounds = (
        int(geometry.shp.bounds[0]),
        int(geometry.shp.bounds[1]),
        int(geometry.shp.bounds[2]),
        int(geometry.shp.bounds[3]),
    )
    array = data_source.read_raster(
        layer_name="ecostress",
        item=item,
        bands=["LST"],
        projection=geometry.projection,
        bounds=bounds,
    )
    chw = array.get_chw_array()
    assert chw.shape[0] == 1
    # LST is float32 Kelvin; valid land surface temperatures are well above 0 K.
    valid = chw[np.isfinite(chw)]
    assert valid.size > 0
    assert valid.max() > 200
