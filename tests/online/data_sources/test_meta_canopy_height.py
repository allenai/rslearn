"""Online integration tests for the Meta CHMv2 data source (hits real S3)."""

import pathlib
from datetime import UTC, datetime

import shapely
from rasterio import CRS

from rslearn.config import QueryConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.meta_canopy_height import MetaCanopyHeightV2
from rslearn.utils import Projection, STGeometry


def _forest_geometry() -> STGeometry:
    """A small window over dense rainforest in the central Amazon (UTM 20S)."""
    ts = datetime(2020, 7, 15, tzinfo=UTC)
    wgs84_geom = STGeometry(WGS84_PROJECTION, shapely.Point(-62.2, -4.0), None)
    dst_projection = Projection(CRS.from_epsg(32720), 10, -10)
    point = wgs84_geom.to_projection(dst_projection).shp
    size = 64
    box = shapely.box(
        point.x - size // 2,
        point.y - size // 2,
        point.x + size // 2,
        point.y + size // 2,
    )
    return STGeometry(dst_projection, box, (ts, ts))


def test_read_raster(tmp_path: pathlib.Path) -> None:
    """Test direct materialization of CHMv2 canopy height over a forested area."""
    geometry = _forest_geometry()
    data_source = MetaCanopyHeightV2(metadata_cache_dir=str(tmp_path / "cache"))
    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([geometry], query_config)[0]
    assert len(item_groups) >= 1
    assert len(item_groups[0].items) >= 1
    item = item_groups[0].items[0]

    bounds = (
        int(geometry.shp.bounds[0]),
        int(geometry.shp.bounds[1]),
        int(geometry.shp.bounds[2]),
        int(geometry.shp.bounds[3]),
    )
    array = data_source.read_raster(
        layer_name="canopy_height",
        item=item,
        bands=["canopy_height"],
        projection=geometry.projection,
        bounds=bounds,
    )
    chw = array.get_chw_array()
    assert chw.shape[0] == 1
    assert chw.shape[1] == 64
    assert chw.shape[2] == 64
    # A forested window should have some non-zero canopy heights.
    assert chw.max() > 0
