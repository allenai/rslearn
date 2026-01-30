"""Online integration tests for OlmoEarthDatasets data source.

These tests hit the live OlmoEarth Datasets API (no mocking).
The API currently only has 2025+ data ingested, so we use recent time ranges.
"""

import pathlib
from datetime import UTC, datetime, timedelta

import pytest
import shapely
from rasterio.crs import CRS
from upath import UPath

from rslearn.config import QueryConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.olmoearth_datasets import Sentinel2
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils.geometry import Projection, STGeometry


@pytest.fixture
def seattle2025() -> STGeometry:
    """Seattle geometry with January 2025 time range.

    OlmoEarth Datasets API only has 2025+ data, so we use this instead of
    the standard seattle2020 fixture. We can update this later.
    """
    ts = datetime(2025, 1, 1, tzinfo=UTC)
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


def test_sentinel2_ingest(tmp_path: pathlib.Path, seattle2025: STGeometry) -> None:
    """Test ingesting Sentinel-2 data via OlmoEarth Datasets API."""
    band_name = "B04"
    data_source = Sentinel2(assets=[band_name])

    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([seattle2025], query_config)[0]
    item = item_groups[0][0]

    tile_store_dir = UPath(tmp_path)
    tile_store = DefaultTileStore(str(tile_store_dir))
    tile_store.set_dataset_path(tile_store_dir)

    layer_name = "layer"
    data_source.ingest(
        TileStoreWithLayer(tile_store, layer_name), item_groups[0], [[seattle2025]]
    )
    assert tile_store.is_raster_ready(layer_name, item.name, [band_name])


def test_sentinel2_direct_materialize(seattle2025: STGeometry) -> None:
    """Test direct materialization (read_raster) for Sentinel-2 data."""
    data_source = Sentinel2(assets=["B04"])

    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([seattle2025], query_config)[0]
    item = item_groups[0][0]

    bounds = (
        int(seattle2025.shp.bounds[0]),
        int(seattle2025.shp.bounds[1]),
        int(seattle2025.shp.bounds[2]),
        int(seattle2025.shp.bounds[3]),
    )

    array = data_source.read_raster(
        layer_name="unused",
        item_name=item.name,
        bands=["B04"],
        projection=seattle2025.projection,
        bounds=bounds,
    )

    assert array.shape == (1, bounds[3] - bounds[1], bounds[2] - bounds[0])
    assert array.max() > 0
