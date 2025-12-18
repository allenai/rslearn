"""Test the STAC client against Planetary Computer API."""

import json
from datetime import UTC, datetime

import shapely

from rslearn.data_sources.planetary_computer import Sentinel2
from rslearn.utils.stac import StacClient


def test_intersects() -> None:
    """Make sure if we give a point, all the items returned intersect that point."""
    client = StacClient(Sentinel2.STAC_ENDPOINT)
    point = shapely.Point(-122.3, 47.6)
    items = client.search(
        collections=[Sentinel2.COLLECTION_NAME],
        intersects=json.loads(shapely.to_geojson(point)),
        date_time=(datetime(2025, 3, 1, tzinfo=UTC), datetime(2025, 3, 10, tzinfo=UTC)),
    )
    print(f"test_intersects got {len(items)} results")
    assert len(items) > 0
    for item in items:
        assert item.geometry is not None
        shp = shapely.geometry.shape(item.geometry)
        assert shp.contains(point)


def test_query() -> None:
    """If we set query to restrict cloud cover, we should only get that cloud cover range."""
    client = StacClient(Sentinel2.STAC_ENDPOINT)
    items = client.search(
        collections=[Sentinel2.COLLECTION_NAME],
        date_time=(datetime(2025, 3, 1, tzinfo=UTC), datetime(2025, 3, 2, tzinfo=UTC)),
        query={
            "eo:cloud_cover": {"gt": 10, "lt": 20},
        },
    )
    print(f"test_query got {len(items)} results")
    assert len(items) > 0
    for item in items:
        cloud_cover = item.properties["eo:cloud_cover"]
        assert cloud_cover > 10 and cloud_cover < 20
