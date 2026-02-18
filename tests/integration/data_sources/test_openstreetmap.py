"""Mocked integration tests for the OpenStreetMap data source."""

import json
import pathlib
from datetime import UTC, datetime

import osmium
import pytest
import shapely
from upath import UPath

from rslearn.config import QueryConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.openstreetmap import FeatureType, Filter, OpenStreetMap
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils.geometry import STGeometry

# A small area for the test (subset of Seattle).
TEST_BOUNDS = (-122.34, 47.60, -122.32, 47.62)


@pytest.fixture
def test_pbf(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a minimal PBF file with a building in the test area."""
    fname = tmp_path / "test.osm.pbf"
    writer = osmium.SimpleWriter(str(fname))

    # Create 4 nodes forming a small building polygon.
    lon_min, lat_min, lon_max, lat_max = TEST_BOUNDS
    center_lon = (lon_min + lon_max) / 2
    center_lat = (lat_min + lat_max) / 2
    offset = 0.001

    node_locations = [
        (center_lon - offset, center_lat - offset),
        (center_lon + offset, center_lat - offset),
        (center_lon + offset, center_lat + offset),
        (center_lon - offset, center_lat + offset),
    ]

    for i, (lon, lat) in enumerate(node_locations, start=1):
        node = osmium.osm.mutable.Node(
            id=i,
            location=osmium.osm.Location(lon, lat),
        )
        writer.add_node(node)

    # Create a way (closed polygon) tagged as a building.
    # Use plain integer node IDs (osmium mutable Way accepts these directly).
    way = osmium.osm.mutable.Way(
        id=1,
        nodes=[1, 2, 3, 4, 1],
        tags={"building": "yes"},
    )
    writer.add_way(way)

    writer.close()
    return fname


def test_ingest(tmp_path: pathlib.Path, test_pbf: pathlib.Path) -> None:
    """Test OpenStreetMap ingest."""
    # The PBF should have a building at the center of TEST_BOUNDS.
    geometry = STGeometry(
        WGS84_PROJECTION,
        shapely.box(*TEST_BOUNDS),
        (datetime(2020, 7, 1, tzinfo=UTC), datetime(2020, 8, 1, tzinfo=UTC)),
    )

    tile_store_dir = UPath(tmp_path / "tiles")
    tile_store_dir.mkdir(parents=True, exist_ok=True)

    data_source = OpenStreetMap(
        pbf_fnames=[str(test_pbf)],
        bounds_fname=tile_store_dir / "bounds.json",
        categories={
            "building": Filter(
                [FeatureType.WAY, FeatureType.RELATION],
                tag_conditions={"building": []},
                to_geometry="Polygon",
                tag_properties={"building": "building"},
            ),
        },
    )

    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([geometry], query_config)[0]
    assert len(item_groups) > 0 and len(item_groups[0]) > 0
    item = item_groups[0][0]

    tile_store = DefaultTileStore(str(tile_store_dir))
    tile_store.set_dataset_path(tile_store_dir)
    layer_name = "layer"

    data_source.ingest(
        TileStoreWithLayer(tile_store, layer_name),
        item_groups[0],
        [[geometry]],
    )

    # Verify that vector data was written with exactly one building feature.
    expected_path = tile_store_dir / layer_name / item.name / "data.geojson"
    assert expected_path.exists()
    with expected_path.open() as f:
        geojson = json.load(f)
    assert len(geojson["features"]) == 1
    assert geojson["features"][0]["properties"]["category"] == "building"
