import os
import pathlib
import random
from datetime import UTC, datetime

import shapely
from upath import UPath

from rslearn.config import LayerType, QueryConfig, SpaceMode, VectorLayerConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.openstreetmap import FeatureType, Filter, OpenStreetMap
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils import STGeometry


class TestOpenStreetMap:
    """Tests the GEE data source."""

    def run_simple_test(self, tile_store_dir: UPath) -> None:
        """Apply test where we ingest an item corresponding to an area of Delaware."""
        # We use Delaware instead of seattle2020 here because delaware-latest.osm.pbf
        # is much smaller than washington-latest.osm.pbf so it speeds up the test.
        delaware_area = STGeometry(
            WGS84_PROJECTION,
            shapely.box(-75.6, 39.1, -75.5, 39.2),
            (
                datetime(2020, 7, 1, tzinfo=UTC),
                datetime(2020, 8, 1, tzinfo=UTC),
            ),
        )

        layer_config = VectorLayerConfig(
            LayerType.VECTOR,
        )
        query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
        # Is there a smaller area we can use?
        data_source = OpenStreetMap(
            config=layer_config,
            pbf_fnames=[
                UPath(
                    "https://download.geofabrik.de/north-america/us/delaware-latest.osm.pbf"
                )
            ],
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
        print("get items")
        item_groups = data_source.get_items([delaware_area], query_config)[0]
        item = item_groups[0][0]
        tile_store = DefaultTileStore(str(tile_store_dir))
        tile_store.set_dataset_path(tile_store_dir)
        layer_name = "layer"
        print("ingest")
        data_source.ingest(
            TileStoreWithLayer(tile_store, layer_name),
            item_groups[0],
            [[delaware_area]],
        )
        expected_path = tile_store_dir / layer_name / item.name / "data.geojson"
        assert expected_path.exists()

    def test_local(self, tmp_path: pathlib.Path) -> None:
        """Test ingesting to local filesystem."""
        tile_store_dir = UPath(tmp_path) / "tiles"
        tile_store_dir.mkdir(parents=True, exist_ok=True)
        self.run_simple_test(tile_store_dir)

    def test_gcs(self) -> None:
        """Test ingesting to GCS.

        Main thing is to test index_cache_dir being on GCS.
        """
        test_id = random.randint(10000, 99999)
        bucket_name = os.environ["TEST_BUCKET"]
        prefix = os.environ["TEST_PREFIX"] + f"test_{test_id}/"
        test_path = UPath(f"gcs://{bucket_name}/{prefix}")
        tile_store_dir = test_path / "tiles"
        self.run_simple_test(tile_store_dir)
