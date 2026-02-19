import pathlib

from upath import UPath

from rslearn.config import (
    QueryConfig,
)
from rslearn.data_sources.climate_data_store import (
    ERA5LandHourly,
    ERA5LandHourlyTimeseries,
    ERA5LandMonthlyMeans,
)
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils import STGeometry


class TestERA5LandMonthlyMeans:
    """Tests the ERA5LandMonthlyMeans data source from the Climate Data Store."""

    TEST_BANDS = ["2m-temperature", "total-precipitation"]

    def test_local(self, tmp_path: pathlib.Path, seattle2020: STGeometry) -> None:
        """Apply test where we ingest an item corresponding to seattle2020."""
        query_config = QueryConfig(
            max_matches=2,  # We expect two items to match
        )
        data_source = ERA5LandMonthlyMeans(band_names=self.TEST_BANDS)
        print("get items")
        item_groups = data_source.get_items([seattle2020], query_config)[0]  # type: ignore
        item_0 = item_groups[0][0]
        item_1 = item_groups[1][0]

        tile_store_dir = UPath(tmp_path) / "tiles"
        tile_store_dir.mkdir(parents=True, exist_ok=True)
        tile_store = DefaultTileStore(str(tile_store_dir))
        tile_store.set_dataset_path(tile_store_dir)
        layer_name = "layer"

        print("ingest")
        data_source.ingest(
            TileStoreWithLayer(tile_store, layer_name), item_groups[0], [[seattle2020]]
        )
        data_source.ingest(
            TileStoreWithLayer(tile_store, layer_name), item_groups[1], [[seattle2020]]
        )
        assert tile_store.is_raster_ready(layer_name, item_0.name, self.TEST_BANDS)
        assert tile_store.is_raster_ready(layer_name, item_1.name, self.TEST_BANDS)


class TestERA5LandHourly:
    """Tests the ERA5LandHourly data source from the Climate Data Store."""

    TEST_BANDS = ["2m-temperature", "total-precipitation"]

    def test_local(self, tmp_path: pathlib.Path, seattle2020: STGeometry) -> None:
        """Apply test where we ingest an item corresponding to seattle2020."""
        query_config = QueryConfig(
            max_matches=2,  # We expect two items to match
        )
        data_source = ERA5LandHourly(
            band_names=self.TEST_BANDS, bounds=[-122.4, 47.6, -122.3, 47.7]
        )
        print("get items")
        item_groups = data_source.get_items([seattle2020], query_config)[0]  # type: ignore
        item_0 = item_groups[0][0]
        item_1 = item_groups[1][0]

        tile_store_dir = UPath(tmp_path) / "tiles"
        tile_store_dir.mkdir(parents=True, exist_ok=True)
        tile_store = DefaultTileStore(str(tile_store_dir))
        tile_store.set_dataset_path(tile_store_dir)
        layer_name = "layer"

        print("ingest")
        data_source.ingest(
            TileStoreWithLayer(tile_store, layer_name), item_groups[0], [[seattle2020]]
        )
        data_source.ingest(
            TileStoreWithLayer(tile_store, layer_name), item_groups[1], [[seattle2020]]
        )
        assert tile_store.is_raster_ready(layer_name, item_0.name, self.TEST_BANDS)
        assert tile_store.is_raster_ready(layer_name, item_1.name, self.TEST_BANDS)


class TestERA5LandHourlyTimeseries:
    """Tests the ERA5LandHourlyTimeseries data source from the Climate Data Store."""

    TEST_BANDS = ["2m-temperature", "total-precipitation"]

    def test_local(self, tmp_path: pathlib.Path, seattle2020: STGeometry) -> None:
        """Apply test where we ingest an item corresponding to seattle2020."""
        query_config = QueryConfig(
            max_matches=2,  # We expect two items to match
        )
        data_source = ERA5LandHourlyTimeseries(band_names=self.TEST_BANDS)
        print("get items")
        item_groups = data_source.get_items([seattle2020], query_config)[0]  # type: ignore
        item_0 = item_groups[0][0]
        item_1 = item_groups[1][0]

        # Verify items have point geometry (snapped to grid)
        assert item_0.geometry.shp.geom_type == "Point"
        assert item_1.geometry.shp.geom_type == "Point"

        tile_store_dir = UPath(tmp_path) / "tiles"
        tile_store_dir.mkdir(parents=True, exist_ok=True)
        tile_store = DefaultTileStore(str(tile_store_dir))
        tile_store.set_dataset_path(tile_store_dir)
        layer_name = "layer"

        print("ingest")
        data_source.ingest(
            TileStoreWithLayer(tile_store, layer_name), item_groups[0], [[seattle2020]]
        )
        data_source.ingest(
            TileStoreWithLayer(tile_store, layer_name), item_groups[1], [[seattle2020]]
        )
        assert tile_store.is_raster_ready(layer_name, item_0.name, self.TEST_BANDS)
        assert tile_store.is_raster_ready(layer_name, item_1.name, self.TEST_BANDS)

    def test_grid_snapping(self) -> None:
        """Test that coordinates are correctly snapped to 0.1 degree grid."""
        data_source = ERA5LandHourlyTimeseries(band_names=self.TEST_BANDS)

        snapped_lon, snapped_lat = data_source._snap_to_grid(-122.38, 47.62)
        assert snapped_lon == -122.4
        assert snapped_lat == 47.6

        snapped_lon, snapped_lat = data_source._snap_to_grid(-122.32, 47.67)
        assert snapped_lon == -122.3
        assert snapped_lat == 47.7
