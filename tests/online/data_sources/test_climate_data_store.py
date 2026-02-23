import pathlib
from datetime import UTC, datetime, timedelta

from rasterio.crs import CRS
from upath import UPath

from rslearn.config import (
    QueryConfig,
)
from rslearn.const import WGS84_EPSG
from rslearn.data_sources.climate_data_store import (
    ERA5LandHourly,
    ERA5LandHourlyTimeseries,
    ERA5LandMonthlyMeans,
)
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils import Projection, STGeometry


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
        """Apply test where we ingest items and verify timestamps span each month's hours."""
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
        ts = TileStoreWithLayer(tile_store, layer_name)
        data_source.ingest(ts, item_groups[0], [[seattle2020]])
        data_source.ingest(ts, item_groups[1], [[seattle2020]])
        assert tile_store.is_raster_ready(layer_name, item_0.name, self.TEST_BANDS)
        assert tile_store.is_raster_ready(layer_name, item_1.name, self.TEST_BANDS)

        # Read back rasters and verify timestamps span each month's hours.
        read_proj = Projection(CRS.from_epsg(WGS84_EPSG), 0.1, -0.1)
        for item, expected_month in [(item_0, 7), (item_1, 8)]:
            bounds = ts.get_raster_bounds(item.name, self.TEST_BANDS, read_proj)
            raster = ts.read_raster(item.name, self.TEST_BANDS, read_proj, bounds)

            # Verify shape.
            assert raster.timestamps is not None
            assert len(raster.timestamps) > 0
            assert raster.array.shape[0] == len(self.TEST_BANDS)
            assert raster.array.shape[1] == len(raster.timestamps)

            # Verify data is not nodata for both variables.
            assert max(raster.array[0, :, 0, 0]) > 0
            assert max(raster.array[1, :, 0, 0]) > 0

            # Verify it has hourly timestamps spanning the month.
            assert 28 * 24 <= len(raster.timestamps) <= 31 * 24
            first_ts = raster.timestamps[0][0]
            second_ts = raster.timestamps[0][1]
            last_ts = raster.timestamps[-1][0]
            assert first_ts == datetime(2020, expected_month, 1, 0, 0, 0, tzinfo=UTC)
            assert (second_ts - first_ts) == timedelta(hours=1)
            assert abs(
                datetime(2020, expected_month + 1, 1, tzinfo=UTC) - last_ts
            ) <= timedelta(hours=1)
