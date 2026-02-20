"""Mocked integration tests for Climate Data Store (ERA5-Land) data sources."""

import json
import pathlib
import shutil
import zipfile
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import cdsapi
import netCDF4
import numpy as np
import pytest
import shapely
from rasterio.crs import CRS
from upath import UPath

from rslearn.config import (
    QueryConfig,
)
from rslearn.const import WGS84_EPSG, WGS84_PROJECTION
from rslearn.data_sources.climate_data_store import (
    ERA5LandHourlyTimeseries,
    ERA5LandMonthlyMeans,
)
from rslearn.dataset import Dataset
from rslearn.dataset.manage import (
    ingest_dataset_windows,
    materialize_dataset_windows,
    prepare_dataset_windows,
)
from rslearn.dataset.window import Window
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.raster_format import GeotiffRasterFormat

PIXEL_SIZE = 0.1
TEST_BANDS = ["2m-temperature", "total-precipitation"]


def _make_monthly_nc(
    nc_path: pathlib.Path,
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
    year: int,
    month: int,
) -> None:
    """Create a netCDF file mimicking ERA5-Land monthly means.

    Band variables have shape (1, height, width) with pixel values set to
    (row_index * width + col_index) so each pixel is uniquely identifiable.

    Args:
        nc_path: where to write the file.
        lon_range: (min_lon, max_lon) of grid cell centers.
        lat_range: (min_lat, max_lat) of grid cell centers, max first (descending).
        year: year for the valid_time.
        month: month for the valid_time.
    """
    # Create longitude and latitude dimensions.
    lons = np.arange(lon_range[0], lon_range[1] + PIXEL_SIZE / 2, PIXEL_SIZE)
    lats = np.arange(lat_range[1], lat_range[0] - PIXEL_SIZE / 2, -PIXEL_SIZE)
    lons = np.round(lons, 1)
    lats = np.round(lats, 1)

    # Initialize the netCDF4 dataset.
    width = len(lons)
    height = len(lats)
    num_timesteps = 1
    ds = netCDF4.Dataset(str(nc_path), "w", format="NETCDF4")
    ds.createDimension("valid_time", num_timesteps)
    ds.createDimension("latitude", height)
    ds.createDimension("longitude", width)

    # Create netCDF4 variables.
    vt = ds.createVariable("valid_time", "f8", ("valid_time",))
    vt.units = "hours since 1900-01-01 00:00:00"
    ref = datetime(1900, 1, 1, tzinfo=UTC)
    target = datetime(year, month, 1, tzinfo=UTC)
    vt[:] = [(target - ref).total_seconds() / 3600.0]

    lat_var = ds.createVariable("latitude", "f8", ("latitude",))
    lat_var[:] = lats

    lon_var = ds.createVariable("longitude", "f8", ("longitude",))
    lon_var[:] = lons

    # Finally add the data variables.
    # We use arange so data[i, j] = i*width+j, and add 1000 for "tp".
    for band_name in ["t2m", "tp"]:
        var = ds.createVariable(
            band_name, "f4", ("valid_time", "latitude", "longitude")
        )
        data = np.arange(height * width, dtype=np.float32).reshape(1, height, width)
        if band_name == "tp":
            data = data + 1000.0
        var[:] = data

    ds.close()


def _make_timeseries_nc(
    nc_path: pathlib.Path,
    year: int,
    month: int,
    band_name: str,
    num_hours: int,
    start_value: int = 0,
) -> None:
    """Create a netCDF file mimicking one ERA5-Land timeseries variable.

    The timeseries API returns one NC per variable, each with a 1D array of length
    num_hours.

    Args:
        nc_path: where to write the file.
        year: year for the valid_time.
        month: month for the valid_time.
        band_name: CDS variable name (e.g., "t2m").
        num_hours: number of hourly time steps.
        start_value: first value in the sequence (default 0). Allows creating
            continuous sequences across multiple months.
    """
    ds = netCDF4.Dataset(str(nc_path), "w", format="NETCDF4")

    ds.createDimension("valid_time", num_hours)

    # Add the timestamps.
    vt = ds.createVariable("valid_time", "f8", ("valid_time",))
    vt.units = "hours since 1900-01-01 00:00:00"
    ref = datetime(1900, 1, 1, tzinfo=UTC)
    start = datetime(year, month, 1, tzinfo=UTC)
    hours_offset = (start - ref).total_seconds() / 3600.0
    vt[:] = [hours_offset + i for i in range(num_hours)]

    # And add the data variable.
    var = ds.createVariable(band_name, "f4", ("valid_time",))
    var[:] = np.arange(start_value, start_value + num_hours, dtype=np.float32)

    ds.close()


class TestERA5LandMonthlyMeans:
    """Mock integration test for ERA5LandMonthlyMeans.

    Creates a synthetic netCDF with a 1x1 degree grid (10x10 pixels at 0.1 deg),
    mocks the CDS API retrieve call, then verifies pixel values and timestamps.
    """

    LON_RANGE = (-1.0, -0.1)  # 10 pixels
    LAT_RANGE = (0.1, 1.0)  # 10 pixels
    YEAR = 2025
    MONTH = 6

    def test_ingest_and_read(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Mock the data source to always download the _make_monthly_nc file.
        nc_path = tmp_path / "test_monthly.nc"
        _make_monthly_nc(nc_path, self.LON_RANGE, self.LAT_RANGE, self.YEAR, self.MONTH)

        def fake_retrieve(dataset: str, request: dict, target: str) -> None:
            shutil.copy(str(nc_path), target)

        mock_client = MagicMock()
        mock_client.retrieve.side_effect = fake_retrieve
        monkeypatch.setattr(cdsapi, "Client", lambda **kwargs: mock_client)

        # Initialize and query the data source.
        # The query doesn't need to be mocked since Items are created for each month
        # covering the entire bounds.
        data_source = ERA5LandMonthlyMeans(
            band_names=TEST_BANDS,
            bounds=[
                self.LON_RANGE[0],
                self.LAT_RANGE[0],
                self.LON_RANGE[1],
                self.LAT_RANGE[1],
            ],
        )

        query_geom = STGeometry(
            WGS84_PROJECTION,
            shapely.box(-0.55, 0.45, -0.45, 0.55),
            (
                datetime(self.YEAR, self.MONTH, 1, tzinfo=UTC),
                datetime(self.YEAR, self.MONTH, 28, tzinfo=UTC),
            ),
        )

        query_config = QueryConfig(max_matches=1)
        item_groups = data_source.get_items([query_geom], query_config)[0]
        assert len(item_groups) == 1
        item = item_groups[0][0]

        # Now ingest the data into a TileStore.
        tile_store_dir = UPath(tmp_path / "tiles")
        tile_store = DefaultTileStore(str(tile_store_dir))
        tile_store.set_dataset_path(tile_store_dir)
        layer_name = "era5_monthly"
        tile_store_layer = TileStoreWithLayer(tile_store, layer_name)

        data_source.ingest(tile_store_layer, [item], [[query_geom]])
        assert tile_store_layer.is_raster_ready(item.name, TEST_BANDS)

        # Read raster at the native 0.1 deg projection.
        read_proj = Projection(CRS.from_epsg(WGS84_EPSG), PIXEL_SIZE, -PIXEL_SIZE)
        bounds = tile_store_layer.get_raster_bounds(item.name, TEST_BANDS, read_proj)
        raster = tile_store_layer.read_raster(item.name, TEST_BANDS, read_proj, bounds)

        # Shape should be (2 bands, 1 timestep, 10 height, 10 width).
        assert raster.array.shape == (2, 1, 10, 10)

        # Verify timestamp corresponds to requested month.
        assert raster.timestamps is not None
        assert raster.timestamps[0][0] == datetime(self.YEAR, self.MONTH, 1, tzinfo=UTC)

        # The first band (t2m) was filled with arange(100).reshape(1, 10, 10).
        band0 = raster.array[0, 0, :, :]
        assert band0[0, 0] == 0.0
        assert band0[9, 9] == 99.0

        # The second band (tp) was filled with arange(100) + 1000.
        band1 = raster.array[1, 0, :, :]
        assert band1[0, 0] == 1000.0
        assert band1[9, 9] == 1099.0


class TestERA5LandHourlyTimeseries:
    """Mock integration test for ERA5LandHourlyTimeseries.

    Creates synthetic timeseries netCDF files (one per variable), packages
    them in a zip, mocks the CDS API, and verifies the parsed timestamps
    and raster geometry.
    """

    # January 2025 has 31 days = 744 hours.
    YEAR = 2025
    MONTH = 1
    NUM_HOURS = 31 * 24  # 744
    LON = -122.3
    LAT = 47.6

    def test_ingest_and_read(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Create two NC files (one per variable) and zip them.
        nc_dir = tmp_path / "nc_files"
        nc_dir.mkdir()
        nc_t2m = nc_dir / "t2m.nc"
        nc_tp = nc_dir / "tp.nc"
        _make_timeseries_nc(nc_t2m, self.YEAR, self.MONTH, "t2m", self.NUM_HOURS)
        _make_timeseries_nc(nc_tp, self.YEAR, self.MONTH, "tp", self.NUM_HOURS)

        zip_path = tmp_path / "timeseries.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(nc_t2m, arcname="t2m.nc")
            zf.write(nc_tp, arcname="tp.nc")

        # Mock the data source to always download this zip file.
        def fake_retrieve(dataset: str, request: dict, target: str) -> None:
            shutil.copy(str(zip_path), target)

        mock_client = MagicMock()
        mock_client.retrieve.side_effect = fake_retrieve
        monkeypatch.setattr(cdsapi, "Client", lambda **kwargs: mock_client)

        # Now we can initialize the data source and query it.
        data_source = ERA5LandHourlyTimeseries(band_names=TEST_BANDS)

        query_geom = STGeometry(
            WGS84_PROJECTION,
            shapely.Point(self.LON, self.LAT),
            (
                datetime(self.YEAR, self.MONTH, 1, tzinfo=UTC),
                datetime(self.YEAR, self.MONTH, 31, tzinfo=UTC),
            ),
        )

        query_config = QueryConfig(max_matches=1)
        item_groups = data_source.get_items([query_geom], query_config)[0]
        # There should be one item since the data source produces point items for each
        # calendar month over the geometry's time range.
        assert len(item_groups) == 1
        item = item_groups[0][0]

        # The item should be snapped to the grid.
        assert item.geometry.shp.geom_type == "Point"
        centroid = item.geometry.shp
        assert round(centroid.x, 1) == self.LON
        assert round(centroid.y, 1) == self.LAT

        # Ingest it into the tile store.
        tile_store_dir = UPath(tmp_path / "tiles")
        tile_store = DefaultTileStore(str(tile_store_dir))
        tile_store.set_dataset_path(tile_store_dir)
        layer_name = "era5_ts"
        ts = TileStoreWithLayer(tile_store, layer_name)

        data_source.ingest(ts, [item], [[query_geom]])
        assert ts.is_raster_ready(item.name, TEST_BANDS)

        # Read raster at the native 0.1 deg projection.
        read_proj = Projection(CRS.from_epsg(WGS84_EPSG), PIXEL_SIZE, -PIXEL_SIZE)
        bounds = ts.get_raster_bounds(item.name, TEST_BANDS, read_proj)
        raster = ts.read_raster(item.name, TEST_BANDS, read_proj, bounds)

        # Shape: (2 bands, 744 hours, H=1, W=1).
        assert raster.array.shape == (2, self.NUM_HOURS, 1, 1)

        # Verify timestamps.
        assert raster.timestamps is not None
        assert len(raster.timestamps) == self.NUM_HOURS
        first_ts = raster.timestamps[0][0]
        last_ts = raster.timestamps[-1][0]
        assert first_ts == datetime(self.YEAR, self.MONTH, 1, 0, 0, 0, tzinfo=UTC)
        assert last_ts == datetime(self.YEAR, self.MONTH, 31, 23, 0, 0, tzinfo=UTC)

        # Each timestamp should be 1 hour apart.
        for i in range(len(raster.timestamps) - 1):
            start_i = raster.timestamps[i][0]
            start_next = raster.timestamps[i + 1][0]
            assert (start_next - start_i) == timedelta(seconds=3600)

        # Verify band values (each variable was filled with arange(num_hours)).
        assert raster.array[0, 0, 0, 0] == 0.0
        assert raster.array[0, -1, 0, 0] == self.NUM_HOURS - 1
        assert raster.array[1, 0, 0, 0] == 0.0
        assert raster.array[1, -1, 0, 0] == self.NUM_HOURS - 1

    def test_full_pipeline_with_temporal_clipping(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test full prepare/ingest/materialize pipeline with temporal clipping.

        Creates data for January and February 2025 with sequential values 1-1416.
        Window covers Jan 15 - Feb 15, so after materialization with
        SPATIAL_MOSAIC_TEMPORAL_STACK, only timesteps within that range should remain.

        Expected:
        - Jan 15 00:00 (hour 336, value 337) through Feb 14 23:00 (hour 1079, value 1080)
        - Total: 744 hours (31 days)
        """
        band_name = "2m-temperature"

        # Create NetCDF files with sequential values
        nc_dir = tmp_path / "nc_files"
        nc_dir.mkdir()

        # January 2025: values 1-744
        nc_jan = nc_dir / "jan_t2m.nc"
        _make_timeseries_nc(nc_jan, 2025, 1, "t2m", 744, start_value=1)

        # February 2025: values 745-1416
        nc_feb = nc_dir / "feb_t2m.nc"
        _make_timeseries_nc(nc_feb, 2025, 2, "t2m", 672, start_value=745)

        # Package each in a zip file
        zip_jan = tmp_path / "jan_timeseries.zip"
        with zipfile.ZipFile(zip_jan, "w") as zf:
            zf.write(nc_jan, arcname="t2m.nc")

        zip_feb = tmp_path / "feb_timeseries.zip"
        with zipfile.ZipFile(zip_feb, "w") as zf:
            zf.write(nc_feb, arcname="t2m.nc")

        # Mock the CDS API to return the appropriate zip file based on month
        def fake_retrieve(dataset: str, request: dict, target: str) -> None:
            # Parse the month from the date field (format: '2025-01-01/2025-01-31')
            date_str = request.get("date", [""])[0]
            if "/" in date_str:
                start_date = date_str.split("/")[0]
                month = start_date.split("-")[1]
            else:
                raise ValueError(f"Unexpected date format in request: {date_str}")

            if month == "01":
                shutil.copy(str(zip_jan), target)
            elif month == "02":
                shutil.copy(str(zip_feb), target)
            else:
                raise ValueError(f"Unexpected month in request: {month}")

        mock_client = MagicMock()
        mock_client.retrieve.side_effect = fake_retrieve
        monkeypatch.setattr(cdsapi, "Client", lambda **kwargs: mock_client)

        # Create a dataset with ERA5LandHourlyTimeseries data source
        ds_path = UPath(tmp_path / "dataset")
        ds_path.mkdir()

        dataset_config = {
            "layers": {
                "era5_layer": {
                    "type": "raster",
                    "compositing_method": "SPATIAL_MOSAIC_TEMPORAL_STACK",
                    "band_sets": [
                        {
                            "dtype": "float32",
                            "bands": [band_name],
                        }
                    ],
                    "data_source": {
                        "class_path": "rslearn.data_sources.climate_data_store.ERA5LandHourlyTimeseries",
                        "init_args": {
                            "band_names": [band_name],
                        },
                        "query_config": {
                            "space_mode": "SINGLE_COMPOSITE",
                        },
                    },
                },
            },
        }

        with (ds_path / "config.json").open("w") as f:
            json.dump(dataset_config, f)

        dataset = Dataset(ds_path)

        # Create a window with time range Jan 15 - Feb 15
        # Convert lat/lon to pixel coordinates at 0.001 degrees/pixel.
        window_pixel_size = 0.001
        window_projection = Projection(
            CRS.from_epsg(WGS84_EPSG), window_pixel_size, -window_pixel_size
        )
        bounds = (
            int(self.LON / window_pixel_size),
            int(self.LAT / -window_pixel_size),
            int(self.LON / window_pixel_size) + 1,
            int(self.LAT / -window_pixel_size) + 1,
        )
        window = Window(
            storage=dataset.storage,
            group="default",
            name="test_window",
            projection=window_projection,
            bounds=bounds,
            time_range=(
                datetime(2025, 1, 15, tzinfo=UTC),
                datetime(2025, 2, 15, tzinfo=UTC),
            ),
        )
        window.save()

        # Run prepare/ingest/materialize.
        windows = dataset.load_windows()
        prepare_dataset_windows(dataset, windows)
        ingest_dataset_windows(dataset, windows)
        materialize_dataset_windows(dataset, windows)

        # Read the materialized raster
        layer_dir = window.get_raster_dir("era5_layer", [band_name], group_idx=0)
        raster_format = GeotiffRasterFormat()
        raster = raster_format.decode_raster(
            layer_dir, window.projection, window.bounds
        )

        # Verify shape: (1 band, 744 timesteps, height, width)
        # The spatial dimensions depend on the window bounds, but temporal should be 744
        assert raster.array.shape == (1, 744, 1, 1)

        # Verify timestamps are clipped correctly
        assert raster.timestamps is not None
        assert len(raster.timestamps) == 744
        assert raster.timestamps[0][0] == datetime(2025, 1, 15, 0, 0, 0, tzinfo=UTC)
        assert raster.timestamps[-1][0] == datetime(2025, 2, 14, 23, 0, 0, tzinfo=UTC)

        # Verify data values are sequential and correct (337, 338, 339, ..., 1080)
        expected_values = np.arange(337, 1081, dtype=np.float32)
        assert np.array_equal(raster.array[0, :, 0, 0], expected_values)
