"""Data source for Copernicus Climate Data Store."""

import os
import tempfile
import zipfile
from datetime import UTC, datetime
from typing import Any

import cdsapi
import netCDF4
import numpy as np
import shapely
from dateutil.relativedelta import relativedelta
from rasterio.crs import CRS
from upath import UPath

from rslearn.config import QueryConfig, SpaceMode
from rslearn.const import WGS84_EPSG, WGS84_PROJECTION
from rslearn.data_sources import DataSource, DataSourceContext, Item
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry
from rslearn.utils.raster_array import RasterArray

logger = get_logger(__name__)


class ERA5Land(DataSource):
    """Base class for ingesting ERA5 land data from the Copernicus Climate Data Store.

    An API key must be passed either in the configuration or via the CDSAPI_KEY
    environment variable. You can acquire an API key by going to the Climate Data Store
    website (https://cds.climate.copernicus.eu/), registering an account and logging
    in, and then getting the API key from the user profile page.

    The band names should match CDS variable names (see the reference at
    https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation). However,
    replace "_" with "-" in the variable names when specifying bands in the layer
    configuration.

    By default, all requests to the API will be for the whole globe. To speed up ingestion,
    we recommend specifying the bounds of the area of interest, in particular for hourly data.
    """

    api_url = "https://cds.climate.copernicus.eu/api"
    DATA_FORMAT = "netcdf"
    DOWNLOAD_FORMAT = "unarchived"
    PIXEL_SIZE = 0.1  # degrees, native resolution is 9km
    NODATA_VALUE = 0.0

    def __init__(
        self,
        dataset: str,
        product_type: str,
        band_names: list[str] | None = None,
        api_key: str | None = None,
        bounds: list[float] | None = None,
        context: DataSourceContext = DataSourceContext(),
    ):
        """Initialize a new ERA5Land instance.

        Args:
            dataset: the CDS dataset name (e.g., "reanalysis-era5-land-monthly-means").
            product_type: the CDS product type (e.g., "monthly_averaged_reanalysis").
            band_names: list of band names to acquire. These should correspond to CDS
                variable names but with "_" replaced with "-". This will only be used
                if the layer config is missing from the context.
            api_key: the API key. If not set, it should be set via the CDSAPI_KEY
                environment variable.
            bounds: optional bounding box as [min_lon, min_lat, max_lon, max_lat].
                If not specified, the whole globe will be used.
            context: the data source context.
        """
        self.dataset = dataset
        self.product_type = product_type
        self.bounds = bounds

        self.band_names: list[str]
        if context.layer_config is not None:
            self.band_names = []
            for band_set in context.layer_config.band_sets:
                for band in band_set.bands:
                    if band in self.band_names:
                        continue
                    self.band_names.append(band)
        elif band_names is not None:
            self.band_names = band_names
        else:
            raise ValueError(
                "band_names must be set if layer_config is not in the context"
            )

        self.client = cdsapi.Client(
            url=self.api_url,
            key=api_key,
        )

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[Item]]]:
        """Get a list if items in the data source intersecting the given geometries.

        Args:
            geometries: the spatiotemporal geometries
            query_config: the query configuration

        Returns:
            List of groups of items that should be retrieved for each geometry.
        """
        # We only support mosaic here, other query modes don't really make sense.
        if query_config.space_mode != SpaceMode.MOSAIC:
            raise ValueError("expected mosaic space mode in the query configuration")

        all_groups = []
        for geometry in geometries:
            if geometry.time_range is None:
                raise ValueError("expected all geometries to have a time range")

            # Compute one mosaic for each month in this geometry.
            cur_date = datetime(
                geometry.time_range[0].year,
                geometry.time_range[0].month,
                1,
                tzinfo=UTC,
            )
            end_date = datetime(
                geometry.time_range[1].year,
                geometry.time_range[1].month,
                1,
                tzinfo=UTC,
            )

            month_dates: list[datetime] = []
            while cur_date <= end_date:
                month_dates.append(cur_date)
                cur_date += relativedelta(months=1)

            cur_groups = []
            for cur_date in month_dates:
                # Collect Item list corresponding to the current month.
                items = []
                item_name = f"era5land_monthlyaveraged_{cur_date.year}_{cur_date.month}"
                # Use bounds if set, otherwise use whole globe
                if self.bounds is not None:
                    bounds = self.bounds
                else:
                    bounds = [-180, -90, 180, 90]
                # Time is just the given month.
                start_date = datetime(cur_date.year, cur_date.month, 1, tzinfo=UTC)
                time_range = (
                    start_date,
                    start_date + relativedelta(months=1),
                )
                geometry = STGeometry(
                    WGS84_PROJECTION,
                    shapely.box(*bounds),
                    time_range,
                )
                items.append(Item(item_name, geometry))
                cur_groups.append(items)
            all_groups.append(cur_groups)

        return all_groups

    def deserialize_item(self, serialized_item: dict) -> Item:
        """Deserializes an item from JSON-decoded data."""
        return Item.deserialize(serialized_item)

    def _parse_nc(
        self, nc_path: UPath
    ) -> tuple[
        np.ndarray,
        list[tuple[datetime, datetime]],
        Projection,
        PixelBounds,
    ]:
        """Convert a netCDF file into a CTHW array with timestamps and geo metadata.

        Args:
            nc_path: path to the NetCDF file.

        Returns:
            A tuple of (array, timestamps, projection, bounds) where array has
            shape (C, T, H, W), timestamps has length T, and projection/bounds
            describe the spatial extent.
        """
        nc = netCDF4.Dataset(nc_path)
        # The file contains a list of variables.
        # These variables include things that are not bands that we want, such as
        # latitude, longitude, valid_time, expver, etc.
        # But the list of variables should include the bands we want in the correct
        # order. And we can distinguish those bands from other "variables" because they
        # will be 3D while the others will be scalars or 1D.

        band_arrays = []
        num_time_steps = None
        for band_name in nc.variables:
            band_data = nc.variables[band_name]
            if len(band_data.shape) != 3:
                # This should be one of those variables that we want to skip.
                continue

            logger.debug(
                f"NC file {nc_path} has variable {band_name} with shape {band_data.shape}"
            )
            # Variable data is stored in a 3D array (time, height, width)
            # For hourly data, time is number of days in the month x 24 hours
            if num_time_steps is None:
                num_time_steps = band_data.shape[0]
            elif band_data.shape[0] != num_time_steps:
                raise ValueError(
                    f"Variable {band_name} has {band_data.shape[0]} time steps, "
                    f"but expected {num_time_steps}"
                )
            # Original shape: (time, height, width)
            band_array = np.array(band_data[:])
            band_arrays.append(band_array)

        # After stacking: (num_variables, time, height, width)
        array = np.stack(band_arrays, axis=0)

        # Replace NaN values with nodata value
        array = np.where(np.isnan(array), self.NODATA_VALUE, array)

        # Build timestamps from valid_time.
        valid_time_var = nc.variables["valid_time"]
        datetimes = [
            datetime(d.year, d.month, d.day, d.hour, d.minute, d.second, tzinfo=UTC)
            for d in netCDF4.num2date(valid_time_var[:], units=valid_time_var.units)
        ]
        timestamps: list[tuple[datetime, datetime]] = []
        for i, start in enumerate(datetimes):
            # If there's only one timestamp, we treat the item as covering a point
            # in time.
            if len(datetimes) == 1:
                timestamps.append((start, start))
            # Otherwise, we use the provided time spacing. For the last timestamp, we
            # use the spacing from the previous timestamp.
            elif i < len(datetimes) - 1:
                timestamps.append((start, datetimes[i + 1]))
            else:
                spacing = datetimes[i] - datetimes[i - 1]
                timestamps.append((start, start + spacing))

        # Build projection and bounds.
        lat = nc.variables["latitude"][:]
        lon = nc.variables["longitude"][:]
        # Convert longitude from 0–360 to -180–180 and sort
        lon = (lon + 180) % 360 - 180
        sorted_indices = lon.argsort()
        lon = lon[sorted_indices]

        # Reorder the data array to match the new longitude order
        array = array[:, :, sorted_indices]

        # Check the spacing of the grid, make sure it's uniform
        for i in range(len(lon) - 1):
            if round(lon[i + 1] - lon[i], 1) != self.PIXEL_SIZE:
                raise ValueError(
                    f"Longitude spacing is not uniform: {lon[i + 1] - lon[i]}"
                )
        for i in range(len(lat) - 1):
            if round(lat[i + 1] - lat[i], 1) != -self.PIXEL_SIZE:
                raise ValueError(
                    f"Latitude spacing is not uniform: {lat[i + 1] - lat[i]}"
                )

        projection = Projection(
            CRS.from_epsg(WGS84_EPSG), self.PIXEL_SIZE, -self.PIXEL_SIZE
        )
        west = lon.min() - self.PIXEL_SIZE / 2
        north = lat.max() + self.PIXEL_SIZE / 2
        col_off = round(west / self.PIXEL_SIZE)
        row_off = round(north / (-self.PIXEL_SIZE))
        bounds: PixelBounds = (
            col_off,
            row_off,
            col_off + len(lon),
            row_off + len(lat),
        )

        nc.close()
        return array, timestamps, projection, bounds

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[Item],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        This method should be overridden by subclasses.

        Args:
            tile_store: the tile store to ingest into
            items: the items to ingest
            geometries: a list of geometries needed for each item
        """
        raise NotImplementedError("Subclasses must implement ingest method")


class ERA5LandMonthlyMeans(ERA5Land):
    """A data source for ingesting ERA5 land monthly averaged data from the Copernicus Climate Data Store.

    This data source corresponds to the reanalysis-era5-land-monthly-means product.
    """

    def __init__(
        self,
        band_names: list[str] | None = None,
        api_key: str | None = None,
        bounds: list[float] | None = None,
        context: DataSourceContext = DataSourceContext(),
    ):
        """Initialize a new ERA5LandMonthlyMeans instance.

        Args:
            band_names: list of band names to acquire. These should correspond to CDS
                variable names but with "_" replaced with "-". This will only be used
                if the layer config is missing from the context.
            api_key: the API key. If not set, it should be set via the CDSAPI_KEY
                environment variable.
            bounds: optional bounding box as [min_lon, min_lat, max_lon, max_lat].
                If not specified, the whole globe will be used.
            context: the data source context.
        """
        super().__init__(
            dataset="reanalysis-era5-land-monthly-means",
            product_type="monthly_averaged_reanalysis",
            band_names=band_names,
            api_key=api_key,
            bounds=bounds,
            context=context,
        )

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[Item],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        Args:
            tile_store: the tile store to ingest into
            items: the items to ingest
            geometries: a list of geometries needed for each item
        """
        # for CDS variable names, replace "-" with "_"
        variable_names = [band.replace("-", "_") for band in self.band_names]

        for item in items:
            if tile_store.is_raster_ready(item.name, self.band_names):
                continue

            # Send the request to the CDS API
            if self.bounds is not None:
                min_lon, min_lat, max_lon, max_lat = self.bounds
                area = [max_lat, min_lon, min_lat, max_lon]
            else:
                area = [90, -180, -90, 180]  # Whole globe

            request = {
                "product_type": [self.product_type],
                "variable": variable_names,
                "year": [f"{item.geometry.time_range[0].year}"],  # type: ignore
                "month": [
                    f"{item.geometry.time_range[0].month:02d}"  # type: ignore
                ],
                "time": ["00:00"],
                "area": area,
                "data_format": self.DATA_FORMAT,
                "download_format": self.DOWNLOAD_FORMAT,
            }
            logger.debug(
                f"CDS API request for year={request['year']} month={request['month']} area={area}"
            )
            with tempfile.TemporaryDirectory() as tmp_dir:
                local_nc_fname = os.path.join(tmp_dir, f"{item.name}.nc")
                self.client.retrieve(self.dataset, request, local_nc_fname)
                array, timestamps, projection, bounds = self._parse_nc(
                    UPath(local_nc_fname)
                )
                tile_store.write_raster(
                    item.name,
                    self.band_names,
                    projection,
                    bounds,
                    RasterArray(array=array, timestamps=timestamps),
                )


class ERA5LandHourly(ERA5Land):
    """A data source for ingesting ERA5 land hourly data from the Copernicus Climate Data Store.

    This data source corresponds to the reanalysis-era5-land product.
    """

    def __init__(
        self,
        band_names: list[str] | None = None,
        api_key: str | None = None,
        bounds: list[float] | None = None,
        context: DataSourceContext = DataSourceContext(),
    ):
        """Initialize a new ERA5LandHourly instance.

        Args:
            band_names: list of band names to acquire. These should correspond to CDS
                variable names but with "_" replaced with "-". This will only be used
                if the layer config is missing from the context.
            api_key: the API key. If not set, it should be set via the CDSAPI_KEY
                environment variable.
            bounds: optional bounding box as [min_lon, min_lat, max_lon, max_lat].
                If not specified, the whole globe will be used.
            context: the data source context.
        """
        super().__init__(
            dataset="reanalysis-era5-land",
            product_type="reanalysis",
            band_names=band_names,
            api_key=api_key,
            bounds=bounds,
            context=context,
        )

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[Item],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        Args:
            tile_store: the tile store to ingest into
            items: the items to ingest
            geometries: a list of geometries needed for each item
        """
        # for CDS variable names, replace "-" with "_"
        variable_names = [band.replace("-", "_") for band in self.band_names]

        for item in items:
            if tile_store.is_raster_ready(item.name, self.band_names):
                continue

            # Send the request to the CDS API
            # If area is not specified, the whole globe will be requested
            time_range = item.geometry.time_range
            if time_range is None:
                raise ValueError("Item must have a time range")

            # For hourly data, request all days in the month and all 24 hours
            start_time = time_range[0]

            # Get all days in the month
            year = start_time.year
            month = start_time.month
            # Get the last day of the month
            if month == 12:
                last_day = 31
            else:
                next_month = datetime(year, month + 1, 1, tzinfo=UTC)
                last_day = (next_month - relativedelta(days=1)).day

            days = [f"{day:02d}" for day in range(1, last_day + 1)]

            # Get all 24 hours
            hours = [f"{hour:02d}:00" for hour in range(24)]

            if self.bounds is not None:
                min_lon, min_lat, max_lon, max_lat = self.bounds
                area = [max_lat, min_lon, min_lat, max_lon]
            else:
                area = [90, -180, -90, 180]  # Whole globe

            request = {
                "product_type": [self.product_type],
                "variable": variable_names,
                "year": [f"{year}"],
                "month": [f"{month:02d}"],
                "day": days,
                "time": hours,
                "area": area,
                "data_format": self.DATA_FORMAT,
                "download_format": self.DOWNLOAD_FORMAT,
            }
            logger.debug(
                f"CDS API request for year={request['year']} month={request['month']} days={len(days)} hours={len(hours)} area={area}"
            )
            with tempfile.TemporaryDirectory() as tmp_dir:
                local_nc_fname = os.path.join(tmp_dir, f"{item.name}.nc")
                self.client.retrieve(self.dataset, request, local_nc_fname)
                array, timestamps, projection, bounds = self._parse_nc(
                    UPath(local_nc_fname)
                )
                tile_store.write_raster(
                    item.name,
                    self.band_names,
                    projection,
                    bounds,
                    RasterArray(array=array, timestamps=timestamps),
                )


class ERA5LandHourlyTimeseries(DataSource):
    """A data source for ingesting ERA5-Land hourly time-series data for individual points.

    This data source corresponds to the reanalysis-era5-land-timeseries dataset, which is
    optimized for retrieving long time-series data for single points rather than spatial
    areas. It always materializes a 1x1 raster for each window, containing data from the
    closest 0.1 degree grid cell to the window's center.

    An API key must be passed either in the configuration or via the CDSAPI_KEY
    environment variable. You can acquire an API key by going to the Climate Data Store
    website (https://cds.climate.copernicus.eu/), registering an account and logging
    in, and then getting the API key from the user profile page.

    The band names should match CDS variable names (see the reference at
    https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation). However,
    replace "_" with "-" in the variable names when specifying bands in the layer
    configuration.
    """

    API_URL = "https://cds.climate.copernicus.eu/api"
    DATASET = "reanalysis-era5-land-timeseries"
    DATA_FORMAT = "netcdf"
    PIXEL_SIZE = 0.1  # degrees, native resolution is 9km
    NODATA_VALUE = 0.0

    def __init__(
        self,
        band_names: list[str] | None = None,
        api_key: str | None = None,
        context: DataSourceContext = DataSourceContext(),
    ):
        """Initialize a new ERA5LandHourlyTimeseries instance.

        Args:
            band_names: list of band names to acquire. These should correspond to CDS
                variable names but with "_" replaced with "-". This will only be used
                if the layer config is missing from the context.
            api_key: the API key. If not set, it should be set via the CDSAPI_KEY
                environment variable.
            context: the data source context.
        """
        self.band_names: list[str]
        if context.layer_config is not None:
            self.band_names = []
            for band_set in context.layer_config.band_sets:
                for band in band_set.bands:
                    if band in self.band_names:
                        continue
                    self.band_names.append(band)
        elif band_names is not None:
            self.band_names = band_names
        else:
            raise ValueError(
                "band_names must be set if layer_config is not in the context"
            )

        self.client = cdsapi.Client(
            url=self.API_URL,
            key=api_key,
        )

    def _snap_to_grid(self, lon: float, lat: float) -> tuple[float, float]:
        """Snap coordinates to the nearest 0.1 degree grid cell center.

        The ERA5-Land timeseries dataset uses a 0.1 degree grid. When a requested
        location doesn't exactly match a grid point, the API automatically selects
        the nearest grid point.

        Args:
            lon: the longitude in degrees
            lat: the latitude in degrees

        Returns:
            A tuple of (snapped_lon, snapped_lat) rounded to 1 decimal place.
        """
        snapped_lon = round(lon, 1)
        snapped_lat = round(lat, 1)
        return (snapped_lon, snapped_lat)

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[Item]]]:
        """Get a list of items in the data source intersecting the given geometries.

        For each geometry, this method extracts the centroid, snaps it to the nearest
        0.1 degree grid cell, and creates items for each month in the time range.
        Items are uniquely identified by their grid coordinates and time range,
        allowing multiple geometries in the same grid cell to share the same item.

        Args:
            geometries: the spatiotemporal geometries
            query_config: the query configuration

        Returns:
            List of groups of items that should be retrieved for each geometry.
        """
        # We only support mosaic here, other query modes don't really make sense.
        if query_config.space_mode != SpaceMode.MOSAIC:
            raise ValueError("expected mosaic space mode in the query configuration")

        all_groups = []
        for geometry in geometries:
            if geometry.time_range is None:
                raise ValueError("expected all geometries to have a time range")

            # Convert geometry to WGS84 and get the centroid.
            # This assumes that the geometry is smaller than 0.1 x 0.1 degrees
            # So that we can use the centroid to find the nearest grid cell.
            wgs84_geometry = geometry.to_projection(WGS84_PROJECTION)
            centroid = wgs84_geometry.shp.centroid
            snapped_lon, snapped_lat = self._snap_to_grid(centroid.x, centroid.y)

            # Compute one item for each month in this geometry's time range.
            cur_date = datetime(
                geometry.time_range[0].year,
                geometry.time_range[0].month,
                1,
                tzinfo=UTC,
            )
            end_date = datetime(
                geometry.time_range[1].year,
                geometry.time_range[1].month,
                1,
                tzinfo=UTC,
            )

            month_dates: list[datetime] = []
            while cur_date <= end_date:
                month_dates.append(cur_date)
                cur_date += relativedelta(months=1)

            cur_groups = []
            for cur_date in month_dates:
                # Create a unique item name based on grid coordinates and time
                # Use consistent formatting for lat/lon to ensure proper caching
                lat_str = f"{snapped_lat:.1f}".replace("-", "n")
                lon_str = f"{snapped_lon:.1f}".replace("-", "n")
                item_name = (
                    f"era5land_timeseries_{lat_str}_{lon_str}_"
                    f"{cur_date.year}_{cur_date.month:02d}"
                )

                # Time is just the given month.
                start_date = datetime(cur_date.year, cur_date.month, 1, tzinfo=UTC)
                time_range = (
                    start_date,
                    start_date + relativedelta(months=1),
                )

                # Create a point geometry at the snapped grid location
                point_geom = shapely.Point(snapped_lon, snapped_lat)
                item_geometry = STGeometry(
                    WGS84_PROJECTION,
                    point_geom,
                    time_range,
                )

                items = [Item(item_name, item_geometry)]
                cur_groups.append(items)

            all_groups.append(cur_groups)

        return all_groups

    def deserialize_item(self, serialized_item: Any) -> Item:
        """Deserializes an item from JSON-decoded data."""
        assert isinstance(serialized_item, dict)
        return Item.deserialize(serialized_item)

    def _parse_nc_timeseries(
        self, nc_paths: list[UPath], lon: float, lat: float
    ) -> tuple[
        np.ndarray,
        list[tuple[datetime, datetime]],
        Projection,
        PixelBounds,
    ]:
        """Parse timeseries netCDF files into a CTHW array with timestamps.

        The timeseries API may return multiple netCDF files (one per variable).
        This method reads data from all files and combines them into a
        (C, T, 1, 1) array with per-hour timestamps.

        Args:
            nc_paths: list of paths to the netCDF files.
            lon: the longitude of the point.
            lat: the latitude of the point.

        Returns:
            A tuple of (array, timestamps, projection, bounds).
        """
        # The timeseries files contain 1D arrays for each variable (time dimension only)
        # We need to collect these from all files and stack them into a multi-band raster
        band_arrays = []
        num_time_steps = None
        all_datetimes: list[datetime] | None = None

        for nc_path in nc_paths:
            nc = netCDF4.Dataset(nc_path)

            if "valid_time" in nc.variables and all_datetimes is None:
                valid_time_var = nc.variables["valid_time"]
                all_datetimes = [
                    datetime(
                        d.year,
                        d.month,
                        d.day,
                        d.hour,
                        d.minute,
                        d.second,
                        tzinfo=UTC,
                    )
                    for d in netCDF4.num2date(
                        valid_time_var[:], units=valid_time_var.units
                    )
                ]

            for band_name in nc.variables:
                band_data = nc.variables[band_name]
                # Timeseries variables are 1D (time only), skip metadata variables
                if len(band_data.shape) != 1:
                    continue
                # Skip coordinate/time variables
                if band_name in (
                    "time",
                    "valid_time",
                    "latitude",
                    "longitude",
                    "expver",
                ):
                    continue

                logger.debug(
                    f"NC file {nc_path} has variable {band_name} with shape {band_data.shape}"
                )

                if num_time_steps is None:
                    num_time_steps = band_data.shape[0]
                elif band_data.shape[0] != num_time_steps:
                    raise ValueError(
                        f"Variable {band_name} has {band_data.shape[0]} time steps, "
                        f"but expected {num_time_steps}"
                    )

                # Get the 1D array for this variable.
                band_array = np.array(band_data[:])
                band_arrays.append(band_array)

            nc.close()

        if not band_arrays:
            raise ValueError(f"No valid band data found in {nc_paths}")

        # Stack: (C, T) then reshape to (C, T, 1, 1).
        stacked = np.stack(band_arrays, axis=0)  # (C, T)
        array = stacked[:, :, np.newaxis, np.newaxis]  # (C, T, 1, 1)

        # Replace NaN values with nodata value
        array = np.where(np.isnan(array), self.NODATA_VALUE, array)

        # Build timestamps.
        if all_datetimes is None:
            raise ValueError(
                f"No valid_time variable found in any of the NetCDF files: {nc_paths}"
            )

        timestamps: list[tuple[datetime, datetime]] = []
        for i, start in enumerate(all_datetimes):
            if len(all_datetimes) == 1:
                timestamps.append((start, start))
            elif i < len(all_datetimes) - 1:
                timestamps.append((start, all_datetimes[i + 1]))
            else:
                spacing = all_datetimes[i] - all_datetimes[i - 1]
                timestamps.append((start, start + spacing))

        # Build projection and bounds for a 1x1 pixel.
        projection = Projection(
            CRS.from_epsg(WGS84_EPSG), self.PIXEL_SIZE, -self.PIXEL_SIZE
        )
        west = lon - self.PIXEL_SIZE / 2
        north = lat + self.PIXEL_SIZE / 2
        col_off = round(west / self.PIXEL_SIZE)
        row_off = round(north / (-self.PIXEL_SIZE))
        bounds: PixelBounds = (col_off, row_off, col_off + 1, row_off + 1)

        return array, timestamps, projection, bounds

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[Item],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        Args:
            tile_store: the tile store to ingest into
            items: the items to ingest
            geometries: a list of geometries needed for each item
        """
        # for CDS variable names, replace "-" with "_"
        variable_names = [band.replace("-", "_") for band in self.band_names]

        for item in items:
            if tile_store.is_raster_ready(item.name, self.band_names):
                continue

            time_range = item.geometry.time_range
            if time_range is None:
                raise ValueError("Item must have a time range")

            # Extract coordinates from the item's point geometry
            centroid = item.geometry.shp.centroid
            lon = centroid.x
            lat = centroid.y

            # Build the date range string for the API request
            start_time = time_range[0]
            end_time = time_range[1] - relativedelta(days=1)  # Exclusive end
            date_str = (
                f"{start_time.year}-{start_time.month:02d}-{start_time.day:02d}/"
                f"{end_time.year}-{end_time.month:02d}-{end_time.day:02d}"
            )

            request = {
                "variable": variable_names,
                "location": {"longitude": lon, "latitude": lat},
                "date": [date_str],
                "data_format": self.DATA_FORMAT,
            }
            logger.debug(f"CDS API request for location=({lat}, {lon}) date={date_str}")

            with tempfile.TemporaryDirectory() as tmp_dir:
                local_zip_fname = os.path.join(tmp_dir, f"{item.name}.zip")

                # The timeseries API returns a zip file containing the NetCDF
                self.client.retrieve(self.DATASET, request, local_zip_fname)

                # Extract all NetCDF files from the zip
                # The API may return multiple NC files (one per variable)
                with zipfile.ZipFile(local_zip_fname, "r") as zip_ref:
                    nc_files = [f for f in zip_ref.namelist() if f.endswith(".nc")]
                    if not nc_files:
                        raise ValueError(
                            f"No NetCDF file found in downloaded zip: {local_zip_fname}"
                        )
                    zip_ref.extractall(tmp_dir)
                    local_nc_paths = [
                        UPath(os.path.join(tmp_dir, nc_file)) for nc_file in nc_files
                    ]

                array, timestamps, projection, bounds = self._parse_nc_timeseries(
                    local_nc_paths, lon, lat
                )
                tile_store.write_raster(
                    item.name,
                    self.band_names,
                    projection,
                    bounds,
                    RasterArray(array=array, timestamps=timestamps),
                )
