"""Data source for Copernicus Climate Data Store."""

import os
import tempfile
from datetime import UTC, datetime
from typing import Any

import cdsapi
import netCDF4
import numpy as np
import rasterio
import shapely
from dateutil.relativedelta import relativedelta
from rasterio.transform import from_origin
from upath import UPath

from rslearn.config import QueryConfig, SpaceMode
from rslearn.const import WGS84_EPSG, WGS84_PROJECTION
from rslearn.data_sources import DataSource, DataSourceContext, Item
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.geometry import STGeometry

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

    def deserialize_item(self, serialized_item: Any) -> Item:
        """Deserializes an item from JSON-decoded data."""
        assert isinstance(serialized_item, dict)
        return Item.deserialize(serialized_item)

    def _convert_nc_to_tif(self, nc_path: UPath, tif_path: UPath) -> None:
        """Convert a netCDF file to a GeoTIFF file.

        Args:
            nc_path: the path to the netCDF file
            tif_path: the path to the output GeoTIFF file
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
            band_array = np.expand_dims(band_array, axis=1)
            band_arrays.append(band_array)

        # After concatenation: (time, num_variables, height, width)
        stacked_array = np.concatenate(band_arrays, axis=1)

        # After reshaping: (time x num_variables, height, width)
        array = stacked_array.reshape(
            -1, stacked_array.shape[2], stacked_array.shape[3]
        )

        # Get metadata for the GeoTIFF
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
        west = lon.min() - self.PIXEL_SIZE / 2
        north = lat.max() + self.PIXEL_SIZE / 2
        pixel_size_x, pixel_size_y = self.PIXEL_SIZE, self.PIXEL_SIZE
        transform = from_origin(west, north, pixel_size_x, pixel_size_y)
        crs = f"EPSG:{WGS84_EPSG}"
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=array.shape[1],
            width=array.shape[2],
            count=array.shape[0],
            dtype=array.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(array)

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
                local_tif_fname = os.path.join(tmp_dir, f"{item.name}.tif")
                self.client.retrieve(self.dataset, request, local_nc_fname)
                self._convert_nc_to_tif(
                    UPath(local_nc_fname),
                    UPath(local_tif_fname),
                )
                tile_store.write_raster_file(
                    item.name, self.band_names, UPath(local_tif_fname)
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
                local_tif_fname = os.path.join(tmp_dir, f"{item.name}.tif")
                self.client.retrieve(self.dataset, request, local_nc_fname)
                self._convert_nc_to_tif(
                    UPath(local_nc_fname),
                    UPath(local_tif_fname),
                )
                tile_store.write_raster_file(
                    item.name, self.band_names, UPath(local_tif_fname)
                )
