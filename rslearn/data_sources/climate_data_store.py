"""Data source for Copernicus Climate Data Store."""

import os
import re
import tempfile
import zipfile
from datetime import UTC, datetime

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


# Maximum number of hours in any month (31 days × 24 hours).
# Used to pad shorter months so that every monthly item has a fixed band count.
MAX_HOURS_PER_MONTH = 31 * 24  # 744

# Fill value used for padded timesteps and to replace NaN from the source data.
FILL_VALUE = -9999.0


def expand_hourly_band_names(base_bands: list[str]) -> list[str]:
    """Expand base variable names into timestep-indexed band names.

    The ordering matches the reshape in _convert_nc_to_tif: for each timestep t,
    for each variable v, the band name is "{v}_t{t:03d}".

    Args:
        base_bands: the base variable names
            (e.g. ["2m-temperature", "total-precipitation"])

    Returns:
        A flat list of band names with length MAX_HOURS_PER_MONTH * len(base_bands).
    """
    expanded = []
    for t in range(MAX_HOURS_PER_MONTH):
        for band in base_bands:
            expanded.append(f"{band}_t{t:03d}")
    return expanded


def extract_base_band_names(bands: list[str]) -> list[str]:
    """Extract unique base names by stripping ``_t{timestep:03d}`` suffixes.

    If the bands were produced by `expand_hourly_band_names` this
    reverses the expansion and returns the original base names in
    order.

    Args:
        bands: a list of band names, either expanded (e.g.
            ``["2m-temperature_t000", "2m-temperature_t001", ...]``) or
            already base names (e.g. ``["2m-temperature"]``).

    Returns:
        An ordered, deduplicated list of base names.
    """
    seen: set[str] = set()
    base_names: list[str] = []
    for band in bands:
        base = re.compile(r"_t\d{3}$").sub("", band)
        if base not in seen:
            seen.add(base)
            base_names.append(base)
    return base_names


def _resolve_fill_value(context: "DataSourceContext") -> float:
    """Resolve the fill/nodata value from the layer config, falling back to FILL_VALUE.

    When a layer config is available, the first non-None ``nodata_vals`` entry
    from the first band set is used.  All nodata values within a band set are
    expected to be identical for ERA5 data (they are broadcast from a single
    scalar in the config), so only the first entry is inspected.

    Args:
        context: the data source context, which may contain a layer config.

    Returns:
        The fill value to use for padding and NaN replacement.
    """
    if context.layer_config is not None:
        for band_set in context.layer_config.band_sets:
            if isinstance(band_set.nodata_vals, list) and len(band_set.nodata_vals) > 0:
                return float(band_set.nodata_vals[0])
    return FILL_VALUE


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

    def deserialize_item(self, serialized_item: dict) -> Item:
        """Deserializes an item from JSON-decoded data."""
        return Item.deserialize(serialized_item)

    def _convert_nc_to_tif(
        self,
        nc_path: UPath,
        tif_path: UPath,
        max_time_steps: int | None = None,
        fill_value: float = FILL_VALUE,
    ) -> None:
        """Convert a netCDF file to a GeoTIFF file.

        Args:
            nc_path: the path to the netCDF file
            tif_path: the path to the output GeoTIFF file
            max_time_steps: if set, pad the time dimension to this many steps using
                fill_value. This ensures a fixed band count regardless of the actual
                number of timesteps in the data (e.g. different month lengths).
            fill_value: the value used for padding and for replacing NaN in the
                source data. Should match the nodata_vals in the dataset config
                so that materialization correctly identifies invalid pixels.
                Defaults to FILL_VALUE (-9999.0).
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

        # Pad time dimension to max_time_steps if specified, using fill_value
        if max_time_steps is not None:
            actual_steps = stacked_array.shape[0]
            if actual_steps > max_time_steps:
                raise ValueError(
                    f"Data has {actual_steps} time steps but max_time_steps={max_time_steps}"
                )
            if actual_steps < max_time_steps:
                pad_shape = (
                    max_time_steps - actual_steps,
                    stacked_array.shape[1],
                    stacked_array.shape[2],
                    stacked_array.shape[3],
                )
                pad_array = np.full(pad_shape, fill_value, dtype=stacked_array.dtype)
                stacked_array = np.concatenate([stacked_array, pad_array], axis=0)

        # After reshaping: (time x num_variables, height, width)
        array = stacked_array.reshape(
            -1, stacked_array.shape[2], stacked_array.shape[3]
        )

        # Replace any NaN values from the source data with fill_value
        array = np.nan_to_num(array, nan=fill_value)

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

    Each monthly item is padded to MAX_HOURS_PER_MONTH (744) timesteps so that
    the band count is fixed regardless of month length. Shorter months are padded
    with the nodata/fill value. Band names are auto-expanded from the base names
    names: for each timestep t and variable v, the band is named "{v}_t{t:03d}".

    The fill value used for padding and NaN replacement is read from the layer
    config's ``nodata_vals`` when available, otherwise defaults to FILL_VALUE
    (-9999.0).  The dataset config should always set ``nodata_vals`` (required
    when ``num_timesteps`` is set) to keep the data source and materialisation
    in sync.
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
            band_names: list of base variable names to acquire (e.g.
                ["2m-temperature", "total-precipitation"]). These should correspond
                to CDS variable names but with "_" replaced with "-". Only needed
                when no layer config is available in the context; when the layer
                config is present the base names are derived automatically from
                the (possibly expanded) band names in the config.
            api_key: the API key. If not set, it should be set via the CDSAPI_KEY
                environment variable.
            bounds: optional bounding box as [min_lon, min_lat, max_lon, max_lat].
                If not specified, the whole globe will be used.
            context: the data source context.
        """
        self.fill_value = _resolve_fill_value(context)
        super().__init__(
            dataset="reanalysis-era5-land",
            product_type="reanalysis",
            band_names=band_names,
            api_key=api_key,
            bounds=bounds,
            context=context,
        )
        # self.band_names was set by the base class — either from
        # context.layer_config (expanded names) or from the band_names
        # parameter (base names).  Derive the base variable names and then
        # re-expand so that both attributes are always consistent.
        self.base_band_names = extract_base_band_names(self.band_names)
        self.band_names = expand_hourly_band_names(self.base_band_names)

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
        variable_names = [band.replace("-", "_") for band in self.base_band_names]

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
                    max_time_steps=MAX_HOURS_PER_MONTH,
                    fill_value=self.fill_value,
                )
                tile_store.write_raster_file(
                    item.name, self.band_names, UPath(local_tif_fname)
                )


class ERA5LandHourlyTimeseries(DataSource):
    """A data source for ingesting ERA5-Land hourly time-series data for individual points.

    This data source corresponds to the reanalysis-era5-land-timeseries dataset, which is
    optimized for retrieving long time-series data for single points rather than spatial
    areas. It uses a 0.1 degree grid and automatically snaps requested coordinates to the
    nearest grid cell center to avoid duplicate requests.

    Each monthly item is padded to MAX_HOURS_PER_MONTH (744) timesteps so that
    the band count is fixed regardless of month length. Shorter months are padded
    with the nodata/fill value. Band names are auto-expanded from the base variable
    names: for each timestep t and variable v, the band is named "{v}_t{t:03d}".

    The fill value used for padding and NaN replacement is read from the layer
    config's ``nodata_vals`` when available, otherwise defaults to FILL_VALUE
    (-9999.0).  The dataset config should always set ``nodata_vals`` (required
    when ``num_timesteps`` is set) to keep the data source and materialisation
    in sync.

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

    def __init__(
        self,
        band_names: list[str] | None = None,
        api_key: str | None = None,
        context: DataSourceContext = DataSourceContext(),
    ):
        """Initialize a new ERA5LandHourlyTimeseries instance.

        Args:
            band_names: list of base variable names to acquire (e.g.
                ["2m-temperature", "total-precipitation"]). These should correspond
                to CDS variable names but with "_" replaced with "-". Only needed
                when no layer config is available in the context; when the layer
                config is present the base names are derived automatically from
                the (possibly expanded) band names in the config.
            api_key: the API key. If not set, it should be set via the CDSAPI_KEY
                environment variable.
            context: the data source context.
        """
        # Resolve band names from config or from the explicit parameter.
        raw_bands: list[str]
        if context.layer_config is not None:
            raw_bands = [
                band
                for band_set in context.layer_config.band_sets
                for band in band_set.bands
            ]
        elif band_names is not None:
            raw_bands = band_names
        else:
            raise ValueError(
                "band_names must be set if layer_config is not in the context"
            )

        self.base_band_names = extract_base_band_names(raw_bands)
        self.band_names = expand_hourly_band_names(self.base_band_names)
        self.fill_value = _resolve_fill_value(context)

        self.client = cdsapi.Client(
            url=self.API_URL,
            key=api_key,
        )

    def _snap_to_grid(self, lon: float, lat: float) -> tuple[float, float]:
        """Snap coordinates to the nearest 0.1 degree grid cell center.

        The ERA5-Land timeseries dataset uses a 0.1 degree grid. When a requested
        location doesn't exactly match a grid point, the API automatically selects
        the nearest grid point. This method pre-computes the snapped coordinates
        to enable proper caching and avoid duplicate requests.

        Args:
            lon: the longitude in degrees
            lat: the latitude in degrees

        Returns:
            A tuple of (snapped_lon, snapped_lat) rounded to 1 decimal place.
        """
        snapped_lon = round(round(lon / self.PIXEL_SIZE) * self.PIXEL_SIZE, 1)
        snapped_lat = round(round(lat / self.PIXEL_SIZE) * self.PIXEL_SIZE, 1)
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

    def _convert_nc_to_tif(
        self, nc_paths: list[UPath], tif_path: UPath, lon: float, lat: float
    ) -> None:
        """Convert timeseries netCDF files to a GeoTIFF file.

        The timeseries API may return multiple NetCDF files (one per variable).
        This method reads data from all files and combines them into a single
        1x1 pixel GeoTIFF with time steps encoded as bands. The time dimension
        is padded to MAX_HOURS_PER_MONTH so that every monthly item has a fixed
        band count, and any NaN values are replaced with FILL_VALUE.

        Args:
            nc_paths: list of paths to the netCDF files
            tif_path: the path to the output GeoTIFF file
            lon: the longitude of the point
            lat: the latitude of the point
        """
        # The timeseries files contain 1D arrays for each variable (time dimension only)
        # We need to collect these from all files and stack them into a multi-band raster
        band_arrays = []
        num_time_steps = None

        for nc_path in nc_paths:
            nc = netCDF4.Dataset(nc_path)

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

                # Get the 1D array and reshape to (time, 1) for stacking
                band_array = np.array(band_data[:])
                band_array = np.expand_dims(band_array, axis=1)
                band_arrays.append(band_array)

            nc.close()

        if not band_arrays:
            raise ValueError(f"No valid band data found in {nc_paths}")

        # Stack arrays: shape becomes (time, num_variables)
        stacked_array = np.concatenate(band_arrays, axis=1)

        # Pad time dimension to MAX_HOURS_PER_MONTH using self.fill_value
        actual_steps = stacked_array.shape[0]
        if actual_steps > MAX_HOURS_PER_MONTH:
            raise ValueError(
                f"Data has {actual_steps} time steps but "
                f"MAX_HOURS_PER_MONTH={MAX_HOURS_PER_MONTH}"
            )
        if actual_steps < MAX_HOURS_PER_MONTH:
            pad_shape = (MAX_HOURS_PER_MONTH - actual_steps, stacked_array.shape[1])
            pad_array = np.full(pad_shape, self.fill_value, dtype=stacked_array.dtype)
            stacked_array = np.concatenate([stacked_array, pad_array], axis=0)

        # Reshape to (MAX_HOURS_PER_MONTH * num_variables, 1, 1) for raster format
        # This creates a 1x1 pixel raster with all time steps as bands
        array = stacked_array.reshape(-1, 1, 1)

        # Replace any NaN values from the source data with self.fill_value
        array = np.nan_to_num(array, nan=self.fill_value)

        # Create a minimal geotransform for the single point
        # The point is at the center of a 0.1 degree pixel
        west = lon - self.PIXEL_SIZE / 2
        north = lat + self.PIXEL_SIZE / 2
        transform = from_origin(west, north, self.PIXEL_SIZE, self.PIXEL_SIZE)
        crs = f"EPSG:{WGS84_EPSG}"

        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=1,
            width=1,
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

        Args:
            tile_store: the tile store to ingest into
            items: the items to ingest
            geometries: a list of geometries needed for each item
        """
        # for CDS variable names, replace "-" with "_"
        variable_names = [band.replace("-", "_") for band in self.base_band_names]

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
                local_tif_fname = os.path.join(tmp_dir, f"{item.name}.tif")

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

                self._convert_nc_to_tif(
                    local_nc_paths,
                    UPath(local_tif_fname),
                    lon,
                    lat,
                )
                tile_store.write_raster_file(
                    item.name, self.band_names, UPath(local_tif_fname)
                )
