"""Data source for Copernicus Climate Data Store."""

import math
import os
import tempfile
from datetime import datetime, timezone
from typing import Any

import cdsapi
import netCDF4
import numpy as np
import rasterio
import shapely
from dateutil.relativedelta import relativedelta
from rasterio.transform import from_origin
from upath import UPath

from rslearn.config import QueryConfig, RasterLayerConfig, SpaceMode
from rslearn.const import WGS84_BOUNDS, WGS84_EPSG, WGS84_PROJECTION
from rslearn.data_sources import DataSource, Item
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.geometry import STGeometry

logger = get_logger(__name__)


class ERA5LandMonthlyMeans(DataSource):
    """A data source for ingesting ERA5 land monthly averaged data from the Copernicus Climate Data Store.

    An API key must be passed either in the configuration or via the CDSAPI_KEY
    environment variable. You can acquire an API key by going to the Climate Data Store
    website (https://cds.climate.copernicus.eu/), registering an account and logging
    in, and then

    The band names should match CDS variable names (see the reference at
    https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation). However,
    replace "_" with "-" in the variable names when specifying bands in the layer
    configuration.

    This data source corresponds to the reanalysis-era5-land-monthly-means product.

    All requests to the API will be for patches corresponding to the specified patch
    size (defaulting to 10x10 degrees). The patch size defines a grid in WGS84
    coordinates, and windows are matched to items that correspond to these grid cells.
    (Temporally, the items are gridded by month.) Although the API supports arbitrary
    bounds in the requests, using the fixed grid helps to minimize duplicative requests
    across windows.
    """

    api_url = "https://cds.climate.copernicus.eu/api"

    # see: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land-monthly-means
    DATASET = "reanalysis-era5-land-monthly-means"
    PRODUCT_TYPE = "monthly_averaged_reanalysis"
    DATA_FORMAT = "netcdf"
    DOWNLOAD_FORMAT = "unarchived"
    PIXEL_SIZE = 0.1  # degrees, native resolution is 9km

    def __init__(
        self,
        band_names: list[str],
        api_key: str | None = None,
        patch_size: float = 10,
    ):
        """Initialize a new ERA5LandMonthlyMeans instance.

        Args:
            band_names: list of band names to acquire. These should correspond to CDS
                variable names but with "_" replaced with "-".
            api_key: the API key. If not set, it should be set via the CDSAPI_KEY
                environment variable.
            patch_size: the size of items to request from the data store, in degrees.
        """
        self.band_names = band_names
        self.patch_size = patch_size

        self.client = cdsapi.Client(
            url=self.api_url,
            key=api_key,
        )

    @staticmethod
    def from_config(
        config: RasterLayerConfig, ds_path: UPath
    ) -> "ERA5LandMonthlyMeans":
        """Creates a new ERA5LandMonthlyMeans instance from a configuration dictionary.

        Args:
            config: the LayerConfig of the layer containing this data source
            ds_path: the path to the data source

        Returns:
            A new ERA5LandMonthlyMeans instance
        """
        if config.data_source is None:
            raise ValueError("data_source is required")
        d = config.data_source.config_dict

        # Determine band names based on the configured band sets.
        band_names = []
        for band_set in config.band_sets:
            for band in band_set.bands:
                if band in band_names:
                    continue
                band_names.append(band)
        kwargs: dict[str, Any] = dict(
            band_names=band_names,
        )

        simple_optionals = ["api_key", "patch_size"]
        for k in simple_optionals:
            if k in d:
                kwargs[k] = d[k]

        return ERA5LandMonthlyMeans(**kwargs)

    def _tile_to_item(self, col: int, row: int, year: int, month: int) -> Item:
        """Get the Item for the given column and row on our fixed grid.

        Args:
            col: the column.
            row: the row.
            year: the year.
            month: the month.

        Returns:
            the corresponding Item.
        """
        item_name = f"{col}_{row}_{year}_{month}"

        # The bounds is based on the grid.
        bounds = (
            col * self.patch_size,
            row * self.patch_size,
            (col + 1) * self.patch_size,
            (row + 1) * self.patch_size,
        )

        # Time is just the given month.
        start_date = datetime(year, month, 1, tzinfo=timezone.utc)
        time_range = (
            start_date,
            start_date + relativedelta(months=1),
        )

        geometry = STGeometry(
            WGS84_PROJECTION,
            shapely.box(*bounds),
            time_range,
        )
        return Item(item_name, geometry)

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
                tzinfo=timezone.utc,
            )
            end_date = datetime(
                geometry.time_range[1].year,
                geometry.time_range[1].month,
                1,
                tzinfo=timezone.utc,
            )

            month_dates: list[datetime] = []
            while cur_date <= end_date:
                month_dates.append(cur_date)
                cur_date += relativedelta(months=1)

            # Determine which patches on our fixed grid that this geometry intersects.
            wgs84_geometry = geometry.to_projection(WGS84_PROJECTION)
            shp_bounds = wgs84_geometry.shp.bounds
            cell_bounds = (
                math.floor(shp_bounds[0] / self.patch_size),
                math.floor(shp_bounds[1] / self.patch_size),
                math.floor(shp_bounds[2] / self.patch_size) + 1,
                math.floor(shp_bounds[3] / self.patch_size) + 1,
            )

            cur_groups = []
            for cur_date in month_dates:
                # Collect Item list corresponding to the patches and the current month.
                items = []
                for col in range(cell_bounds[0], cell_bounds[2]):
                    for row in range(cell_bounds[1], cell_bounds[3]):
                        items.append(
                            self._tile_to_item(col, row, cur_date.year, cur_date.month)
                        )
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
        bands_data = []
        for band_name in nc.variables:
            band_data = nc.variables[band_name]
            if len(band_data.shape) != 3:
                # This should be one of those variables that we want to skip.
                continue

            logger.debug(
                f"NC file {nc_path} has variable {band_name} with shape {band_data.shape}"
            )
            # Variable data is stored in a 3D array (1, height, width)
            if band_data.shape[0] != 1:
                raise ValueError(
                    f"Bad shape for band {band_name}, expected 1 band but got {band_data.shape[0]}"
                )
            bands_data.append(band_data[0, :, :])

        array = np.array(bands_data)  # (num_bands, height, width)
        if array.shape[0] != len(self.band_names):
            raise ValueError(
                f"Expected to get {len(self.band_names)} bands but got {array.shape[0]}"
            )

        # Get metadata for the GeoTIFF
        lat = nc.variables["latitude"][:]
        lon = nc.variables["longitude"][:]
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

            # Compute the bounds to request.
            # Given that ERA5 is at a very coarse resolution, we add a buffer of half a pixel size
            # to the bounds to ensure that we fully cover the geometries
            # But we don't go beyond the valid WGS84 bounds.
            bounds = (
                item.geometry.shp.bounds[0] - self.PIXEL_SIZE / 2,
                item.geometry.shp.bounds[1] - self.PIXEL_SIZE / 2,
                item.geometry.shp.bounds[2] - self.PIXEL_SIZE / 2,
                item.geometry.shp.bounds[3] - self.PIXEL_SIZE / 2,
            )
            bounds = (
                max(bounds[0], WGS84_BOUNDS[0]),
                max(bounds[1], WGS84_BOUNDS[1]),
                min(bounds[2], WGS84_BOUNDS[2]),
                min(bounds[3], WGS84_BOUNDS[3]),
            )

            # Send the request to the CDS API
            bounds = item.geometry.shp.bounds
            request = {
                "product_type": [self.PRODUCT_TYPE],
                "variable": variable_names,
                "year": [f"{item.geometry.time_range[0].year}"],  # type: ignore
                "month": [
                    f"{item.geometry.time_range[0].month:02d}"  # type: ignore
                ],
                "time": ["00:00"],
                "area": [
                    bounds[3],  # North
                    bounds[0],  # West
                    bounds[1],  # South
                    bounds[2],  # East
                ],
                "data_format": self.DATA_FORMAT,
                "download_format": self.DOWNLOAD_FORMAT,
            }
            logger.debug(
                f"CDS API request for bounds {request['area']} and year={request['year']} month={request['month']}"
            )
            with tempfile.TemporaryDirectory() as tmp_dir:
                local_nc_fname = os.path.join(tmp_dir, f"{item.name}.nc")
                local_tif_fname = os.path.join(tmp_dir, f"{item.name}.tif")
                self.client.retrieve(self.DATASET, request, local_nc_fname)
                self._convert_nc_to_tif(
                    UPath(local_nc_fname),
                    UPath(local_tif_fname),
                )
                tile_store.write_raster_file(
                    item.name, self.band_names, UPath(local_tif_fname)
                )
