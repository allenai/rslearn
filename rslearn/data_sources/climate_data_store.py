"""Data source for Copernicus Climate Data Store."""

import os
import tempfile

import cdsapi
import netCDF4
import numpy as np
import rasterio
from dateutil.relativedelta import relativedelta
from rasterio.transform import from_origin
from upath import UPath

from rslearn.config import QueryConfig, RasterLayerConfig
from rslearn.const import WGS84_EPSG, WGS84_PROJECTION
from rslearn.data_sources import DataSource, Item
from rslearn.data_sources.utils import match_candidate_items_to_window
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils import STGeometry

logger = get_logger(__name__)


class ERA5LandMonthlyMeans(DataSource):
    """A data source for ingesting ERA5 land monthly averaged data from the Copernicus Climate Data Store.

    The API key should be set via environment variable (CDS_API_KEY).
    """

    api_url = "https://cds.climate.copernicus.eu/api"

    # see: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land-monthly-means
    DATASET = "reanalysis-era5-land-monthly-means"
    PRODUCT_TYPE = "monthly_averaged_reanalysis"
    PIXEL_SIZE = 0.1  # degrees, native resolution is 9km

    # see: https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation
    # For variable full & short names

    def __init__(
        self,
        config: RasterLayerConfig,
        api_key: str | None = None,
    ):
        """Initialize a new ERA5LandMonthlyMeans instance.

        Args:
            config: the LayerConfig of the layer containing this data source
            api_key: the API key
        """
        self.config = config
        self.dataset = self.DATASET
        self.product_type = self.PRODUCT_TYPE

        if api_key is None:
            api_key = os.environ["CDS_API_KEY"]
        # Follow CDS API docs: https://pypi.org/project/cdsapi/
        # Copy both url and key to $HOME/.cdsapirc
        with open(os.path.expanduser("~/.cdsapirc"), "w") as f:
            f.write(f"url: {self.api_url}\nkey: {api_key}")

        self.client = cdsapi.Client()

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
        return ERA5LandMonthlyMeans(
            config=config,
            api_key=d["api_key"],
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
        wgs84_geometries = [
            geometry.to_projection(WGS84_PROJECTION) for geometry in geometries
        ]
        groups = []
        for geometry in wgs84_geometries:
            cur_items = []
            # Create an item for each year & month within the time range
            start_date = geometry.time_range[0].replace(day=1)  # type: ignore
            end_date = geometry.time_range[1].replace(day=1)  # type: ignore
            while start_date <= end_date:
                # Get the first and last day of the current year & month
                # Use this as the item's time range
                item_start_date, item_end_date = (
                    start_date,
                    start_date + relativedelta(months=1) - relativedelta(days=1),
                )
                item_geometry = STGeometry(
                    geometry.projection, geometry.shp, (item_start_date, item_end_date)
                )
                item_bounds = item_geometry.shp.bounds
                item_name = f"{item_bounds[0]}_{item_bounds[1]}_{item_start_date.year}_{item_start_date.month:02d}"
                item = Item(item_name, item_geometry)
                cur_items.append(item)
                start_date += relativedelta(months=1)
            cur_groups = match_candidate_items_to_window(
                geometry, cur_items, query_config
            )
            groups.append(cur_groups)

        return groups

    def _convert_nc_to_tif(self, nc_path: UPath, tif_path: UPath) -> None:
        """Convert a netCDF file to a GeoTIFF file.

        Args:
            nc_path: the path to the netCDF file
            tif_path: the path to the output GeoTIFF file
        """
        nc = netCDF4.Dataset(nc_path)
        bands_data = []
        for band_name in nc.variables:
            band_data = nc.variables[band_name]
            # Variable data is stored in a 3D array (1, height, width)
            if len(band_data.shape) == 3:
                bands_data.append(band_data[0, :, :])
        tif_data = np.array(bands_data)  # (num_bands, height, width)
        # Get metadata for the GeoTIFF
        lat = nc.variables["latitude"][:]
        lon = nc.variables["longitude"][:]
        west = lon.min()
        north = lat.max()
        pixel_size_x, pixel_size_y = self.PIXEL_SIZE, self.PIXEL_SIZE
        transform = from_origin(west, north, pixel_size_x, pixel_size_y)
        crs = f"EPSG:{WGS84_EPSG}"
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=tif_data.shape[1],
            width=tif_data.shape[2],
            count=tif_data.shape[0],
            dtype=tif_data.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            for i in range(tif_data.shape[0]):
                dst.write(tif_data[i, :, :], i + 1)  # Write each band to the GeoTIFF

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
        bands = []
        for band_set in self.config.band_sets:
            if band_set.bands is None:
                continue
            for band in band_set.bands:
                if band in bands:
                    continue
                bands.append(band)

        for item in items:
            if tile_store.is_raster_ready(item.name, bands):
                continue

            # Send the request to the CDS API
            bounds = item.geometry.shp.bounds
            request = {
                "product_type": [self.product_type],
                "variable": bands,
                "year": [f"{item.geometry.time_range[0].year}"],  # type: ignore
                "month": [
                    f"{item.geometry.time_range[0].month:02d}"  # type: ignore
                ],
                "time": ["00:00"],
                "area": [
                    bounds[3],
                    bounds[0],
                    bounds[1],
                    bounds[2],
                ],
                "data_format": "netcdf",
                "download_format": "unarchived",
            }
            with tempfile.TemporaryDirectory() as tmp_dir:
                local_nc_fname = os.path.join(tmp_dir, f"{item.name}.nc")
                local_tif_fname = os.path.join(tmp_dir, f"{item.name}.tif")
                self.client.retrieve(self.dataset, request, local_nc_fname)
                self._convert_nc_to_tif(
                    UPath(local_nc_fname),
                    UPath(local_tif_fname),
                )
                tile_store.write_raster_file(item.name, bands, UPath(local_tif_fname))
