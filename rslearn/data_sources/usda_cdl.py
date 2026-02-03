"""Crop type data from the USDA Cropland Data Layer."""

import os
import tempfile
import zipfile
from datetime import UTC, datetime, timedelta

import requests
import requests.auth
import shapely
from upath import UPath

from rslearn.config import QueryConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import DataSource, DataSourceContext, Item
from rslearn.data_sources.utils import match_candidate_items_to_window
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.geometry import STGeometry

logger = get_logger(__name__)


class CDL(DataSource):
    """Data source for crop type data from the USDA Cropland Data Layer.

    See https://www.nass.usda.gov/Research_and_Science/Cropland/SARS1a.php for details
    about the data.

    There is one GeoTIFF item per year from 2008. Each GeoTIFF spans the entire
    continental US, and has a single band.
    """

    BASE_URL = (
        "https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/"
    )
    ZIP_FILENAMES = {
        2024: "2024_30m_cdls.zip",
        2023: "2023_30m_cdls.zip",
        2022: "2022_30m_cdls.zip",
        2021: "2021_30m_cdls.zip",
        2020: "2020_30m_cdls.zip",
        2019: "2019_30m_cdls.zip",
        2018: "2018_30m_cdls.zip",
        2017: "2017_30m_cdls.zip",
        2016: "2016_30m_cdls.zip",
        2015: "2015_30m_cdls.zip",
        2014: "2014_30m_cdls.zip",
        2013: "2013_30m_cdls.zip",
        2012: "2012_30m_cdls.zip",
        2011: "2011_30m_cdls.zip",
        2010: "2010_30m_cdls.zip",
        2009: "2009_30m_cdls.zip",
        2008: "2008_30m_cdls.zip",
    }

    # The bounds of each GeoTIFF in WGS84 coordinates, based on the 2023 map.
    BOUNDS = shapely.box(-127.9, 23.0, -65.3, 48.3)

    def __init__(
        self,
        timeout: timedelta = timedelta(seconds=10),
        context: DataSourceContext = DataSourceContext(),
    ):
        """Initialize a new CDL instance.

        Args:
            timeout: timeout for requests.
            context: the data source context.
        """
        self.timeout = timeout

        # Get the band name from the layer config, which should have a single band set
        # with a single band. If the layer config is not available in the context, we
        # default to "cdl".
        if context.layer_config is not None:
            if len(context.layer_config.band_sets) != 1:
                raise ValueError("expected a single band set")
            if len(context.layer_config.band_sets[0].bands) != 1:
                raise ValueError("expected band set to have a single band")
            self.band_name = context.layer_config.band_sets[0].bands[0]
        else:
            self.band_name = "cdl"

    def get_item_by_name(self, name: str) -> Item:
        """Gets an item by name.

        Args:
            name: the name of the item to get. For CDL, the item name is the filename
                of the zip file containing the per-year GeoTIFF.

        Returns:
            the Item object
        """
        year = int(name[0:4])
        geometry = STGeometry(
            WGS84_PROJECTION,
            self.BOUNDS,
            (
                datetime(year, 1, 1, tzinfo=UTC),
                datetime(year + 1, 1, 1, tzinfo=UTC),
            ),
        )
        return Item(name, geometry)

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[Item]]]:
        """Get a list of items in the data source intersecting the given geometries.

        Args:
            geometries: the spatiotemporal geometries
            query_config: the query configuration

        Returns:
            List of groups of items that should be retrieved for each geometry.
        """
        # First enumerate all items.
        # Then we simply pass this to match_candidate_items_to_window.
        items: list[Item] = []
        for year, fname in self.ZIP_FILENAMES.items():
            geometry = STGeometry(
                WGS84_PROJECTION,
                self.BOUNDS,
                (
                    datetime(year, 1, 1, tzinfo=UTC),
                    datetime(year + 1, 1, 1, tzinfo=UTC),
                ),
            )
            items.append(Item(fname, geometry))

        groups = []
        for geometry in geometries:
            cur_groups = match_candidate_items_to_window(geometry, items, query_config)
            groups.append(cur_groups)

        return groups

    def deserialize_item(self, serialized_item: dict) -> Item:
        """Deserializes an item from JSON-decoded data."""
        return Item.deserialize(serialized_item)

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
        for item in items:
            if tile_store.is_raster_ready(item.name, [self.band_name]):
                continue

            # Download the zip file.
            url = self.BASE_URL + item.name
            logger.debug(f"Downloading CDL GeoTIFF from {url}")
            response = requests.get(
                url, stream=True, timeout=self.timeout.total_seconds()
            )
            response.raise_for_status()

            with tempfile.TemporaryDirectory() as tmp_dir:
                # Store it in temporary directory.
                zip_fname = os.path.join(tmp_dir, "data.zip")
                with open(zip_fname, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Extract the .tif file.
                logger.debug(f"Extracting GeoTIFF from {item.name}")
                with zipfile.ZipFile(zip_fname) as zip_f:
                    candidate_member_names = [
                        member_name
                        for member_name in zip_f.namelist()
                        if member_name.endswith(".tif")
                    ]
                    if len(candidate_member_names) != 1:
                        raise ValueError(
                            f"expected CDL zip to have one .tif file but got {candidate_member_names}"
                        )
                    local_fname = zip_f.extract(candidate_member_names[0], path=tmp_dir)

                # Now we can ingest it.
                logger.debug(f"Ingesting data for {item.name}")
                tile_store.write_raster_file(
                    item.name, [self.band_name], UPath(local_fname)
                )
