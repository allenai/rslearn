"""Elevation data from the Shuttle Radar Topography Mission via NASA Earthdata."""

import math
import os
import tempfile
import zipfile
from typing import Any

import requests
import requests.auth
import shapely
from upath import UPath

from rslearn.config import QueryConfig, RasterLayerConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import DataSource, Item
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.geometry import STGeometry

logger = get_logger(__name__)


class SRTM(DataSource):
    """Data source for SRTM elevation data using NASA Earthdata credentials.

    See https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/ and
    https://dwtkns.com/srtm30m/ for details about the data.

    The data is split into 1x1-degree tiles, where the filename ends with e.g.
    S28W055.SRTMGL1.hgt.zip (so only the first seven characters change).

    These URLs can only be accessed with a NASA Earthdata username and password.

    The zip file contains a single hgt file which can be read by rasterio. It has a
    single 16-bit signed integer band indicating the elevation.

    Items from this data source do not come with a time range. The band name will match
    that specified in the band set, which should have a single band.
    """

    BASE_URL = "https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/"
    FILENAME_SUFFIX = ".SRTMGL1.hgt.zip"

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        band_name: str = "srtm",
        timeout: int = 10,
    ):
        """Initialize a new SRTM instance.

        Args:
            username: NASA Earthdata account username. If not set, it is read from the
                NASA_EARTHDATA_USERNAME environment variable.
            password: NASA Earthdata account password. If not set, it is read from the
                NASA_EARTHDATA_PASSWORD environment variable.
            band_name: what to call the band.
            timeout: timeout for downloading data.
        """
        self.band_name = band_name
        self.timeout = timeout

        if username is None:
            username = os.environ["NASA_EARTHDATA_USERNAME"]
        self.username = username

        if password is None:
            password = os.environ["NASA_EARTHDATA_PASSWORD"]
        self.password = password

        self.session = requests.session()

    @staticmethod
    def from_config(config: RasterLayerConfig, ds_path: UPath) -> "SRTM":
        """Creates a new SRTM instance from a configuration dictionary."""
        if config.data_source is None:
            raise ValueError("config.data_source is required")
        d = config.data_source.config_dict
        if len(config.band_sets[0].bands) != 1:
            raise ValueError("expected band set to have a single band")
        kwargs: dict[str, Any] = {
            "band_name": config.band_sets[0].bands[0],
        }
        simple_optionals = ["username", "password"]
        for k in simple_optionals:
            if k in d:
                kwargs[k] = d[k]

        return SRTM(**kwargs)

    def get_item_by_name(self, name: str) -> Item:
        """Gets an item by name.

        Args:
            name: the name of the item to get. For SRTM, the item name is the filename
                of the zip file containing the hgt file.

        Returns:
            the Item object
        """
        if not name.endswith(self.FILENAME_SUFFIX):
            raise ValueError(
                f"expected item name to end with {self.FILENAME_SUFFIX}, but got {name}"
            )
        # Parse the first seven characters, e.g. S28W055.
        # We do this to reconstruct the geometry of the item.
        lat_sign = name[0]
        lat_degrees = int(name[1:3])
        lon_sign = name[4]
        lon_degrees = int(name[5:8])

        if lat_sign == "N":
            lat_min = lat_degrees
        elif lat_sign == "S":
            lat_min = -lat_degrees
        else:
            raise ValueError(f"invalid item name {name}")

        if lon_sign == "E":
            lon_min = lon_degrees
        elif lon_sign == "W":
            lon_min = -lon_degrees
        else:
            raise ValueError(f"invalid item name {name}")

        geometry = STGeometry(
            WGS84_PROJECTION,
            shapely.box(lon_min, lat_min, lon_min + 1, lat_min + 1),
            None,
        )
        return Item(name, geometry)

    def _lon_lat_to_item(self, lon_min: int, lat_min: int) -> Item:
        """Get an item based on the 1x1 longitude/latitude grid.

        Args:
            lon_min: the starting longitude integer of the grid cell.
            lat_min: the starting latitude integer of the grid cell.

        Returns:
            the Item object.
        """
        # Construct the filename for this grid cell.
        # The item name is just the filename.
        if lon_min < 0:
            lon_part = f"W{-lon_min:03d}"
        else:
            lon_part = f"E{lon_min:03d}"
        if lat_min < 0:
            lat_part = f"S{-lat_min:02d}"
        else:
            lat_part = f"N{lat_min:02d}"
        fname = lat_part + lon_part + self.FILENAME_SUFFIX

        # We also need the geometry for the item.
        geometry = STGeometry(
            WGS84_PROJECTION,
            shapely.box(lon_min, lat_min, lon_min + 1, lat_min + 1),
            None,
        )

        return Item(fname, geometry)

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
        # We only support mosaic here, other query modes don't really make sense.
        if query_config.space_mode != SpaceMode.MOSAIC or query_config.max_matches != 1:
            raise ValueError(
                "expected mosaic with max_matches=1 for the query configuration"
            )

        groups = []
        for geometry in geometries:
            # We iterate over each 1x1 cell that this geometry intersects and include
            # the corresponing item in this item group.
            # Since it is a mosaic with one match, there will just be one item group
            # for each item.
            wgs84_geometry = geometry.to_projection(WGS84_PROJECTION)
            shp_bounds = wgs84_geometry.shp.bounds
            cell_bounds = (
                math.floor(shp_bounds[0]),
                math.floor(shp_bounds[1]),
                math.ceil(shp_bounds[2]),
                math.ceil(shp_bounds[3]),
            )
            # lon_min/lat_min are the lower range of each cell.
            items = []
            for lon_min in range(cell_bounds[0], cell_bounds[2]):
                for lat_min in range(cell_bounds[1], cell_bounds[3]):
                    items.append(self._lon_lat_to_item(lon_min, lat_min))

            logger.debug(f"Got {len(items)} items (grid cells) for geometry")
            groups.append([items])

        return groups

    def deserialize_item(self, serialized_item: Any) -> Item:
        """Deserializes an item from JSON-decoded data."""
        assert isinstance(serialized_item, dict)
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

            # Download the item.
            # We first attempt to access it directly, which works if we have already
            # authenticated. If not, we get redirected to a login endpoint where we
            # need to use basic authentication; the endpoint will redirect us back to
            # the original URL.
            url = self.BASE_URL + item.name
            logger.debug(f"Downloading SRTM data for {item.name} from {url}")

            # Try to access directly.
            response = self.session.get(
                url, stream=True, timeout=self.timeout, allow_redirects=False
            )

            if response.status_code == 302:
                # Encountered redirect, so set response to actually access the redirect
                # URL. This time we follow redirects since it will take us back to the
                # original URL.
                redirect_url = response.headers["Location"]
                logger.debug(f"Following redirect to {redirect_url}")
                auth = requests.auth.HTTPBasicAuth(self.username, self.password)
                response = self.session.get(
                    redirect_url, stream=True, timeout=self.timeout, auth=auth
                )

            if response.status_code == 404:
                # Some grid cells don't exist so this isn't a big issue.
                logger.warning(
                    f"Skipping item {item.name} because there is no data at that cell"
                )
                continue
            response.raise_for_status()

            with tempfile.TemporaryDirectory() as tmp_dir:
                # Store it in temporary directory.
                zip_fname = os.path.join(tmp_dir, "data.zip")
                with open(zip_fname, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Extract the .hgt file.
                logger.debug(f"Extracting data for {item.name}")
                with zipfile.ZipFile(zip_fname) as zip_f:
                    member_names = zip_f.namelist()
                    if len(member_names) != 1:
                        raise ValueError(
                            f"expected SRTM zip to have one member but got {member_names}"
                        )
                    local_fname = zip_f.extract(member_names[0], path=tmp_dir)

                # Now we can ingest it.
                logger.debug(f"Ingesting data for {item.name}")
                tile_store.write_raster_file(
                    item.name, [self.band_name], UPath(local_fname)
                )
