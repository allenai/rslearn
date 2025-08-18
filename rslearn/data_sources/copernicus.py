"""Data source for raster data in ESA Copernicus API."""

import functools
import io
import json
import os
import pathlib
import shutil
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any
from urllib.parse import quote
from zipfile import ZipFile

import numpy as np
import numpy.typing as npt
import rasterio
import requests
import shapely
from upath import UPath

from rslearn.config import QueryConfig, RasterLayerConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.data_source import DataSource, Item
from rslearn.data_sources.utils import match_candidate_items_to_window
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.fsspec import open_atomic
from rslearn.utils.geometry import (
    FloatBounds,
    STGeometry,
    flatten_shape,
    split_shape_at_prime_meridian,
)
from rslearn.utils.grid_index import GridIndex
from rslearn.utils.raster_format import get_raster_projection_and_bounds

SENTINEL2_TILE_URL = "https://sentiwiki.copernicus.eu/__attachments/1692737/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.zip"
SENTINEL2_KML_NAMESPACE = "{http://www.opengis.net/kml/2.2}"

logger = get_logger(__name__)


def get_harmonize_callback(
    tree: "ET.ElementTree[ET.Element[str]] | ET.Element[str]",
) -> Callable[[npt.NDArray], npt.NDArray] | None:
    """Gets the harmonization callback based on the metadata XML.

    Harmonization ensures that scenes before and after processing baseline 04.00
    are comparable. 04.00 introduces +1000 offset to the pixel values to include
    more information about dark areas.

    Args:
        tree: the parsed XML tree

    Returns:
        None if no callback is needed, or the callback to subtract the new offset
    """
    offset = None

    # The metadata will use different tag for L1C / L2A.
    # L1C: RADIO_ADD_OFFSET
    # L2A: BOA_ADD_OFFSET
    for potential_tag in ["RADIO_ADD_OFFSET", "BOA_ADD_OFFSET"]:
        for el in tree.iter(potential_tag):
            if el.text is None:
                raise ValueError(f"text is missing in {el}")
            value = int(el.text)
            if offset is None:
                offset = value
                assert offset <= 0
                # For now assert the offset is always -1000.
                assert offset == -1000
            else:
                assert offset == value

    if offset is None or offset == 0:
        return None

    def callback(array: npt.NDArray) -> npt.NDArray:
        # Subtract positive number instead of add negative number since only the former
        # works with uint16 array.
        assert array.shape[0] == 1 and array.dtype == np.uint16
        return np.clip(array, -offset, None) - (-offset)  # type: ignore

    return callback


def get_sentinel2_tile_index() -> dict[str, list[FloatBounds]]:
    """Get the Sentinel-2 tile index.

    This is a map from tile name to a list of WGS84 bounds of the tile. A tile may have
    multiple bounds if it crosses the antimeridian.
    """
    # Identify the Sentinel-2 tile names and bounds using the KML file.
    # First, download the zip file and extract and parse the KML.
    buf = io.BytesIO()
    with urllib.request.urlopen(SENTINEL2_TILE_URL) as response:
        shutil.copyfileobj(response, buf)
    buf.seek(0)
    with ZipFile(buf) as zipf:
        member_names = zipf.namelist()
        if len(member_names) != 1:
            raise ValueError(
                "Sentinel-2 tile zip file unexpectedly contains more than one file"
            )

        with zipf.open(member_names[0]) as memberf:
            tree = ET.parse(memberf)

    # Map from the tile name to a list of the longitude/latitude bounds.
    tile_index: dict[str, list[FloatBounds]] = {}

    # The KML is list of Placemark so iterate over those.
    for placemark_node in tree.iter(SENTINEL2_KML_NAMESPACE + "Placemark"):
        # The <name> node specifies the Sentinel-2 tile name.
        name_node = placemark_node.find(SENTINEL2_KML_NAMESPACE + "name")
        if name_node is None or name_node.text is None:
            raise ValueError("Sentinel-2 KML has Placemark without valid name node")

        tile_name = name_node.text

        # There may be one or more <coordinates> nodes depending on whether it is a
        # MultiGeometry. Some are polygons and some are points, but generally the
        # points just seem to be the center of the tile. So we create one polygon for
        # each coordinate list that is not a point, union them, and then split the
        # union geometry over the antimeridian.
        shapes = []
        for coord_node in placemark_node.iter(SENTINEL2_KML_NAMESPACE + "coordinates"):
            points = []
            # It is list of space-separated coordinates like:
            #   180,-73.0597374076,0 176.8646237862,-72.9914734628,0 ...
            if coord_node.text is None:
                raise ValueError("Sentinel-2 KML has coordinates node missing text")

            point_strs = coord_node.text.strip().split()
            for point_str in point_strs:
                parts = point_str.split(",")
                if len(parts) != 2 and len(parts) != 3:
                    continue

                lon = float(parts[0])
                lat = float(parts[1])
                points.append((lon, lat))

            # At least three points to get a polygon.
            if len(points) < 3:
                continue

            shapes.append(shapely.Polygon(points))

        if len(shapes) == 0:
            raise ValueError("Sentinel-2 KML has Placemark with no coordinates")

        # Now we union the shapes and split them at the antimeridian. This avoids
        # issues where the tile bounds go from -180 to 180 longitude and thus match
        # with anything at the same latitude.
        union_shp = shapely.unary_union(shapes)
        split_shapes = flatten_shape(split_shape_at_prime_meridian(union_shp))
        bounds_list: list[FloatBounds] = []
        for shp in split_shapes:
            bounds_list.append(shp.bounds)
        tile_index[tile_name] = bounds_list

    return tile_index


def _cache_sentinel2_tile_index(cache_dir: UPath) -> None:
    """Cache the tiles from SENTINEL2_TILE_URL.

    This way we just need to download it once.
    """
    json_fname = cache_dir / "tile_index.json"

    if json_fname.exists():
        return

    logger.info(f"caching list of Sentinel-2 tiles to {json_fname}")
    with open_atomic(json_fname, "w") as f:
        json.dump(get_sentinel2_tile_index(), f)


@functools.cache
def load_sentinel2_tile_index(cache_dir: UPath) -> GridIndex:
    """Load a GridIndex over Sentinel-2 tiles.

    This function is cached so the GridIndex only needs to be constructed once (per
    process).

    Args:
        cache_dir: the directory to cache the list of Sentinel-2 tiles.

    Returns:
        GridIndex over the tile names
    """
    _cache_sentinel2_tile_index(cache_dir)
    json_fname = cache_dir / "tile_index.json"
    with json_fname.open() as f:
        json_data = json.load(f)

    grid_index = GridIndex(0.5)
    for tile_name, bounds_list in json_data.items():
        for bounds in bounds_list:
            grid_index.insert(bounds, tile_name)

    return grid_index


def get_sentinel2_tiles(geometry: STGeometry, cache_dir: UPath) -> list[str]:
    """Get all Sentinel-2 tiles (like 01CCV) intersecting the given geometry.

    Args:
        geometry: the geometry to check.
        cache_dir: directory to cache the tiles.

    Returns:
        list of Sentinel-2 tile names that intersect the geometry.
    """
    tile_index = load_sentinel2_tile_index(cache_dir)
    wgs84_geometry = geometry.to_projection(WGS84_PROJECTION)
    # If the shape is a collection, it could be cutting across prime meridian.
    # So we query each component shape separately and collect the results to avoid
    # issues.
    # We assume the caller has already applied split_at_prime_meridian.
    results = set()
    for shp in flatten_shape(wgs84_geometry.shp):
        for result in tile_index.query(shp.bounds):
            assert isinstance(result, str)
            results.add(result)
    return list(results)


class ApiError(Exception):
    """An error from Copernicus API."""

    pass


class CopernicusItem(Item):
    """An item in the Copernicus data source."""

    def __init__(self, name: str, geometry: STGeometry, product_uuid: str) -> None:
        """Create a new CopernicusItem.

        Args:
            name: the item name
            geometry: the spatiotemporal item extent.
            product_uuid: the product UUID from Copernicus API.
        """
        super().__init__(name, geometry)
        self.product_uuid = product_uuid

    def serialize(self) -> dict[str, Any]:
        """Serializes the item to a JSON-encodable dictionary."""
        d = super().serialize()
        d["product_uuid"] = self.product_uuid
        return d

    @staticmethod
    def deserialize(d: dict[str, Any]) -> "CopernicusItem":
        """Deserializes an item from a JSON-decoded dictionary."""
        item = super(CopernicusItem, CopernicusItem).deserialize(d)
        return CopernicusItem(
            name=item.name,
            geometry=item.geometry,
            product_uuid=d["product_uuid"],
        )


class Copernicus(DataSource):
    """Scenes from the ESA Copernicus OData API.

    See https://documentation.dataspace.copernicus.eu/APIs/OData.html for details about
    the API and how to get an access token.
    """

    BASE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1"

    # The key in response dictionary for the next URL in paginated response.
    NEXT_LINK_KEY = "@odata.nextLink"

    # Chunk size to use when streaming a download.
    CHUNK_SIZE = 8192

    # Expected date format in filter strings.
    DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

    # URL to get access tokens.
    TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"  # nosec

    # We use a different URL for downloads. The BASE_URL would redirect to this URL but
    # it makes it difficult since requests library drops the Authorization header after
    # the redirect. So it is easier to access the DOWNLOAD_URL directly.
    DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1"

    def __init__(
        self,
        glob_to_bands: dict[str, list[str]],
        access_token: str | None = None,
        query_filter: str | None = None,
        order_by: str | None = None,
        sort_by: str | None = None,
        sort_desc: bool = False,
        timeout: float = 10,
    ):
        """Create a new Copernicus.

        Args:
            glob_to_bands: dictionary from a filename or glob string of an asset inside
                the product zip file, to the list of bands that the asset contains.
            access_token: API access token. See
                https://documentation.dataspace.copernicus.eu/APIs/OData.html for how
                to get a token. If not set, it is read from the environment variable
                COPERNICUS_ACCESS_TOKEN. If that environment variable doesn't exist,
                then we attempt to read the username/password from COPERNICUS_USERNAME
                and COPERNICUS_PASSWORD (this is useful since access tokens are only
                valid for an hour).
            password: set API username/password instead of access token.
            query_filter: filter string to include when searching for items. This will
                be appended to other name, geographic, and sensing time filters where
                applicable. For example, "Collection/Name eq 'SENTINEL-2'". See the API
                documentation for more examples.
            order_by: order by string to include when searching for items. For example,
                "ContentDate/Start asc". See the API documentation for more examples.
            sort_by: sort by the product attribute with this name. If set, attributes
                will be expanded when listing products. Note that while order_by uses
                the API to order products, the API provides limited options, and
                sort_by instead is done after the API call.
            sort_desc: for sort_by, sort in descending order instead of ascending
                order.
            timeout: timeout for requests.
        """
        self.glob_to_bands = glob_to_bands
        self.query_filter = query_filter
        self.order_by = order_by
        self.sort_by = sort_by
        self.sort_desc = sort_desc
        self.timeout = timeout

        self.username = None
        self.password = None
        self.access_token = access_token

        if self.access_token is None:
            if "COPERNICUS_ACCESS_TOKEN" in os.environ:
                self.access_token = os.environ["COPERNICUS_ACCESS_TOKEN"]
            else:
                self.username = os.environ["COPERNICUS_USERNAME"]
                self.password = os.environ["COPERNICUS_PASSWORD"]

    @staticmethod
    def from_config(config: RasterLayerConfig, ds_path: UPath) -> "Copernicus":
        """Creates a new Copernicus instance from a configuration dictionary."""
        if config.data_source is None:
            raise ValueError("config.data_source is required")
        d = config.data_source.config_dict
        kwargs: dict[str, Any] = dict(
            glob_to_bands=d["glob_to_bands"],
        )

        simple_optionals = [
            "access_token",
            "query_filter",
            "order_by",
            "sort_by",
            "sort_desc",
        ]
        for k in simple_optionals:
            if k in d:
                kwargs[k] = d[k]

        return Copernicus(**kwargs)

    def deserialize_item(self, serialized_item: Any) -> CopernicusItem:
        """Deserializes an item from JSON-decoded data."""
        assert isinstance(serialized_item, dict)
        return CopernicusItem.deserialize(serialized_item)

    def _get(self, path: str) -> dict[str, Any]:
        """Get the API path and return JSON content."""
        url = self.BASE_URL + path
        logger.debug(f"GET {url}")
        response = requests.get(url, timeout=self.timeout)
        if response.status_code != 200:
            content = str(response.content)
            raise ApiError(
                f"expected status code 200 but got {response.status_code} ({content})"
            )
        return response.json()

    def _build_filter_string(self, base_filter: str) -> str:
        """Build a filter string combining base_filter with user-provided filter.

        Args:
            base_filter: the base filter string that the caller wants to include.

        Returns:
            a filter string that combines base_filter with the optional user-provided
                filter.
        """
        if self.query_filter is None:
            return base_filter
        else:
            return f"{base_filter} and {self.query_filter}"

    def _product_to_item(self, product: dict[str, Any]) -> CopernicusItem:
        """Convert a product dictionary from API response to an Item.

        Args:
            product: the product dictionary that comes from an API response to the
                /Products endpoint.

        Returns:
            corresponding Item.
        """
        name = product["Name"]
        uuid = product["Id"]
        shp = shapely.geometry.shape(product["GeoFootprint"])
        time_range = (
            datetime.fromisoformat(product["ContentDate"]["Start"]),
            datetime.fromisoformat(product["ContentDate"]["End"]),
        )
        geom = STGeometry(WGS84_PROJECTION, shp, time_range)

        return CopernicusItem(name, geom, uuid)

    def _paginate(self, path: str) -> list[Any]:
        """Iterate over pages of responses for the given path.

        If the response includes "@odata.nextLink", then we continue to request until
        it no longer has any nextLink. The values in response["value"] must be a list
        and are concatenated.

        Args:
            path: the initial path to request. Additional requests will be made if a
                nextLink appears in the response.

        Returns:
            the concatenated values across responses.
        """
        all_values = []

        while True:
            response = self._get(path)
            all_values.extend(response["value"])
            if self.NEXT_LINK_KEY not in response:
                break

            # Use the next link, but we only want the path not the base URL.
            next_link = response[self.NEXT_LINK_KEY]
            if not next_link.startswith(self.BASE_URL):
                raise ValueError(
                    f"got next link {next_link} but it does not start with the base URL {self.BASE_URL}"
                )
            path = next_link.split(self.BASE_URL)[1]

        return all_values

    def _get_product(
        self, name: str, expand_attributes: bool = False
    ) -> dict[str, Any]:
        """Get the product dict from Copernicus API given scene name.

        Args:
            name: the scene name to get.
            expand_attributes: whether to request API to provide the attributes of the
                returned product.

        Returns:
            the decoded JSON product dict.
        """
        filter_string = self._build_filter_string(f"Name eq '{quote(name)}'")
        path = f"/Products?$filter={filter_string}"
        if expand_attributes:
            path += "&$expand=Attributes"
        response = self._get(path)
        products = response["value"]
        if len(products) != 1:
            raise ValueError(
                f"expected one product from {path} but got {len(products)}"
            )
        return products[0]

    def get_item_by_name(self, name: str) -> CopernicusItem:
        """Gets an item by name.

        Args:
            name: the name of the item to get

        Returns:
            the item object
        """
        product = self._get_product(name)
        return self._product_to_item(product)

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[CopernicusItem]]]:
        """Get a list of items in the data source intersecting the given geometries.

        Args:
            geometries: the spatiotemporal geometries
            query_config: the query configuration

        Returns:
            List of groups of items that should be retrieved for each geometry.
        """
        groups = []
        for geometry in geometries:
            # Perform a spatial + temporal search.
            # We use EPSG:4326 (WGS84) for the spatial search; the API expects WKT in
            # addition to the EPSG identifier.
            wgs84_geometry = geometry.to_projection(WGS84_PROJECTION)
            wgs84_wkt = wgs84_geometry.shp.wkt
            filter_string = (
                f"OData.CSC.Intersects(area=geography'SRID=4326;{wgs84_wkt}')"
            )

            if wgs84_geometry.time_range is not None:
                start = wgs84_geometry.time_range[0].strftime(self.DATE_FORMAT)
                end = wgs84_geometry.time_range[1].strftime(self.DATE_FORMAT)
                filter_string += f" and ContentDate/Start gt {start}"
                filter_string += f" and ContentDate/End lt {end}"

            filter_string = self._build_filter_string(filter_string)
            path = f"/Products?$filter={filter_string}&$top=1000"

            if self.order_by is not None:
                path += f"&$orderby={self.order_by}"
            if self.sort_by is not None:
                path += "&$expand=Attributes"

            products = self._paginate(path)

            if self.sort_by is not None:
                # Define helper function that computes the sort value.
                def get_attribute_value(product: dict[str, Any]) -> Any:
                    attribute_by_name = {
                        attribute["Name"]: attribute["Value"]
                        for attribute in product["Attributes"]
                    }
                    return attribute_by_name[self.sort_by]

                products.sort(
                    key=get_attribute_value,
                    reverse=self.sort_desc,
                )

            candidate_items = [self._product_to_item(product) for product in products]
            cur_groups = match_candidate_items_to_window(
                geometry, candidate_items, query_config
            )
            groups.append(cur_groups)

        return groups

    def _get_access_token(self) -> str:
        """Get the access token to use for downloads.

        If the username/password are set, we need to get the token from API.
        """
        if self.access_token is not None:
            return self.access_token

        response = requests.post(
            self.TOKEN_URL,
            data={
                "grant_type": "password",
                "username": self.username,
                "password": self.password,
                "client_id": "cdse-public",
            },
            timeout=self.timeout,
        )
        return response.json()["access_token"]

    def _zip_member_glob(self, member_names: list[str], pattern: str) -> str:
        """Pick the zip member name that matches the given pattern.

        Args:
            member_names: the list of names in the zip file.
            pattern: the glob pattern to match.

        Returns:
            the member name matching the pattern.

        Raises:
            ValueError: if there is no matching member.
        """
        for name in member_names:
            if pathlib.PurePosixPath(name).match(pattern):
                return name
        raise ValueError(f"no zip member matching {pattern}")

    def _process_product_zip(
        self, tile_store: TileStoreWithLayer, item: CopernicusItem, local_zip_fname: str
    ) -> None:
        """Ingest rasters in the specified product zip file.

        Args:
            tile_store: the tile store to ingest the rasters into.
            item: the item to download and ingest.
            local_zip_fname: the local filename where the product zip file has been
                downloaded.
        """
        with ZipFile(local_zip_fname) as zipf:
            member_names = zipf.namelist()

            # Get each raster that is needed.
            for glob_pattern, band_names in self.glob_to_bands.items():
                if tile_store.is_raster_ready(item.name, band_names):
                    continue

                member_name = self._zip_member_glob(member_names, glob_pattern)

                # Extract it to a temporary directory.
                with tempfile.TemporaryDirectory() as tmp_dir:
                    logger.debug(f"Extracting {member_name} for bands {band_names}")
                    local_raster_fname = zipf.extract(member_name, path=tmp_dir)

                    # Now we can ingest it.
                    logger.debug(f"Ingesting the raster for bands {band_names}")
                    tile_store.write_raster_file(
                        item.name, band_names, UPath(local_raster_fname)
                    )

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[CopernicusItem],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        Args:
            tile_store: the tile store to ingest into
            items: the items to ingest
            geometries: a list of geometries needed for each item
        """
        for item in items:
            # The product zip file is one big download, so we download it if any raster
            # hasn't been ingested yet.
            any_rasters_needed = False
            for band_names in self.glob_to_bands.values():
                if tile_store.is_raster_ready(item.name, band_names):
                    continue
                any_rasters_needed = True
                break
            if not any_rasters_needed:
                continue

            # Download the product zip file to temporary directory.
            with tempfile.TemporaryDirectory() as tmp_dir:
                path = f"/Products({item.product_uuid})/$value"
                logger.debug(
                    f"Downloading product zip file from {self.DOWNLOAD_URL + path}"
                )

                access_token = self._get_access_token()
                headers = {
                    "Authorization": f"Bearer {access_token}",
                }
                response = requests.get(
                    self.DOWNLOAD_URL + path,
                    stream=True,
                    headers=headers,
                    timeout=self.timeout,
                )
                if response.status_code != 200:
                    content = str(response.content)
                    raise ApiError(
                        f"expected status code 200 but got {response.status_code} ({content})"
                    )

                local_zip_fname = os.path.join(tmp_dir, "product.zip")
                with open(local_zip_fname, "wb") as f:
                    for chunk in response.iter_content(chunk_size=self.CHUNK_SIZE):
                        f.write(chunk)

                # Process each raster we need from the zip file.
                self._process_product_zip(tile_store, item, local_zip_fname)


class Sentinel2ProductType(str, Enum):
    """The Sentinel-2 product type."""

    L1C = "S2MSI1C"
    L2A = "S2MSI2A"


class Sentinel2(Copernicus):
    """A data source for Sentinel-2 data from the Copernicus API."""

    BANDS = {
        "B01": ["B01"],
        "B02": ["B02"],
        "B03": ["B03"],
        "B04": ["B04"],
        "B05": ["B05"],
        "B06": ["B06"],
        "B07": ["B07"],
        "B08": ["B08"],
        "B09": ["B09"],
        "B11": ["B11"],
        "B12": ["B12"],
        "B8A": ["B8A"],
        "TCI": ["R", "G", "B"],
        # L2A-only products.
        "AOT": ["AOT"],
        "WVP": ["WVP"],
        "SCL": ["SCL"],
    }

    # Glob pattern for image files within the product zip file.
    GLOB_PATTERNS = {
        Sentinel2ProductType.L1C: {
            "B01": "*/GRANULE/*/IMG_DATA/*_B01.jp2",
            "B02": "*/GRANULE/*/IMG_DATA/*_B02.jp2",
            "B03": "*/GRANULE/*/IMG_DATA/*_B03.jp2",
            "B04": "*/GRANULE/*/IMG_DATA/*_B04.jp2",
            "B05": "*/GRANULE/*/IMG_DATA/*_B05.jp2",
            "B06": "*/GRANULE/*/IMG_DATA/*_B06.jp2",
            "B07": "*/GRANULE/*/IMG_DATA/*_B07.jp2",
            "B08": "*/GRANULE/*/IMG_DATA/*_B08.jp2",
            "B8A": "*/GRANULE/*/IMG_DATA/*_B8A.jp2",
            "B09": "*/GRANULE/*/IMG_DATA/*_B09.jp2",
            "B10": "*/GRANULE/*/IMG_DATA/*_B10.jp2",
            "B11": "*/GRANULE/*/IMG_DATA/*_B11.jp2",
            "B12": "*/GRANULE/*/IMG_DATA/*_B12.jp2",
            "TCI": "*/GRANULE/*/IMG_DATA/*_TCI.jp2",
        },
        Sentinel2ProductType.L2A: {
            # In L2A, products are grouped by resolution.
            # They are downsampled at lower resolutions too, so here we specify to just
            # use the highest resolution one.
            "B01": "*/GRANULE/*/IMG_DATA/R20m/*_B01_20m.jp2",
            "B02": "*/GRANULE/*/IMG_DATA/R10m/*_B02_10m.jp2",
            "B03": "*/GRANULE/*/IMG_DATA/R10m/*_B03_10m.jp2",
            "B04": "*/GRANULE/*/IMG_DATA/R10m/*_B04_10m.jp2",
            "B05": "*/GRANULE/*/IMG_DATA/R20m/*_B05_20m.jp2",
            "B06": "*/GRANULE/*/IMG_DATA/R20m/*_B06_20m.jp2",
            "B07": "*/GRANULE/*/IMG_DATA/R20m/*_B07_20m.jp2",
            "B08": "*/GRANULE/*/IMG_DATA/R10m/*_B08_10m.jp2",
            "B8A": "*/GRANULE/*/IMG_DATA/R20m/*_B8A_20m.jp2",
            "B09": "*/GRANULE/*/IMG_DATA/R60m/*_B09_60m.jp2",
            "B11": "*/GRANULE/*/IMG_DATA/R20m/*_B11_20m.jp2",
            "B12": "*/GRANULE/*/IMG_DATA/R20m/*_B12_20m.jp2",
            "TCI": "*/GRANULE/*/IMG_DATA/R10m/*_TCI_10m.jp2",
            "AOT": "*/GRANULE/*/IMG_DATA/R10m/*_AOT_10m.jp2",
            "WVP": "*/GRANULE/*/IMG_DATA/R10m/*_WVP_10m.jp2",
            "SCL": "*/GRANULE/*/IMG_DATA/R20m/*_SCL_20m.jp2",
        },
    }

    # Pattern of XML file within the product zip file.
    METADATA_PATTERN = "*/MTD_MSIL*.xml"

    def __init__(
        self,
        assets: list[str],
        product_type: Sentinel2ProductType,
        harmonize: bool = False,
        **kwargs: Any,
    ):
        """Create a new Sentinel2.

        Args:
            assets: list of assets corresponding to keys in BANDS, e.g. ["TCI", "B08"].
            product_type: desired product type, L1C or L2A.
            harmonize: harmonize pixel values across different processing baselines,
                see https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
            kwargs: additional arguments to pass to Copernicus.
        """
        # Create glob to bands map.
        glob_to_bands = {}
        for asset_key in assets:
            band_names = self.BANDS[asset_key]
            glob_pattern = self.GLOB_PATTERNS[product_type][asset_key]
            glob_to_bands[glob_pattern] = band_names

        # Create query filter based on the product type.
        query_filter = f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq '{quote(product_type.value)}')"

        super().__init__(
            glob_to_bands=glob_to_bands,
            query_filter=query_filter,
            **kwargs,
        )
        self.harmonize = harmonize

    @staticmethod
    def from_config(config: RasterLayerConfig, ds_path: UPath) -> "Sentinel2":
        """Creates a new Sentinel2 instance from a configuration dictionary."""
        if config.data_source is None:
            raise ValueError("config.data_source is required")
        d = config.data_source.config_dict

        # Determine needed assets based on the configured band sets.
        needed_assets: set[str] = set()
        for asset_key, asset_bands in Sentinel2.BANDS.items():
            for band_set in config.band_sets:
                if not set(band_set.bands).intersection(set(asset_bands)):
                    continue
                needed_assets.add(asset_key)

        kwargs: dict[str, Any] = dict(
            assets=list(needed_assets),
        )

        if "product_type" in d:
            kwargs["product_type"] = Sentinel2ProductType(d["product_type"])

        simple_optionals = [
            "harmonize",
            "access_token",
            "order_by",
            "sort_by",
            "sort_desc",
        ]
        for k in simple_optionals:
            if k in d:
                kwargs[k] = d[k]

        return Sentinel2(**kwargs)

    # Override to support harmonization step.
    def _process_product_zip(
        self, tile_store: TileStoreWithLayer, item: CopernicusItem, local_zip_fname: str
    ) -> None:
        """Ingest rasters in the specified product zip file.

        Args:
            tile_store: the tile store to ingest the rasters into.
            item: the item to download and ingest.
            local_zip_fname: the local filename where the product zip file has been
                downloaded.
        """
        with ZipFile(local_zip_fname) as zipf:
            member_names = zipf.namelist()

            harmonize_callback = None
            if self.harmonize:
                # Need to check the product XML to see what the callback should be.
                # It's in the zip file.
                member_name = self._zip_member_glob(member_names, self.METADATA_PATTERN)
                with zipf.open(member_name) as f:
                    xml_data = ET.parse(f)
                harmonize_callback = get_harmonize_callback(xml_data)

            # Get each raster that is needed.
            for glob_pattern, band_names in self.glob_to_bands.items():
                if tile_store.is_raster_ready(item.name, band_names):
                    continue

                member_name = self._zip_member_glob(member_names, glob_pattern)

                # Extract it to a temporary directory.
                with tempfile.TemporaryDirectory() as tmp_dir:
                    logger.debug(f"Extracting {member_name} for bands {band_names}")
                    local_raster_fname = zipf.extract(member_name, path=tmp_dir)

                    logger.debug(f"Ingesting the raster for bands {band_names}")

                    if harmonize_callback is None or band_names == ["R", "G", "B"]:
                        # No callback -- we can just ingest the file directly.
                        # Or it is TCI product which is not impacted by the harmonization issue.
                        tile_store.write_raster_file(
                            item.name, band_names, UPath(local_raster_fname)
                        )

                    else:
                        # In this case we need to read the array, convert the pixel
                        # values, and pass modified array directly to the TileStore.
                        with rasterio.open(local_raster_fname) as src:
                            array = src.read()
                            projection, bounds = get_raster_projection_and_bounds(src)
                        array = harmonize_callback(array)
                        tile_store.write_raster(
                            item.name, band_names, projection, bounds, array
                        )


class Sentinel1ProductType(str, Enum):
    """The Sentinel-1 product type."""

    IW_GRDH = "IW_GRDH_1S"


class Sentinel1Polarisation(str, Enum):
    """The Sentinel-1 polarisation."""

    VV_VH = "VV&VH"


class Sentinel1OrbitDirection(str, Enum):
    """The Sentinel-1 orbit direction."""

    ASCENDING = "ASCENDING"
    DESCENDING = "DESCENDING"


class Sentinel1(Copernicus):
    """A data source for Sentinel-1 data from the Copernicus API."""

    GLOB_TO_BANDS = {
        Sentinel1Polarisation.VV_VH: {
            "*/measurement/*-vh-*.tiff": ["vh"],
            "*/measurement/*-vv-*.tiff": ["vv"],
        }
    }

    # Pattern of XML file within the product zip file.
    METADATA_PATTERN = "*/MTD_MSIL*.xml"

    def __init__(
        self,
        product_type: Sentinel1ProductType,
        polarisation: Sentinel1Polarisation,
        orbit_direction: Sentinel1OrbitDirection | None = None,
        **kwargs: Any,
    ):
        """Create a new Sentinel1.

        Args:
            product_type: desired product type.
            polarisation: desired polarisation(s).
            orbit_direction: optional orbit direction to filter by.
            kwargs: additional arguments to pass to Copernicus.
        """
        # Create query filter based on the product type.
        query_filter = (
            f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq '{quote(product_type.value)}')"
            + f" and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'polarisationChannels' and att/OData.CSC.StringAttribute/Value eq '{quote(polarisation.value)}')"
        )
        if orbit_direction:
            query_filter += f" and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'orbitDirection' and att/OData.CSC.StringAttribute/Value eq '{quote(orbit_direction.value)}')"

        super().__init__(
            glob_to_bands=self.GLOB_TO_BANDS[polarisation],
            query_filter=query_filter,
            **kwargs,
        )

    @staticmethod
    def from_config(config: RasterLayerConfig, ds_path: UPath) -> "Sentinel1":
        """Creates a new Sentinel1 instance from a configuration dictionary."""
        if config.data_source is None:
            raise ValueError("config.data_source is required")
        d = config.data_source.config_dict

        kwargs: dict[str, Any] = dict(
            product_type=Sentinel1ProductType[d["product_type"]],
            polarisation=Sentinel1Polarisation[d["polarisation"]],
        )

        if "orbit_direction" in d:
            kwargs["orbit_direction"] = Sentinel1OrbitDirection[d["orbit_direction"]]

        simple_optionals = [
            "access_token",
            "order_by",
            "sort_by",
            "sort_desc",
        ]
        for k in simple_optionals:
            if k in d:
                kwargs[k] = d[k]

        return Sentinel1(**kwargs)
