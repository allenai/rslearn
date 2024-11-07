"""Data source for Landsat data from USGS M2M API.

# TODO: Handle the requests in a helper function for none checking
"""

import io
import json
import shutil
import time
import uuid
from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from typing import Any, BinaryIO

import pytimeparse
import rasterio
import requests
import shapely
from upath import UPath

from rslearn.config import QueryConfig, RasterLayerConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import DataSource, Item
from rslearn.data_sources.utils import match_candidate_items_to_window
from rslearn.tile_stores import PrefixedTileStore, TileStore
from rslearn.utils import STGeometry

from .raster_source import get_needed_projections, ingest_raster


class APIException(Exception):
    """Exception raised for M2M API errors."""

    pass


class M2MAPIClient:
    """An API client for interacting with the USGS M2M API."""

    api_url = "https://m2m.cr.usgs.gov/api/api/json/stable/"
    pagination_size = 1000
    TIMEOUT = 1000000  # Set very high to start

    def __init__(self, username: str, password: str) -> None:
        """Initialize a new M2MAPIClient.

        Args:
            username: the EROS username
            password: the EROS password
        """
        self.username = username
        self.password = password
        json_data = json.dumps({"username": self.username, "password": self.password})
        response = requests.post(
            self.api_url + "login", data=json_data, timeout=self.TIMEOUT
        )
        response.raise_for_status()
        self.auth_token = response.json()["data"]

    def request(
        self, endpoint: str, data: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Make a request to the API.

        Args:
            endpoint: the endpoint to call
            data: POST data to pass

        Returns:
            JSON response data if any
        """
        response = requests.post(
            self.api_url + endpoint,
            headers={"X-Auth-Token": self.auth_token},
            data=json.dumps(data),
            timeout=self.TIMEOUT,
        )
        response.raise_for_status()
        if response.text:
            response_dict = response.json()

            if response_dict["errorMessage"]:
                raise APIException(response_dict["errorMessage"])
            return response_dict
        return None

    def close(self) -> None:
        """Logout from the API."""
        self.request("logout")

    def __enter__(self) -> "M2MAPIClient":
        """Enter function to provide with semantics."""
        return self

    def __exit__(self) -> None:
        """Exit function to provide with semantics.

        Logs out the API.
        """
        self.close()

    def get_filters(self, dataset_name: str) -> list[dict[str, Any]]:
        """Returns filters available for the given dataset.

        Args:
            dataset_name: the dataset name e.g. landsat_ot_c2_l1

        Returns:
            list of filter objects
        """
        response_dict = self.request("dataset-filters", {"datasetName": dataset_name})
        if response_dict is None:
            raise APIException("No response from API")
        return response_dict["data"]

    def scene_search(
        self,
        dataset_name: str,
        acquisition_time_range: tuple[datetime, datetime] | None = None,
        cloud_cover_range: tuple[int, int] | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for scenes matching the arguments.

        Args:
            dataset_name: the dataset name e.g. landsat_ot_c2_l1
            acquisition_time_range: optional filter on the acquisition time
            cloud_cover_range: optional filter on the cloud cover
            bbox: optional spatial filter
            metadata_filter: optional metadata filter dict
        """
        base_data: dict[str, Any] = {"datasetName": dataset_name, "sceneFilter": {}}
        if acquisition_time_range:
            base_data["sceneFilter"]["acquisitionFilter"] = {
                "start": acquisition_time_range[0].isoformat(),
                "end": acquisition_time_range[1].isoformat(),
            }
        if cloud_cover_range:
            base_data["sceneFilter"]["cloudCoverFilter"] = {
                "min": cloud_cover_range[0],
                "max": cloud_cover_range[1],
                "includeUnknown": False,
            }
        if bbox:
            base_data["sceneFilter"]["spatialFilter"] = {
                "filterType": "mbr",
                "lowerLeft": {"longitude": bbox[0], "latitude": bbox[1]},
                "upperRight": {"longitude": bbox[2], "latitude": bbox[3]},
            }
        if metadata_filter:
            base_data["sceneFilter"]["metadataFilter"] = metadata_filter

        starting_number = 1
        results = []
        while True:
            cur_data = base_data.copy()
            cur_data["startingNumber"] = starting_number
            cur_data["maxResults"] = self.pagination_size
            response_dict = self.request("scene-search", cur_data)
            if response_dict is None:
                raise APIException("No response from API")
            data = response_dict["data"]
            results.extend(data["results"])
            if data["recordsReturned"] < self.pagination_size:
                break
            starting_number += self.pagination_size

        return results

    def get_scene_metadata(self, dataset_name: str, entity_id: str) -> dict[str, Any]:
        """Get detailed metadata for a scene.

        Args:
            dataset_name: the dataset name in which to search
            entity_id: the entity ID of the scene

        Returns:
            full scene metadata
        """
        response_dict = self.request(
            "scene-metadata",
            {
                "datasetName": dataset_name,
                "entityId": entity_id,
                "metadataType": "full",
            },
        )
        if response_dict is None:
            raise APIException("No response from API")
        return response_dict["data"]

    def get_downloadable_products(
        self, dataset_name: str, entity_id: str
    ) -> list[dict[str, Any]]:
        """Get the downloadable products for a given scene.

        Args:
            dataset_name: the dataset name
            entity_id: the entity ID of the scene

        Returns:
            list of downloadable products
        """
        data = {"datasetName": dataset_name, "entityIds": [entity_id]}
        response_dict = self.request("download-options", data)
        if response_dict is None:
            raise APIException("No response from API")
        return response_dict["data"]

    def get_download_url(self, entity_id: str, product_id: str) -> str:
        """Get the download URL for a given product.

        Args:
            entity_id: the entity ID of the product
            product_id: the product ID of the product

        Returns:
            the download URL
        """
        label = str(uuid.uuid4())
        data = {
            "downloads": [
                {"label": label, "entityId": entity_id, "productId": product_id}
            ]
        }
        response_dict = self.request("download-request", data)
        if response_dict is None:
            raise APIException("No response from API")
        response = response_dict["data"]
        while True:
            response_dict = self.request("download-retrieve", {"label": label})
            if response_dict is None:
                raise APIException("No response from API")
            response = response_dict["data"]
            if len(response["available"]) > 0:
                return response["available"][0]["url"]
            if len(response["requested"]) == 0:
                raise Exception("Did not get download URL")
            if response["requested"][0].get("url"):
                return response["requested"][0]["url"]
            time.sleep(10)


class LandsatOliTirsItem(Item):
    """An item in the LandsatOliTirs data source."""

    dataset_name = "landsat_ot_c2_l1"

    def __init__(
        self, name: str, geometry: STGeometry, entity_id: str, cloud_cover: float
    ):
        """Creates a new LandsatOliTirsItem.

        Args:
            name: unique name of the item
            geometry: the spatial and temporal extent of the item
            entity_id: the entity ID of this item
            cloud_cover: the cloud cover percentage
        """
        super().__init__(name, geometry)
        self.entity_id = entity_id
        self.cloud_cover = cloud_cover

    def serialize(self) -> dict[str, Any]:
        """Serializes the item to a JSON-encodable dictionary."""
        d = super().serialize()
        d["entity_id"] = self.entity_id
        d["cloud_cover"] = self.cloud_cover
        return d

    @staticmethod
    def deserialize(d: dict[str, Any]) -> Item:
        """Deserializes an item from a JSON-decoded dictionary."""
        item = super(LandsatOliTirsItem, LandsatOliTirsItem).deserialize(d)
        return LandsatOliTirsItem(
            name=item.name,
            geometry=item.geometry,
            entity_id=d["entity_id"],
            cloud_cover=d["cloud_cover"],
        )


class LandsatOliTirs(DataSource):
    """A data source for Landsat data from the USGS M2M API."""

    bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"]

    dataset_name = "landsat_ot_c2_l1"

    def __init__(
        self,
        config: RasterLayerConfig,
        username: str,
        password: str,
        max_time_delta: timedelta = timedelta(days=30),
        sort_by: str | None = None,
    ):
        """Initialize a new LandsatOliTirs instance.

        Args:
            config: the LayerConfig of the layer containing this data source
            username: EROS username
            password: EROS password
            max_time_delta: maximum time before a query start time or after a
                query end time to look for products. This is required due to the large
                number of available products, and defaults to 30 days.
            sort_by: can be "cloud_cover", default arbitrary order; only has effect for
                SpaceMode.WITHIN.
        """
        self.config = config
        self.max_time_delta = max_time_delta
        self.sort_by = sort_by

        self.client = M2MAPIClient(username, password)

    @staticmethod
    def from_config(config: RasterLayerConfig, ds_path: UPath) -> "LandsatOliTirs":
        """Creates a new LandsatOliTirs instance from a configuration dictionary."""
        if config.data_source is None:
            raise ValueError("data_source is required")
        d = config.data_source.config_dict
        if "max_time_delta" in d:
            max_time_delta = timedelta(seconds=pytimeparse.parse(d["max_time_delta"]))
        else:
            max_time_delta = timedelta(days=30)
        return LandsatOliTirs(
            config=config,
            username=d["username"],
            password=d["password"],
            max_time_delta=max_time_delta,
            sort_by=d.get("sort_by"),
        )

    def _scene_metadata_to_item(self, result: dict[str, Any]) -> LandsatOliTirsItem:
        """Convert scene metadata from the API to a LandsatOliTirsItem."""
        metadata_dict = {}
        for el in result["metadata"]:
            metadata_dict[el["fieldName"]] = el["value"]
        shp = shapely.geometry.shape(result["spatialCoverage"])

        # Parse time either "2022-01-29 05:46:37.339474" or "2022-01-29 05:46:37".
        if "." in metadata_dict["Start Time"]:
            ts = datetime.strptime(metadata_dict["Start Time"], "%Y-%m-%d %H:%M:%S.%f")
        else:
            ts = datetime.strptime(metadata_dict["Start Time"], "%Y-%m-%d %H:%M:%S")
        ts = ts.replace(tzinfo=timezone.utc)

        return LandsatOliTirsItem(
            name=result["displayId"],
            geometry=STGeometry(WGS84_PROJECTION, shp, (ts, ts)),
            entity_id=result["entityId"],
            cloud_cover=result["cloudCover"],
        )

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[LandsatOliTirsItem]]]:
        """Get a list of items in the data source intersecting the given geometries.

        Args:
            geometries: the spatiotemporal geometries
            query_config: the query configuration

        Returns:
            List of groups of items that should be retrieved for each geometry.
        """
        groups = []
        for geometry in geometries:
            wgs84_geometry = geometry.to_projection(WGS84_PROJECTION)
            bounds = wgs84_geometry.shp.bounds
            kwargs = {"dataset_name": self.dataset_name, "bbox": bounds}
            if geometry.time_range is not None:
                kwargs["acquisition_time_range"] = geometry.time_range
            results = self.client.scene_search(**kwargs)
            items = []
            for result in results:
                scene_metadata = self.client.get_scene_metadata(
                    self.dataset_name, result["entityId"]
                )
                item = self._scene_metadata_to_item(scene_metadata)
                items.append(item)

            if self.sort_by == "cloud_cover":
                items.sort(
                    key=lambda item: item.cloud_cover if item.cloud_cover >= 0 else 100
                )
            elif self.sort_by is not None:
                raise ValueError(f"invalid sort_by setting ({self.sort_by})")

            cur_groups = match_candidate_items_to_window(geometry, items, query_config)
            groups.append(cur_groups)
        return groups

    def get_item_by_name(self, name: str) -> Item:
        """Gets an item by name."""
        # Get the filter to use.
        filter_options = self.client.get_filters(self.dataset_name)
        product_identifier_filter = None
        for filter_option in filter_options:
            if filter_option["fieldLabel"] != "Landsat Product Identifier L1":
                continue
            product_identifier_filter = filter_option["id"]
        if not product_identifier_filter:
            raise APIException("did not find filter for product identifier")

        # Use the filter to get the scene.
        results = self.client.scene_search(
            self.dataset_name,
            metadata_filter={
                "filterType": "value",
                "filterId": product_identifier_filter,
                "operand": "=",
                "value": name,
            },
        )
        if len(results) != 1:
            raise APIException(f"expected one result but got {len(results)}")

        scene_metadata = self.client.get_scene_metadata(
            self.dataset_name, results[0]["entityId"]
        )
        return self._scene_metadata_to_item(scene_metadata)

    def deserialize_item(self, serialized_item: Any) -> Item:
        """Deserializes an item from JSON-decoded data."""
        assert isinstance(serialized_item, dict)
        return LandsatOliTirsItem.deserialize(serialized_item)

    def _get_download_urls(self, item: Item) -> dict[str, tuple[str, str]]:
        """Gets the download URLs for each band.

        Args:
            item: the item to download

        Returns:
            dictionary mapping from band name to (fname, download URL)
        """
        assert isinstance(item, LandsatOliTirsItem)
        options = self.client.get_downloadable_products(
            self.dataset_name, item.entity_id
        )
        wanted_bands = {band for band in self.bands}
        download_urls = {}
        for option in options:
            if not option.get("secondaryDownloads"):
                continue
            for secondary_download in option["secondaryDownloads"]:
                display_id = secondary_download["displayId"]
                if not display_id.endswith(".TIF"):
                    continue
                band = display_id.split(".TIF")[0].split("_")[-1]
                if band not in wanted_bands:
                    continue
                if band in download_urls:
                    continue
                download_url = self.client.get_download_url(
                    secondary_download["entityId"], secondary_download["id"]
                )
                download_urls[band] = (display_id, download_url)
        return download_urls

    def retrieve_item(self, item: Item) -> Generator[tuple[str, BinaryIO], None, None]:
        """Retrieves the rasters corresponding to an item as file streams."""
        download_urls = self._get_download_urls(item)
        for _, (display_id, download_url) in download_urls.items():
            buf = io.BytesIO()
            with requests.get(download_url, stream=True, timeout=self.TIMEOUT) as r:
                r.raise_for_status()
                shutil.copyfileobj(r.raw, buf)
            buf.seek(0)
            yield (display_id, buf)

    def ingest(
        self,
        tile_store: TileStore,
        items: list[LandsatOliTirsItem],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        Args:
            tile_store: the tile store to ingest into
            items: the items to ingest
            geometries: a list of geometries needed for each item
        """
        for item, cur_geometries in zip(items, geometries):
            download_urls = self._get_download_urls(item)
            for band in self.bands:
                band_names = [band]
                cur_tile_store = PrefixedTileStore(
                    tile_store, (item.name, "_".join(band_names))
                )
                needed_projections = get_needed_projections(
                    cur_tile_store, band_names, self.config.band_sets, cur_geometries
                )
                if not needed_projections:
                    continue

                buf = io.BytesIO()
                with requests.get(
                    download_urls[band][1], stream=True, timeout=self.TIMEOUT
                ) as r:
                    r.raise_for_status()
                    shutil.copyfileobj(r.raw, buf)
                buf.seek(0)
                with rasterio.open(buf) as raster:
                    for projection in needed_projections:
                        ingest_raster(
                            tile_store=cur_tile_store,
                            raster=raster,
                            projection=projection,
                            time_range=item.geometry.time_range,
                            layer_config=self.config,
                        )
