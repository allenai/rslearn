"""API client for the USGS M2M API."""

import json
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Any

import requests


class APIException(Exception):
    """Exception raised for M2M API errors."""

    pass


class M2MAPIClient:
    """An API client for interacting with the USGS M2M API."""

    api_url = "https://m2m.cr.usgs.gov/api/api/json/stable/"
    pagination_size = 1000

    def __init__(
        self,
        username: str | None = None,
        token: str | None = None,
        timeout: timedelta = timedelta(seconds=120),
        session: requests.Session | None = None,
    ) -> None:
        """Initialize a new M2MAPIClient.

        Args:
            username: the EROS username. If None, uses M2M_USERNAME environment variable.
            token: the application token. If None, uses M2M_TOKEN environment variable.
            timeout: timeout for requests.
            session: optional requests session to use. If None, a default session is
                created.
        """
        if username is None:
            username = os.environ.get("M2M_USERNAME")
            if username is None:
                raise ValueError(
                    "username must be provided or M2M_USERNAME environment variable must be set"
                )
        if token is None:
            token = os.environ.get("M2M_TOKEN")
            if token is None:
                raise ValueError(
                    "token must be provided or M2M_TOKEN environment variable must be set"
                )

        self.username = username
        self.timeout = timeout
        self.session = session if session is not None else requests.Session()

        json_data = json.dumps({"username": username, "token": token})
        response = self.session.post(
            self.api_url + "login-token",
            data=json_data,
            timeout=self.timeout.total_seconds(),
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
        response = self.session.post(
            self.api_url + endpoint,
            headers={"X-Auth-Token": self.auth_token},
            data=json.dumps(data),
            timeout=self.timeout.total_seconds(),
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

    def __exit__(
        self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any
    ) -> None:
        """Exit function to provide with semantics.

        Args:
            exc_type: exception type if an exception was raised
            exc_val: exception value if an exception was raised
            exc_tb: exception traceback if an exception was raised

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

        # Check if URL is immediately available in the response
        if response.get("availableDownloads"):
            return response["availableDownloads"][0]["url"]

        # Otherwise poll download-retrieve until the URL is ready
        while True:
            time.sleep(10)
            response_dict = self.request("download-retrieve", {"label": label})
            if response_dict is None:
                raise APIException("No response from API")
            response = response_dict["data"]
            if len(response["available"]) > 0:
                return response["available"][0]["url"]
            if len(response["requested"]) == 0:
                print(response)
                raise APIException("Did not get download URL")
            if response["requested"][0].get("url"):
                return response["requested"][0]["url"]
