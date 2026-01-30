"""Data source for the olmoearth_datasets API.

This is currently an Ai2-internal service that caches scene metadata from various data
providers (Planetary Computer, miscellaneous AWS S3 buckets, etc.). Here, during
ingestion or materialization, we try to download image data from each provider, falling
back to the next provider when we encounter errors.
"""

import json
import os
import random
import tempfile
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import numpy.typing as npt
import planetary_computer
import rasterio
import requests
import shapely
from pydantic import BaseModel, Field
from rasterio.enums import Resampling

from rslearn.config import QueryConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.data_source import DataSourceContext, Item
from rslearn.data_sources.tile_store_data_source import TileStoreDataSource
from rslearn.data_sources.utils import match_candidate_items_to_window
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry
from rslearn.utils.raster_format import get_raster_projection_and_bounds

logger = get_logger(__name__)

# Default API endpoint.
DEFAULT_API_URL = "https://datasets-staging.olmoearth.allenai.org"


class OlmoEarthAsset(BaseModel):
    """An asset from a data provider in the OlmoEarth Datasets API."""

    name: str = Field(description="The asset name (e.g., 'B04', 'visual').")
    url: str = Field(description="The URL to download the asset.")
    bands: list[str] = Field(description="List of band names in this asset.")


class OlmoEarthDataProvider(BaseModel):
    """A data provider entry from the OlmoEarth Datasets API response."""

    provider_name: str = Field(
        description="The provider name (e.g., 'planetary_computer')."
    )
    provider_id: str = Field(description="The provider-specific item ID.")
    collection: str = Field(description="The collection name.")
    properties: dict[str, Any] = Field(
        description="Provider-specific properties (processing_baseline, etc.)."
    )
    assets: list[OlmoEarthAsset] = Field(
        description="List of assets available from this provider."
    )


class OlmoEarthItem(Item):
    """An item from the OlmoEarth Datasets API."""

    def __init__(
        self,
        name: str,
        geometry: STGeometry,
        properties: dict[str, Any],
        data_providers: list[OlmoEarthDataProvider],
    ):
        """Create a new OlmoEarthItem.

        Args:
            name: unique name of the item (the item ID from the API).
            geometry: the spatial and temporal extent of the item.
            properties: item properties (cloud_cover, satellite, etc.).
            data_providers: list of data providers with their assets.
        """
        super().__init__(name, geometry)
        self.properties = properties
        self.data_providers = data_providers

    def serialize(self) -> dict[str, Any]:
        """Serializes the item to a JSON-encodable dictionary."""
        d = super().serialize()
        d["properties"] = self.properties
        d["data_providers"] = [dp.model_dump() for dp in self.data_providers]
        return d

    @staticmethod
    def deserialize(d: dict[str, Any]) -> "OlmoEarthItem":
        """Deserializes an item from a JSON-decoded dictionary."""
        item = super(OlmoEarthItem, OlmoEarthItem).deserialize(d)
        return OlmoEarthItem(
            name=item.name,
            geometry=item.geometry,
            properties=d["properties"],
            data_providers=[
                OlmoEarthDataProvider.model_validate(dp) for dp in d["data_providers"]
            ],
        )


class OlmoEarthDatasets(TileStoreDataSource[OlmoEarthItem]):
    """Data source for satellite imagery metadata from the OlmoEarth Datasets API.

    This is a generic base class that queries the OlmoEarth Datasets API for scene
    metadata and downloads assets from various data providers. Subclasses should
    specify the collection and asset/band mappings.

    When downloading, if an item has multiple data providers, the providers are
    shuffled and tried in sequence until one succeeds. This provides automatic
    fallback if one provider is unavailable.

    This class also supports direct materialization from COG URLs via TileStoreDataSource.
    """

    def __init__(
        self,
        collection: str,
        asset_bands: dict[str, list[str]],
        api_url: str = DEFAULT_API_URL,
        query: dict[str, Any] | None = None,
        sort_by: str | None = None,
        sort_ascending: bool = True,
        timeout: timedelta = timedelta(seconds=30),
        context: DataSourceContext = DataSourceContext(),
    ):
        """Initialize a new OlmoEarthDatasets instance.

        Args:
            collection: the collection name to query (e.g., "sentinel-2-l2a").
            asset_bands: mapping from asset name to list of band names in that asset.
            api_url: the base URL of the OlmoEarth Datasets API.
            query: optional query filters to include in API requests. Supports filters
                like cloud_cover, satellite, mgrs_tile, orbit_direction, etc. Example:
                {"cloud_cover": {"lt": 10}, "satellite": {"eq": "S2A"}}.
            sort_by: sort by this property (e.g., "cloud_cover", "collected_at").
            sort_ascending: whether to sort ascending (or descending).
            timeout: timeout for API and download requests.
            context: the data source context.
        """
        TileStoreDataSource.__init__(self, asset_bands=asset_bands)

        self.collection = collection
        self.api_url = api_url.rstrip("/")
        self.query = query
        self.sort_by = sort_by
        self.sort_ascending = sort_ascending
        self.timeout = timeout
        self.context = context

    def _search_request(
        self,
        collection: str | None = None,
        item_id: str | None = None,
        intersects_geometry: dict[str, Any] | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        query: dict[str, Any] | None = None,
        sort_by: str | None = None,
        sort_ascending: bool = True,
    ) -> list[dict[str, Any]]:
        """Make a search request to the OlmoEarth Datasets API.

        Automatically paginates through all results.

        Args:
            collection: filter by collection name.
            item_id: filter by exact item ID.
            intersects_geometry: GeoJSON geometry to filter by spatial intersection.
            time_range: tuple of (start, end) datetimes to filter by collected_at.
            query: additional query filters (cloud_cover, satellite, etc.).
            sort_by: property to sort by.
            sort_ascending: sort direction.

        Returns:
            list of item records from the API.

        Raises:
            requests.HTTPError: if the request fails.
        """
        payload: dict[str, Any] = {}
        if collection is not None:
            payload["collection"] = {"eq": collection}
        if item_id is not None:
            payload["id"] = {"eq": item_id}
        if intersects_geometry is not None:
            payload["intersects_geometry"] = intersects_geometry
        if time_range is not None:
            payload["collected_at"] = {
                "gte": time_range[0].isoformat(),
                "lt": time_range[1].isoformat(),
            }
        if sort_by is not None:
            payload["sort_by"] = sort_by
            payload["sort_direction"] = "asc" if sort_ascending else "desc"

        # Add arbitrary filters specified by the user.
        if query is not None:
            payload.update(query)

        url = f"{self.api_url}/api/v1/items/search"

        # Paginate through all results.
        page_size = 1000
        all_records: list[dict[str, Any]] = []
        offset = 0

        while True:
            payload["limit"] = page_size
            payload["offset"] = offset

            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout.total_seconds(),
            )
            response.raise_for_status()
            data = response.json()

            records = data.get("records", [])
            all_records.extend(records)

            # Stop if we got fewer than requested (no more results).
            if len(records) < page_size:
                break

            offset += len(records)

            logger.debug(
                "OlmoEarthDatasets paginating: fetched %d records so far",
                len(all_records),
            )

        return all_records

    def _api_record_to_item(self, record: dict[str, Any]) -> OlmoEarthItem:
        """Convert an API record to an OlmoEarthItem.

        Args:
            record: a record from the API response.

        Returns:
            an OlmoEarthItem instance.
        """
        props = record["properties"]

        # Parse geometry.
        shp = shapely.geometry.shape(props["geometry"])

        # Parse time range from collected_at.
        # The API returns a single collected_at timestamp, so we use it as both
        # start and end of the time range.
        collected_at = datetime.fromisoformat(props["collected_at"])
        time_range = (collected_at, collected_at)

        geometry = STGeometry(WGS84_PROJECTION, shp, time_range)

        # Parse data providers.
        data_providers: list[OlmoEarthDataProvider] = []
        for provider_name, provider_data in record.get("data_providers", {}).items():
            assets = [
                OlmoEarthAsset(
                    name=asset["name"],
                    url=asset["url"],
                    bands=asset["bands"],
                )
                for asset in provider_data.get("assets", [])
            ]
            data_providers.append(
                OlmoEarthDataProvider(
                    provider_name=provider_name,
                    provider_id=provider_data.get("id", ""),
                    collection=provider_data.get("collection", ""),
                    properties=provider_data.get("properties", {}),
                    assets=assets,
                )
            )

        # Extract relevant properties.
        item_properties = {
            k: v
            for k, v in props.items()
            if k not in ("geometry",)  # Don't duplicate geometry in properties.
        }

        return OlmoEarthItem(
            name=record["id"],
            geometry=geometry,
            properties=item_properties,
            data_providers=data_providers,
        )

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[OlmoEarthItem]]]:
        """Get a list of items in the data source intersecting the given geometries.

        Args:
            geometries: the spatiotemporal geometries.
            query_config: the query configuration.

        Returns:
            List of groups of items that should be retrieved for each geometry.
        """
        groups = []
        for geometry in geometries:
            wgs84_geometry = geometry.to_projection(WGS84_PROJECTION)

            records = self._search_request(
                collection=self.collection,
                intersects_geometry=json.loads(shapely.to_geojson(wgs84_geometry.shp)),
                time_range=wgs84_geometry.time_range,
                query=self.query,
                sort_by=self.sort_by,
                sort_ascending=self.sort_ascending,
            )
            logger.debug("OlmoEarthDatasets API returned %d items", len(records))

            # Convert records to items.
            candidate_items = [self._api_record_to_item(record) for record in records]

            # Use the standard matching logic.
            cur_groups = match_candidate_items_to_window(
                geometry, candidate_items, query_config
            )
            groups.append(cur_groups)

        return groups

    def get_item_by_name(self, name: str) -> OlmoEarthItem:
        """Gets an item by name.

        Args:
            name: the item ID.

        Returns:
            the OlmoEarthItem.

        Raises:
            ValueError: if the item is not found or multiple items match.
        """
        records = self._search_request(item_id=name)
        if len(records) != 1:
            raise ValueError(
                f"Expected 1 item for {name}, got {len(records)} from OlmoEarth API"
            )
        return self._api_record_to_item(records[0])

    def deserialize_item(self, serialized_item: Any) -> OlmoEarthItem:
        """Deserializes an item from JSON-decoded data."""
        assert isinstance(serialized_item, dict)
        return OlmoEarthItem.deserialize(serialized_item)

    def _sign_url(self, provider_name: str, url: str) -> str:
        """Sign a URL if needed based on the provider.

        Args:
            provider_name: the data provider name.
            url: the asset URL.

        Returns:
            the signed URL (or original if no signing needed).
        """
        if provider_name == "planetary_computer":
            return planetary_computer.sign(url)
        return url

    def _download_asset(
        self,
        provider: OlmoEarthDataProvider,
        asset: OlmoEarthAsset,
        dest_path: str,
    ) -> None:
        """Download an asset from a provider.

        Args:
            provider: the data provider.
            asset: the asset to download.
            dest_path: the local file path to save to.

        Raises:
            requests.HTTPError: if the download fails.
        """
        url = self._sign_url(provider.provider_name, asset.url)
        logger.debug(
            "Downloading asset %s from provider %s: %s",
            asset.name,
            provider.provider_name,
            url,
        )
        with requests.get(url, stream=True, timeout=self.timeout.total_seconds()) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    def _get_providers_with_asset(
        self, item: OlmoEarthItem, asset_name: str
    ) -> list[tuple[OlmoEarthDataProvider, OlmoEarthAsset]]:
        """Get all providers that have the specified asset.

        Args:
            item: the item.
            asset_name: the asset name to look for.

        Returns:
            list of (provider, asset) tuples that have the asset.
        """
        results = []
        for provider in item.data_providers:
            for asset in provider.assets:
                if asset.name == asset_name:
                    results.append((provider, asset))
                    break
        return results

    def _ingest_raster(
        self,
        tile_store: TileStoreWithLayer,
        item_name: str,
        provider: OlmoEarthDataProvider,
        asset: OlmoEarthAsset,
    ) -> None:
        """Download and ingest a single raster asset.

        Args:
            tile_store: the tile store to ingest into.
            item_name: the name of the item.
            provider: the data provider to download from.
            asset: the asset to download.
        """
        from upath import UPath

        with tempfile.TemporaryDirectory() as tmp_dir:
            local_fname = os.path.join(tmp_dir, f"{asset.name}.tif")
            self._download_asset(provider, asset, local_fname)

            # Apply any post-processing callback.
            callback = self.get_raster_callback(item_name, asset.bands)
            if callback is not None:
                with rasterio.open(local_fname) as src:
                    array = src.read()
                    projection, raster_bounds = get_raster_projection_and_bounds(src)
                array = callback(array)
                tile_store.write_raster(
                    item_name,
                    asset.bands,
                    projection,
                    raster_bounds,
                    array,
                )
            else:
                tile_store.write_raster_file(item_name, asset.bands, UPath(local_fname))

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[OlmoEarthItem],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        For each item and asset, tries providers in shuffled order until one succeeds.
        Does not retry on failure (retries are handled outside the data source).

        Args:
            tile_store: the tile store to ingest into.
            items: the items to ingest.
            geometries: a list of geometries needed for each item.
        """
        for item in items:
            for asset_key, band_names in self.asset_bands.items():
                if tile_store.is_raster_ready(item.name, band_names):
                    continue

                # Get all providers that have this asset.
                providers_with_asset = self._get_providers_with_asset(item, asset_key)
                if not providers_with_asset:
                    logger.warning(
                        "Item %s has no providers with asset %s",
                        item.name,
                        asset_key,
                    )
                    continue

                # Shuffle providers to distribute load and provide automatic fallback.
                random.shuffle(providers_with_asset)

                # Try each provider in sequence.
                for idx, (provider, asset) in enumerate(providers_with_asset):
                    is_last = idx == len(providers_with_asset) - 1
                    try:
                        self._ingest_raster(tile_store, item.name, provider, asset)
                        logger.debug(
                            "Successfully ingested item %s asset %s from %s",
                            item.name,
                            asset.name,
                            provider.provider_name,
                        )
                        break
                    except Exception as e:
                        if is_last:
                            # If we have gone through all the candidate providers, then raise
                            # the error. rslearn logic outside the data source will handle
                            # retries.
                            raise
                        logger.warning(
                            "Failed to download item %s asset %s from %s: %s",
                            item.name,
                            asset.name,
                            provider.provider_name,
                            e,
                        )

    # --- Hooks for subclasses ---

    def get_raster_callback(
        self, item_name: str, bands: list[str]
    ) -> Callable[[npt.NDArray[Any]], npt.NDArray[Any]] | None:
        """Return a callback to post-process raster data after reading.

        Subclasses can override this to apply transformations to the raw raster data,
        such as harmonization for Sentinel-2 data.

        Args:
            item_name: the name of the item.
            bands: the bands being read.

        Returns:
            A callback function that takes an array and returns a modified array,
            or None if no post-processing is needed.
        """
        return None

    # --- TileStoreDataSource implementation ---

    def get_asset_url(self, item_name: str, asset_key: str) -> str:
        """Not implemented - use read_raster instead which iterates over providers."""
        raise NotImplementedError("Use read_raster instead")

    def read_raster(
        self,
        layer_name: str,
        item_name: str,
        bands: list[str],
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
    ) -> npt.NDArray[Any]:
        """Read raster data, trying multiple providers.

        Args:
            layer_name: the layer name or alias.
            item_name: the item to read.
            bands: the list of bands identifying which specific raster to read.
            projection: the projection to read in.
            bounds: the bounds to read.
            resampling: the resampling method to use in case reprojection is needed.

        Returns:
            the raster data as a numpy array.
        """
        asset_key = self._get_asset_key_by_bands(bands)
        item = self.get_item_by_name(item_name)
        providers_with_asset = self._get_providers_with_asset(item, asset_key)

        if not providers_with_asset:
            raise ValueError(f"No provider has asset {asset_key} for item {item_name}")

        # Shuffle providers to distribute load and provide automatic fallback.
        random.shuffle(providers_with_asset)

        # Try each provider in sequence.
        for idx, (provider, asset) in enumerate(providers_with_asset):
            is_last = idx == len(providers_with_asset) - 1
            try:
                asset_url = self._sign_url(provider.provider_name, asset.url)
                raw_data = self._read_raster_from_url(
                    asset_url, projection, bounds, resampling
                )

                # Apply any post-processing callback.
                callback = self.get_raster_callback(item_name, bands)
                if callback is not None:
                    raw_data = callback(raw_data)

                return raw_data
            except Exception as e:
                if is_last:
                    # If we have gone through all the candidate providers, then raise
                    # the error. rslearn logic outside the data source will handle
                    # retries.
                    raise
                logger.warning(
                    "Failed to read item %s asset %s from %s: %s",
                    item_name,
                    asset.name,
                    provider.provider_name,
                    e,
                )

        # Should never reach here, but satisfy type checker.
        raise RuntimeError("No providers available")

    def get_raster_bands(self, layer_name: str, item_name: str) -> list[list[str]]:
        """Get the sets of bands that have been stored for the specified item.

        Returns the bands from asset_bands for assets that exist on the item.
        """
        item = self.get_item_by_name(item_name)
        all_bands = []
        for asset_key, band_names in self.asset_bands.items():
            # Check if any provider has this asset.
            if self._get_providers_with_asset(item, asset_key):
                all_bands.append(band_names)
        return all_bands


class Sentinel2(OlmoEarthDatasets):
    """Data source for Sentinel-2 L2A data via the OlmoEarth Datasets API.

    This data source queries the OlmoEarth Datasets API for Sentinel-2 L2A scene
    metadata and downloads assets from available providers (Planetary Computer, AWS).
    """

    COLLECTION = "sentinel-2-l2a"

    # Map from asset name to the bands it contains.
    BANDS: dict[str, list[str]] = {
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
        "visual": ["R", "G", "B"],
        "SCL": ["SCL"],
    }

    def __init__(
        self,
        harmonize: bool = False,
        assets: list[str] | None = None,
        api_url: str = DEFAULT_API_URL,
        query: dict[str, Any] | None = None,
        sort_by: str | None = None,
        sort_ascending: bool = True,
        timeout: timedelta = timedelta(seconds=30),
        context: DataSourceContext = DataSourceContext(),
    ):
        """Initialize a new Sentinel2 instance.

        Args:
            harmonize: harmonize pixel values across different processing baselines.
                See https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
            assets: list of asset names to ingest. If None, assets are derived from the
                layer config's band sets (if available), otherwise all default assets.
            api_url: the base URL of the OlmoEarth Datasets API.
            query: optional query filters. Example: {"cloud_cover": {"lt": 10}}.
            sort_by: sort by this property (default None).
            sort_ascending: whether to sort ascending.
            timeout: timeout for API and download requests.
            context: the data source context.
        """
        self.harmonize = harmonize

        # Determine which assets we need based on the layer config.
        if context.layer_config is not None:
            asset_bands: dict[str, list[str]] = {}
            for asset_key, band_names in self.BANDS.items():
                # See if the bands provided by this asset intersect with the bands in
                # at least one configured band set.
                for band_set in context.layer_config.band_sets:
                    if not set(band_set.bands).intersection(set(band_names)):
                        continue
                    asset_bands[asset_key] = band_names
                    break
        elif assets is not None:
            asset_bands = {asset_key: self.BANDS[asset_key] for asset_key in assets}
        else:
            asset_bands = self.BANDS

        super().__init__(
            collection=self.COLLECTION,
            asset_bands=asset_bands,
            api_url=api_url,
            query=query,
            sort_by=sort_by,
            sort_ascending=sort_ascending,
            timeout=timeout,
            context=context,
        )

    def _get_providers_with_asset(
        self, item: OlmoEarthItem, asset_key: str
    ) -> list[tuple[OlmoEarthDataProvider, OlmoEarthAsset]]:
        """Get all providers that have the specified asset by searching for matching bands.

        Different providers use different asset names (e.g., PC uses "B04", AWS uses
        "red"), but they report the same bands. So we search by bands instead of
        asset name for cross-provider compatibility.

        Args:
            item: the item.
            asset_key: the canonical asset name to look for.

        Returns:
            list of (provider, asset) tuples that have an asset with matching bands.
        """
        # Get the expected bands for this asset.
        expected_bands = self.BANDS.get(asset_key)
        if expected_bands is None:
            # Unknown asset, fall back to searching by name.
            return super()._get_providers_with_asset(item, asset_key)

        # Search for assets with matching bands.
        expected_bands_set = set(expected_bands)
        results = []
        for provider in item.data_providers:
            for asset in provider.assets:
                if set(asset.bands) == expected_bands_set:
                    results.append((provider, asset))
                    break
        return results

    def get_raster_callback(
        self, item_name: str, bands: list[str]
    ) -> Callable[[npt.NDArray[Any]], npt.NDArray[Any]] | None:
        """Get harmonization callback for a Sentinel-2 item if needed.

        Checks boa_offset_applied flag first (AWS provider sets this explicitly),
        then falls back to parsing the processing baseline from the scene ID.

        Args:
            item_name: the name of the item.
            bands: the bands being read (TCI/visual doesn't need harmonization).

        Returns:
            A harmonization callback, or None if not needed.
        """
        from rslearn.data_sources.copernicus import get_harmonize_callback_from_scene_id

        # TCI (visual) image does not need harmonization.
        if not self.harmonize or bands == ["R", "G", "B"]:
            return None

        # Check boa_offset_applied flag (AWS provider sets this explicitly).
        item = self.get_item_by_name(item_name)
        for provider in item.data_providers:
            boa_offset_applied = provider.properties.get("boa_offset_applied")
            if boa_offset_applied is False:
                return None

        # Use scene ID to determine harmonization.
        return get_harmonize_callback_from_scene_id(item_name)
