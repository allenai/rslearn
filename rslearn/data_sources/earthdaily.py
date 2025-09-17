"""Data on EarthDaily."""

import json
import os
import tempfile
from datetime import timedelta
from typing import Any, Literal

import affine
import numpy.typing as npt
import pystac
import pystac_client
import rasterio
import requests
import shapely
from earthdaily import EDSClient, EDSConfig
from rasterio.enums import Resampling
from upath import UPath

from rslearn.config import LayerConfig, QueryConfig, RasterLayerConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import DataSource, Item
from rslearn.data_sources.utils import match_candidate_items_to_window
from rslearn.dataset import Window
from rslearn.dataset.materialize import RasterMaterializer
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStore, TileStoreWithLayer
from rslearn.utils.fsspec import join_upath
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry

logger = get_logger(__name__)


class EarthDailyItem(Item):
    """An item in the EarthDaily data source."""

    def __init__(self, name: str, geometry: STGeometry, asset_urls: dict[str, str]):
        """Creates a new EarthDailyItem.

        Args:
            name: unique name of the item
            geometry: the spatial and temporal extent of the item
            asset_urls: map from asset key to the asset URL.
        """
        super().__init__(name, geometry)
        self.asset_urls = asset_urls

    def serialize(self) -> dict[str, Any]:
        """Serializes the item to a JSON-encodable dictionary."""
        d = super().serialize()
        d["asset_urls"] = self.asset_urls
        return d

    @staticmethod
    def deserialize(d: dict[str, Any]) -> "EarthDailyItem":
        """Deserializes an item from a JSON-decoded dictionary."""
        item = super(EarthDailyItem, EarthDailyItem).deserialize(d)
        return EarthDailyItem(
            name=item.name,
            geometry=item.geometry,
            asset_urls=d["asset_urls"],
        )


class EarthDaily(DataSource, TileStore):
    """A data source for EarthDaily data.

    This requires the following environment variables to be set:
    - EDS_CLIENT_ID
    - EDS_SECRET
    - EDS_AUTH_URL
    - EDS_API_URL
    """

    def __init__(
        self,
        collection_name: str,
        asset_bands: dict[str, list[str]],
        query: dict[str, Any] | None = None,
        sort_by: str | None = None,
        sort_ascending: bool = True,
        timeout: timedelta = timedelta(seconds=10),
        skip_items_missing_assets: bool = False,
        cache_dir: UPath | None = None,
        max_retries: int = 3,
        retry_backoff_factor: float = 5.0,
        service_name: Literal["platform"] = "platform",
    ):
        """Initialize a new EarthDaily instance.

        Args:
            collection_name: the STAC collection name on EarthDaily.
            asset_bands: assets to ingest, mapping from asset name to the list of bands
                in that asset.
            query: optional query argument to STAC searches.
            sort_by: sort by this property in the STAC items.
            sort_ascending: whether to sort ascending (or descending).
            timeout: timeout for API requests.
            skip_items_missing_assets: skip STAC items that are missing any of the
                assets in asset_bands during get_items.
            cache_dir: optional directory to cache items by name, including asset URLs.
                If not set, there will be no cache and instead STAC requests will be
                needed each time.
            max_retries: the maximum number of retry attempts for HTTP requests that fail
                due to transient errors (e.g., 429, 500, 502, 503, 504 status codes).
            retry_backoff_factor: backoff factor for exponential retry delays between HTTP
                request attempts.  The delay between retries is calculated using the formula:
                `(retry_backoff_factor * (2 ** (retry_count - 1)))` seconds.
            service_name: the service name, only "platform" is supported, the other
                services "legacy" and "internal" are not supported.
        """
        self.collection_name = collection_name
        self.asset_bands = asset_bands
        self.query = query
        self.sort_by = sort_by
        self.sort_ascending = sort_ascending
        self.timeout = timeout
        self.skip_items_missing_assets = skip_items_missing_assets
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor
        self.service_name = service_name

        if cache_dir is not None:
            self.cache_dir = cache_dir
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.eds_client: EDSClient | None = None
        self.client: pystac_client.Client | None = None
        self.collection: pystac_client.CollectionClient | None = None

    @staticmethod
    def from_config(config: RasterLayerConfig, ds_path: UPath) -> "EarthDaily":
        """Creates a new EarthDaily instance from a configuration dictionary."""
        if config.data_source is None:
            raise ValueError("config.data_source is required")
        d = config.data_source.config_dict

        kwargs: dict[str, Any] = dict(
            collection_name=d["collection_name"],
            service_name=d["service_name"],
            asset_bands=d["asset_bands"],
        )

        if "timeout_seconds" in d:
            kwargs["timeout"] = timedelta(seconds=d["timeout_seconds"])

        if "cache_dir" in d:
            kwargs["cache_dir"] = join_upath(ds_path, d["cache_dir"])

        if "max_retries" in d:
            kwargs["max_retries"] = d["max_retries"]

        if "retry_backoff_factor" in d:
            kwargs["retry_backoff_factor"] = d["retry_backoff_factor"]

        simple_optionals = ["query", "sort_by", "sort_ascending"]
        for k in simple_optionals:
            if k in d:
                kwargs[k] = d[k]

        return EarthDaily(**kwargs)

    def _load_client(
        self,
    ) -> tuple[EDSClient, pystac_client.Client, pystac_client.CollectionClient]:
        """Lazily load EDS client.

        We don't load it when creating the data source because it takes time and caller
        may not be calling get_items. Additionally, loading it during the get_items
        call enables leveraging the retry loop functionality in
        prepare_dataset_windows.
        """
        if self.eds_client is not None:
            return self.eds_client, self.client, self.collection

        self.eds_client = EDSClient(
            EDSConfig(
                max_retries=self.max_retries,
                retry_backoff_factor=self.retry_backoff_factor,
            )
        )

        if self.service_name == "platform":
            self.client = self.eds_client.platform.pystac_client
            self.collection = self.client.get_collection(self.collection_name)
        else:
            raise ValueError(f"Invalid service name: {self.service_name}")

        return self.eds_client, self.client, self.collection

    def _stac_item_to_item(self, stac_item: pystac.Item) -> EarthDailyItem:
        shp = shapely.geometry.shape(stac_item.geometry)

        metadata = stac_item.common_metadata
        if metadata.start_datetime is not None and metadata.end_datetime is not None:
            time_range = (
                metadata.start_datetime,
                metadata.end_datetime,
            )
        elif stac_item.datetime is not None:
            time_range = (stac_item.datetime, stac_item.datetime)
        else:
            raise ValueError(
                f"item {stac_item.id} unexpectedly missing start_datetime, end_datetime, and datetime"
            )

        geom = STGeometry(WGS84_PROJECTION, shp, time_range)
        asset_urls = {
            asset_key: asset_obj.extra_fields["alternate"]["download"]["href"]
            for asset_key, asset_obj in stac_item.assets.items()
            if "alternate" in asset_obj.extra_fields
            and "download" in asset_obj.extra_fields["alternate"]
            and "href" in asset_obj.extra_fields["alternate"]["download"]
        }
        return EarthDailyItem(stac_item.id, geom, asset_urls)

    def get_item_by_name(self, name: str) -> EarthDailyItem:
        """Gets an item by name.

        Args:
            name: the name of the item to get

        Returns:
            the item object
        """
        # If cache_dir is set, we cache the item. First here we check if it is already
        # in the cache.
        cache_fname: UPath | None = None
        if self.cache_dir:
            cache_fname = self.cache_dir / f"{name}.json"
        if cache_fname is not None and cache_fname.exists():
            with cache_fname.open() as f:
                return EarthDailyItem.deserialize(json.load(f))

        # No cache or not in cache, so we need to make the STAC request.
        _, _, collection = self._load_client()
        stac_item = collection.get_item(name)
        item = self._stac_item_to_item(stac_item)

        # Finally we cache it if cache_dir is set.
        if cache_fname is not None:
            with cache_fname.open("w") as f:
                json.dump(item.serialize(), f)

        return item

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[EarthDailyItem]]]:
        """Get a list of items in the data source intersecting the given geometries.

        Args:
            geometries: the spatiotemporal geometries
            query_config: the query configuration
        """
        _, client, _ = self._load_client()

        groups = []
        for geometry in geometries:
            # Get potentially relevant items from the collection by performing one search
            # for each requested geometry.
            wgs84_geometry = geometry.to_projection(WGS84_PROJECTION)
            logger.debug("performing STAC search for geometry %s", wgs84_geometry)
            result = client.search(
                collections=[self.collection_name],
                intersects=shapely.to_geojson(wgs84_geometry.shp),
                datetime=wgs84_geometry.time_range,
                query=self.query,
            )
            stac_items = [item for item in result.item_collection()]
            logger.debug("STAC search yielded %d items", len(stac_items))

            if self.skip_items_missing_assets:
                # Filter out items that are missing any of the assets in self.asset_bands.
                good_stac_items = []
                for stac_item in stac_items:
                    good = True
                    for asset_key in self.asset_bands.keys():
                        if asset_key in stac_item.assets:
                            continue
                        good = False
                        break
                    if good:
                        good_stac_items.append(stac_item)
                logger.debug(
                    "skip_items_missing_assets filter from %d to %d items",
                    len(stac_items),
                    len(good_stac_items),
                )
                stac_items = good_stac_items

            if self.sort_by is not None:
                stac_items.sort(
                    key=lambda stac_item: stac_item.properties[self.sort_by],
                    reverse=not self.sort_ascending,
                )

            candidate_items = [
                # The only way to get the asset URLs is to get the item by name.
                self.get_item_by_name(stac_item.id)
                for stac_item in stac_items
            ]

            cur_groups = match_candidate_items_to_window(
                geometry, candidate_items, query_config
            )
            groups.append(cur_groups)

        return groups

    def deserialize_item(self, serialized_item: Any) -> EarthDailyItem:
        """Deserializes an item from JSON-decoded data."""
        assert isinstance(serialized_item, dict)
        return EarthDailyItem.deserialize(serialized_item)

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[EarthDailyItem],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        Args:
            tile_store: the tile store to ingest into
            items: the items to ingest
            geometries: a list of geometries needed for each item
        """
        for item in items:
            for asset_key, band_names in self.asset_bands.items():
                if asset_key not in item.asset_urls:
                    continue
                if tile_store.is_raster_ready(item.name, band_names):
                    continue

                asset_url = item.asset_urls[asset_key]
                with tempfile.TemporaryDirectory() as tmp_dir:
                    local_fname = os.path.join(tmp_dir, f"{asset_key}.tif")
                    logger.debug(
                        "EarthDaily download item %s asset %s to %s",
                        item.name,
                        asset_key,
                        local_fname,
                    )
                    with requests.get(
                        asset_url, stream=True, timeout=self.timeout.total_seconds()
                    ) as r:
                        r.raise_for_status()
                        with open(local_fname, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)

                    logger.debug(
                        "EarthDaily ingest item %s asset %s",
                        item.name,
                        asset_key,
                    )
                    tile_store.write_raster_file(
                        item.name, band_names, UPath(local_fname)
                    )

                logger.debug(
                    "EarthDaily done ingesting item %s asset %s",
                    item.name,
                    asset_key,
                )

    def is_raster_ready(
        self, layer_name: str, item_name: str, bands: list[str]
    ) -> bool:
        """Checks if this raster has been written to the store.

        Args:
            layer_name: the layer name or alias.
            item_name: the item.
            bands: the list of bands identifying which specific raster to read.

        Returns:
            whether there is a raster in the store matching the source, item, and
                bands.
        """
        # Always ready since we wrap accesses to EarthDaily.
        return True

    def get_raster_bands(self, layer_name: str, item_name: str) -> list[list[str]]:
        """Get the sets of bands that have been stored for the specified item.

        Args:
            layer_name: the layer name or alias.
            item_name: the item.
        """
        if self.skip_items_missing_assets:
            # In this case we can assume that the item has all of the assets.
            return list(self.asset_bands.values())

        # Otherwise we have to lookup the STAC item to see which assets it has.
        # Here we use get_item_by_name since it handles caching.
        item = self.get_item_by_name(item_name)
        all_bands = []
        for asset_key, band_names in self.asset_bands.items():
            if asset_key not in item.asset_urls:
                continue
            all_bands.append(band_names)
        return all_bands

    def _get_asset_by_band(self, bands: list[str]) -> str:
        """Get the name of the asset based on the band names."""
        for asset_key, asset_bands in self.asset_bands.items():
            if bands == asset_bands:
                return asset_key

        raise ValueError(f"no raster with bands {bands}")

    def get_raster_bounds(
        self, layer_name: str, item_name: str, bands: list[str], projection: Projection
    ) -> PixelBounds:
        """Get the bounds of the raster in the specified projection.

        Args:
            layer_name: the layer name or alias.
            item_name: the item to check.
            bands: the list of bands identifying which specific raster to read. These
                bands must match the bands of a stored raster.
            projection: the projection to get the raster's bounds in.

        Returns:
            the bounds of the raster in the projection.
        """
        item = self.get_item_by_name(item_name)
        geom = item.geometry.to_projection(projection)
        return (
            int(geom.shp.bounds[0]),
            int(geom.shp.bounds[1]),
            int(geom.shp.bounds[2]),
            int(geom.shp.bounds[3]),
        )

    def read_raster(
        self,
        layer_name: str,
        item_name: str,
        bands: list[str],
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
    ) -> npt.NDArray[Any]:
        """Read raster data from the store.

        Args:
            layer_name: the layer name or alias.
            item_name: the item to read.
            bands: the list of bands identifying which specific raster to read. These
                bands must match the bands of a stored raster.
            projection: the projection to read in.
            bounds: the bounds to read.
            resampling: the resampling method to use in case reprojection is needed.

        Returns:
            the raster data
        """
        asset_key = self._get_asset_by_band(bands)
        item = self.get_item_by_name(item_name)
        asset_url = item.asset_urls[asset_key]

        # Construct the transform to use for the warped dataset.
        wanted_transform = affine.Affine(
            projection.x_resolution,
            0,
            bounds[0] * projection.x_resolution,
            0,
            projection.y_resolution,
            bounds[1] * projection.y_resolution,
        )

        with rasterio.open(asset_url) as src:
            with rasterio.vrt.WarpedVRT(
                src,
                crs=projection.crs,
                transform=wanted_transform,
                width=bounds[2] - bounds[0],
                height=bounds[3] - bounds[1],
                resampling=resampling,
            ) as vrt:
                return vrt.read()

    def materialize(
        self,
        window: Window,
        item_groups: list[list[Item]],
        layer_name: str,
        layer_cfg: LayerConfig,
    ) -> None:
        """Materialize data for the window.

        Args:
            window: the window to materialize
            item_groups: the items from get_items
            layer_name: the name of this layer
            layer_cfg: the config of this layer
        """
        assert isinstance(layer_cfg, RasterLayerConfig)
        RasterMaterializer().materialize(
            TileStoreWithLayer(self, layer_name),
            window,
            layer_name,
            layer_cfg,
            item_groups,
        )
