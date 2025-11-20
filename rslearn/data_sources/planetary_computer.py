"""Data on Planetary Computer."""

import json
import os
import tempfile
import xml.etree.ElementTree as ET
from datetime import timedelta
from typing import Any

import affine
import numpy.typing as npt
import planetary_computer
import pystac
import pystac_client
import rasterio
import requests
import shapely
from rasterio.enums import Resampling
from upath import UPath

from rslearn.config import LayerConfig, QueryConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import DataSource, DataSourceContext, Item
from rslearn.data_sources.utils import match_candidate_items_to_window
from rslearn.dataset import Window
from rslearn.dataset.materialize import RasterMaterializer
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStore, TileStoreWithLayer
from rslearn.utils.fsspec import join_upath
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry
from rslearn.utils.raster_format import get_raster_projection_and_bounds

from .copernicus import get_harmonize_callback

logger = get_logger(__name__)


class PlanetaryComputerItem(Item):
    """An item in the PlanetaryComputer data source."""

    def __init__(self, name: str, geometry: STGeometry, asset_urls: dict[str, str]):
        """Creates a new PlanetaryComputerItem.

        Args:
            name: unique name of the item
            geometry: the spatial and temporal extent of the item
            asset_urls: map from asset key to the unsigned asset URL.
        """
        super().__init__(name, geometry)
        self.asset_urls = asset_urls

    def serialize(self) -> dict[str, Any]:
        """Serializes the item to a JSON-encodable dictionary."""
        d = super().serialize()
        d["asset_urls"] = self.asset_urls
        return d

    @staticmethod
    def deserialize(d: dict[str, Any]) -> "PlanetaryComputerItem":
        """Deserializes an item from a JSON-decoded dictionary."""
        item = super(PlanetaryComputerItem, PlanetaryComputerItem).deserialize(d)
        return PlanetaryComputerItem(
            name=item.name,
            geometry=item.geometry,
            asset_urls=d["asset_urls"],
        )


class PlanetaryComputer(DataSource, TileStore):
    """Modality-agnostic data source for data on Microsoft Planetary Computer.

    If there is a subclass available for a modality, it is recommended to use the
    subclass since it provides additional functionality.

    Otherwise, PlanetaryComputer can be configured with the collection name and a
    dictionary of assets and bands to ingest.

    See https://planetarycomputer.microsoft.com/ for details.

    The PC_SDK_SUBSCRIPTION_KEY environment variable can be set for higher rate limits
    but is not needed.
    """

    STAC_ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1"

    # Default threshold for recreating the STAC client to prevent memory leaks
    # from the pystac Catalog's resolved objects cache growing unbounded
    DEFAULT_MAX_ITEMS_PER_CLIENT = 1000

    def __init__(
        self,
        collection_name: str,
        asset_bands: dict[str, list[str]],
        query: dict[str, Any] | None = None,
        sort_by: str | None = None,
        sort_ascending: bool = True,
        timeout: timedelta = timedelta(seconds=10),
        skip_items_missing_assets: bool = False,
        cache_dir: str | None = None,
        max_items_per_client: int | None = None,
        context: DataSourceContext = DataSourceContext(),
    ):
        """Initialize a new PlanetaryComputer instance.

        Args:
            collection_name: the STAC collection name on Planetary Computer.
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
            max_items_per_client: number of STAC items to process before recreating
                the client to prevent memory leaks from the resolved objects cache.
                Defaults to DEFAULT_MAX_ITEMS_PER_CLIENT.
            context: the data source context.
        """
        self.collection_name = collection_name
        self.asset_bands = asset_bands
        self.query = query
        self.sort_by = sort_by
        self.sort_ascending = sort_ascending
        self.timeout = timeout
        self.skip_items_missing_assets = skip_items_missing_assets
        self.max_items_per_client = (
            max_items_per_client or self.DEFAULT_MAX_ITEMS_PER_CLIENT
        )

        if cache_dir is not None:
            if context.ds_path is not None:
                self.cache_dir = join_upath(context.ds_path, cache_dir)
            else:
                self.cache_dir = UPath(cache_dir)

            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

        self.client: pystac_client.Client | None = None
        self._client_item_count = 0

    def _load_client(
        self,
    ) -> pystac_client.Client:
        """Lazily load pystac client.

        We don't load it when creating the data source because it takes time and caller
        may not be calling get_items. Additionally, loading it during the get_items
        call enables leveraging the retry loop functionality in
        prepare_dataset_windows.

        Note: We periodically recreate the client to prevent memory leaks from the
        pystac Catalog's resolved objects cache, which grows unbounded as STAC items
        are deserialized and cached. The cache cannot be cleared or disabled.
        """
        if self.client is None:
            logger.info("Creating initial STAC client")
            self.client = pystac_client.Client.open(self.STAC_ENDPOINT)
            return self.client

        if self._client_item_count < self.max_items_per_client:
            return self.client

        # Recreate client to clear the resolved objects cache
        current_client = self.client
        logger.debug(
            "Recreating STAC client after processing %d items (threshold: %d)",
            self._client_item_count,
            self.max_items_per_client,
        )
        client_root = current_client.get_root()
        client_root.clear_links()
        client_root.clear_items()
        client_root.clear_children()
        self._client_item_count = 0
        self.client = pystac_client.Client.open(self.STAC_ENDPOINT)
        return self.client

    def _stac_item_to_item(self, stac_item: pystac.Item) -> PlanetaryComputerItem:
        shp = shapely.geometry.shape(stac_item.geometry)

        # Get time range.
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
            asset_key: asset_obj.href
            for asset_key, asset_obj in stac_item.assets.items()
        }
        return PlanetaryComputerItem(stac_item.id, geom, asset_urls)

    def get_item_by_name(self, name: str) -> PlanetaryComputerItem:
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
                return PlanetaryComputerItem.deserialize(json.load(f))

        # No cache or not in cache, so we need to make the STAC request.
        logger.debug("Getting STAC item {name}")
        client = self._load_client()

        search_result = client.search(ids=[name], collections=[self.collection_name])
        stac_items = list(search_result.items())

        if not stac_items:
            raise ValueError(
                f"Item {name} not found in collection {self.collection_name}"
            )
        if len(stac_items) > 1:
            raise ValueError(
                f"Multiple items found for ID {name} in collection {self.collection_name}"
            )

        stac_item = stac_items[0]
        item = self._stac_item_to_item(stac_item)

        # Track items processed for client recreation threshold (after deserialization)
        self._client_item_count += 1

        # Finally we cache it if cache_dir is set.
        if cache_fname is not None:
            with cache_fname.open("w") as f:
                json.dump(item.serialize(), f)

        return item

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[PlanetaryComputerItem]]]:
        """Get a list of items in the data source intersecting the given geometries.

        Args:
            geometries: the spatiotemporal geometries
            query_config: the query configuration

        Returns:
            List of groups of items that should be retrieved for each geometry.
        """
        client = self._load_client()

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
            stac_items = [item for item in result.items()]
            # Track items processed for client recreation threshold (after deserialization)
            self._client_item_count += len(stac_items)
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
                self._stac_item_to_item(stac_item) for stac_item in stac_items
            ]

            # Since we made the STAC request, might as well save these to the cache.
            if self.cache_dir is not None:
                for item in candidate_items:
                    cache_fname = self.cache_dir / f"{item.name}.json"
                    if cache_fname.exists():
                        continue
                    with cache_fname.open("w") as f:
                        json.dump(item.serialize(), f)

            cur_groups = match_candidate_items_to_window(
                geometry, candidate_items, query_config
            )
            groups.append(cur_groups)

        return groups

    def deserialize_item(self, serialized_item: Any) -> PlanetaryComputerItem:
        """Deserializes an item from JSON-decoded data."""
        assert isinstance(serialized_item, dict)
        return PlanetaryComputerItem.deserialize(serialized_item)

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[PlanetaryComputerItem],
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

                asset_url = planetary_computer.sign(item.asset_urls[asset_key])

                with tempfile.TemporaryDirectory() as tmp_dir:
                    local_fname = os.path.join(tmp_dir, f"{asset_key}.tif")
                    logger.debug(
                        "PlanetaryComputer download item %s asset %s to %s",
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
                        "PlanetaryComputer ingest item %s asset %s",
                        item.name,
                        asset_key,
                    )
                    tile_store.write_raster_file(
                        item.name, band_names, UPath(local_fname)
                    )

                logger.debug(
                    "PlanetaryComputer done ingesting item %s asset %s",
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
        # Always ready since we wrap accesses to Planetary Computer.
        return True

    def get_raster_bands(self, layer_name: str, item_name: str) -> list[list[str]]:
        """Get the sets of bands that have been stored for the specified item.

        Args:
            layer_name: the layer name or alias.
            item_name: the item.

        Returns:
            a list of lists of bands that are in the tile store (with one raster
                stored corresponding to each inner list). If no rasters are ready for
                this item, returns empty list.
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
        asset_url = planetary_computer.sign(item.asset_urls[asset_key])

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
        RasterMaterializer().materialize(
            TileStoreWithLayer(self, layer_name),
            window,
            layer_name,
            layer_cfg,
            item_groups,
        )


class Sentinel2(PlanetaryComputer):
    """A data source for Sentinel-2 L2A data on Microsoft Planetary Computer.

    See https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a.
    """

    COLLECTION_NAME = "sentinel-2-l2a"

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
        "visual": ["R", "G", "B"],
    }

    def __init__(
        self,
        harmonize: bool = False,
        assets: list[str] | None = None,
        context: DataSourceContext = DataSourceContext(),
        **kwargs: Any,
    ):
        """Initialize a new Sentinel2 instance.

        Args:
            harmonize: harmonize pixel values across different processing baselines,
                see https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
            assets: list of asset names to ingest, or None to ingest all assets. This
                is only used if the layer config is missing from the context.
            context: the data source context.
            kwargs: other arguments to pass to PlanetaryComputer.
        """
        self.harmonize = harmonize

        # Determine which assets we need based on the bands in the layer config.
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
            collection_name=self.COLLECTION_NAME,
            asset_bands=asset_bands,
            # Skip since all of the items should have the same assets.
            skip_items_missing_assets=True,
            context=context,
            **kwargs,
        )

    def _get_product_xml(self, item: PlanetaryComputerItem) -> ET.Element:
        asset_url = planetary_computer.sign(item.asset_urls["product-metadata"])
        response = requests.get(asset_url, timeout=self.timeout.total_seconds())
        response.raise_for_status()
        return ET.fromstring(response.content)

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[PlanetaryComputerItem],
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
                if tile_store.is_raster_ready(item.name, band_names):
                    continue

                asset_url = planetary_computer.sign(item.asset_urls[asset_key])

                with tempfile.TemporaryDirectory() as tmp_dir:
                    local_fname = os.path.join(tmp_dir, f"{asset_key}.tif")
                    logger.debug(
                        "PlanetaryComputer download item %s asset %s to %s",
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
                        "PlanetaryComputer ingest item %s asset %s",
                        item.name,
                        asset_key,
                    )

                    # Harmonize values if needed.
                    # TCI does not need harmonization.
                    harmonize_callback = None
                    if self.harmonize and asset_key != "visual":
                        harmonize_callback = get_harmonize_callback(
                            self._get_product_xml(item)
                        )

                    if harmonize_callback is not None:
                        # In this case we need to read the array, convert the pixel
                        # values, and pass modified array directly to the TileStore.
                        with rasterio.open(local_fname) as src:
                            array = src.read()
                            projection, bounds = get_raster_projection_and_bounds(src)
                        array = harmonize_callback(array)
                        tile_store.write_raster(
                            item.name, band_names, projection, bounds, array
                        )

                    else:
                        tile_store.write_raster_file(
                            item.name, band_names, UPath(local_fname)
                        )

                logger.debug(
                    "PlanetaryComputer done ingesting item %s asset %s",
                    item.name,
                    asset_key,
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
        # We override read_raster because we may need to harmonize the data.
        raw_data = super().read_raster(
            layer_name, item_name, bands, projection, bounds, resampling=resampling
        )

        # TCI (visual) image does not need harmonization.
        if not self.harmonize or bands == self.BANDS["visual"]:
            return raw_data

        item = self.get_item_by_name(item_name)
        harmonize_callback = get_harmonize_callback(self._get_product_xml(item))

        if harmonize_callback is None:
            return raw_data

        array = harmonize_callback(raw_data)
        return array


class Sentinel1(PlanetaryComputer):
    """A data source for Sentinel-1 data on Microsoft Planetary Computer.

    This uses the radiometrically corrected data.

    See https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc.
    """

    COLLECTION_NAME = "sentinel-1-rtc"

    def __init__(
        self,
        band_names: list[str] | None = None,
        context: DataSourceContext = DataSourceContext(),
        **kwargs: Any,
    ):
        """Initialize a new Sentinel1 instance.

        Args:
            band_names: list of bands to try to ingest, if the layer config is missing
                from the context.
            context: the data source context.
            kwargs: additional arguments to pass to PlanetaryComputer.
        """
        # Get band names from the config if possible. If it isn't in the context, then
        # we have to use the provided band names.
        if context.layer_config is not None:
            band_names = list(
                {
                    band
                    for band_set in context.layer_config.band_sets
                    for band in band_set.bands
                }
            )
        if band_names is None:
            raise ValueError(
                "band_names must be set if layer config is not in the context"
            )
        # For Sentinel-1, the asset key should be the same as the band name (and all
        # assets have one band).
        asset_bands = {band: [band] for band in band_names}
        super().__init__(
            collection_name=self.COLLECTION_NAME,
            asset_bands=asset_bands,
            context=context,
            **kwargs,
        )


class Naip(PlanetaryComputer):
    """A data source for NAIP data on Microsoft Planetary Computer.

    See https://planetarycomputer.microsoft.com/dataset/naip.
    """

    COLLECTION_NAME = "naip"
    ASSET_BANDS = {"image": ["R", "G", "B", "NIR"]}

    def __init__(
        self,
        context: DataSourceContext = DataSourceContext(),
        **kwargs: Any,
    ):
        """Initialize a new Naip instance.

        Args:
            context: the data source context.
            kwargs: additional arguments to pass to PlanetaryComputer.
        """
        super().__init__(
            collection_name=self.COLLECTION_NAME,
            asset_bands=self.ASSET_BANDS,
            context=context,
            **kwargs,
        )
