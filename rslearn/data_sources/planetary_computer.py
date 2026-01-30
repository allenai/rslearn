"""Data on Planetary Computer."""

import os
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import numpy.typing as npt
import planetary_computer
import rasterio
import requests
from typing_extensions import override
from upath import UPath

from rslearn.data_sources import DataSourceContext
from rslearn.data_sources.stac import SourceItem, StacDataSource
from rslearn.data_sources.tile_store_data_source import TileStoreDataSource
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.fsspec import join_upath
from rslearn.utils.geometry import STGeometry
from rslearn.utils.raster_format import get_raster_projection_and_bounds
from rslearn.utils.stac import StacClient, StacItem

from .copernicus import get_harmonize_callback

logger = get_logger(__name__)

# Max limit accepted by Planetary Computer API.
PLANETARY_COMPUTER_LIMIT = 1000


class PlanetaryComputerStacClient(StacClient):
    """A StacClient subclass that handles Planetary Computer's pagination limits.

    Planetary Computer STAC API does not support standard pagination and has a max
    limit of 1000. If the initial query returns 1000 items, this client paginates
    by sorting by ID and using gt (greater than) queries to fetch subsequent pages.
    """

    @override
    def search(
        self,
        collections: list[str] | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        intersects: dict[str, Any] | None = None,
        date_time: datetime | tuple[datetime, datetime] | None = None,
        ids: list[str] | None = None,
        limit: int | None = None,
        query: dict[str, Any] | None = None,
        sortby: list[dict[str, str]] | None = None,
    ) -> list[StacItem]:
        # We will use sortby for pagination, so the caller must not set it.
        if sortby is not None:
            raise ValueError("sortby must not be set for PlanetaryComputerStacClient")

        # First, try a simple query with the PC limit to detect if pagination is needed.
        # We always use PLANETARY_COMPUTER_LIMIT for the request because PC doesn't
        # support standard pagination, and we need to detect when we hit the limit
        # to switch to ID-based pagination.
        # We could just start sorting by ID here and do pagination, but we treate it as
        # a special case to avoid sorting since that seems to speed up the query.
        stac_items = super().search(
            collections=collections,
            bbox=bbox,
            intersects=intersects,
            date_time=date_time,
            ids=ids,
            limit=PLANETARY_COMPUTER_LIMIT,
            query=query,
        )

        # If we got fewer than the PC limit, we have all the results.
        if len(stac_items) < PLANETARY_COMPUTER_LIMIT:
            return stac_items

        # We hit the limit, so we need to paginate by ID.
        # Re-fetch with sorting by ID to ensure consistent ordering for pagination.
        logger.debug(
            "Initial request returned %d items (at limit), switching to ID pagination",
            len(stac_items),
        )

        all_items: list[StacItem] = []
        last_id: str | None = None

        while True:
            # Build query with id > last_id if we're paginating.
            combined_query: dict[str, Any] = dict(query) if query else {}
            if last_id is not None:
                combined_query["id"] = {"gt": last_id}

            stac_items = super().search(
                collections=collections,
                bbox=bbox,
                intersects=intersects,
                date_time=date_time,
                ids=ids,
                limit=PLANETARY_COMPUTER_LIMIT,
                query=combined_query if combined_query else None,
                sortby=[{"field": "id", "direction": "asc"}],
            )

            all_items.extend(stac_items)

            # If we got fewer than the limit, we've fetched everything.
            if len(stac_items) < PLANETARY_COMPUTER_LIMIT:
                break

            # Otherwise, paginate using the last item's ID.
            last_id = stac_items[-1].id
            logger.debug(
                "Got %d items, paginating with id > %s",
                len(stac_items),
                last_id,
            )

        logger.debug("Total items fetched: %d", len(all_items))
        return all_items


class PlanetaryComputer(TileStoreDataSource[SourceItem], StacDataSource):
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
            context: the data source context.
        """
        # Initialize the TileStoreDataSource with asset_bands
        TileStoreDataSource.__init__(self, asset_bands=asset_bands)

        # Determine the cache_dir to use.
        cache_upath: UPath | None = None
        if cache_dir is not None:
            if context.ds_path is not None:
                cache_upath = join_upath(context.ds_path, cache_dir)
            else:
                cache_upath = UPath(cache_dir)

            cache_upath.mkdir(parents=True, exist_ok=True)

        # We pass required_assets to StacDataSource of skip_items_missing_assets is set.
        required_assets: list[str] | None = None
        if skip_items_missing_assets:
            required_assets = list(asset_bands.keys())

        StacDataSource.__init__(
            self,
            endpoint=self.STAC_ENDPOINT,
            collection_name=collection_name,
            query=query,
            sort_by=sort_by,
            sort_ascending=sort_ascending,
            required_assets=required_assets,
            cache_dir=cache_upath,
        )

        # Replace the client with PlanetaryComputerStacClient to handle PC's pagination limits.
        self.client = PlanetaryComputerStacClient(self.STAC_ENDPOINT)

        self.timeout = timeout
        self.skip_items_missing_assets = skip_items_missing_assets

    # --- TileStoreDataSource implementation ---

    def get_asset_url(self, item_name: str, asset_key: str) -> str:
        """Get the signed URL to read the asset for the given item and asset key.

        Args:
            item_name: the name of the item.
            asset_key: the key identifying which asset to get.

        Returns:
            the signed URL to read the asset from.
        """
        item = self.get_item_by_name(item_name)
        return planetary_computer.sign(item.asset_urls[asset_key])

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

    # --- DataSource implementation ---

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[SourceItem],
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

    def _get_product_xml(self, item: SourceItem) -> ET.Element:
        asset_url = planetary_computer.sign(item.asset_urls["product-metadata"])
        response = requests.get(asset_url, timeout=self.timeout.total_seconds())
        response.raise_for_status()
        return ET.fromstring(response.content)

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[SourceItem],
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

    def get_read_callback(
        self, item_name: str, asset_key: str
    ) -> Callable[[npt.NDArray[Any]], npt.NDArray[Any]] | None:
        """Return a callback to harmonize Sentinel-2 data if needed.

        Args:
            item_name: the name of the item being read.
            asset_key: the key identifying which asset is being read.

        Returns:
            A callback function for harmonization, or None if not needed.
        """
        # TCI (visual) image does not need harmonization.
        if not self.harmonize or asset_key == "visual":
            return None

        item = self.get_item_by_name(item_name)
        return get_harmonize_callback(self._get_product_xml(item))


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


class CopDemGlo30(PlanetaryComputer):
    """A data source for Copernicus DEM GLO-30 (30m) on Microsoft Planetary Computer.

    See https://planetarycomputer.microsoft.com/dataset/cop-dem-glo-30.
    """

    COLLECTION_NAME = "cop-dem-glo-30"
    DATA_ASSET = "data"

    def __init__(
        self,
        band_name: str = "DEM",
        context: DataSourceContext = DataSourceContext(),
        **kwargs: Any,
    ):
        """Initialize a new CopDemGlo30 instance.

        Args:
            band_name: band name to use if the layer config is missing from the
                context.
            context: the data source context.
            kwargs: additional arguments to pass to PlanetaryComputer.
        """
        if context.layer_config is not None:
            if len(context.layer_config.band_sets) != 1:
                raise ValueError("expected a single band set")
            if len(context.layer_config.band_sets[0].bands) != 1:
                raise ValueError("expected band set to have a single band")
            band_name = context.layer_config.band_sets[0].bands[0]

        super().__init__(
            collection_name=self.COLLECTION_NAME,
            asset_bands={self.DATA_ASSET: [band_name]},
            # Skip since all items should have the same asset(s).
            skip_items_missing_assets=True,
            context=context,
            **kwargs,
        )

    def _stac_item_to_item(self, stac_item: Any) -> SourceItem:
        # Copernicus DEM is static; ignore item timestamps so it matches any window.
        item = super()._stac_item_to_item(stac_item)
        item.geometry = STGeometry(item.geometry.projection, item.geometry.shp, None)
        return item

    def _get_search_time_range(self, geometry: STGeometry) -> None:
        # Copernicus DEM is static; do not filter STAC searches by time.
        return None
