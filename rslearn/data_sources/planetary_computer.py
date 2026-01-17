"""Data on Planetary Computer."""

import os
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Any

import affine
import numpy.typing as npt
import planetary_computer
import rasterio
import requests
from rasterio.enums import Resampling
from upath import UPath

from rslearn.config import LayerConfig
from rslearn.data_sources import DataSourceContext
from rslearn.data_sources.stac import SourceItem, StacDataSource
from rslearn.dataset import Window
from rslearn.dataset.materialize import RasterMaterializer
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStore, TileStoreWithLayer
from rslearn.utils.fsspec import join_upath
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry
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
        """Execute a STAC item search with automatic ID pagination fallback.

        If the initial query returns PLANETARY_COMPUTER_LIMIT items, this method
        switches to ID-based pagination to retrieve all matching items.

        Args:
            collections: only search within the provided collection(s).
            bbox: only return features intersecting the provided bounding box.
            intersects: only return features intersecting this GeoJSON geometry.
            date_time: only return features that have a temporal property intersecting
                the provided time range or timestamp.
            ids: only return the provided item IDs.
            limit: number of items per page.
            query: query dict for STAC query extension.
            sortby: list of sort specifications.

        Returns:
            list of matching STAC items.
        """
        # First, try a simple query with the PC limit to detect if pagination is needed.
        # We always use PLANETARY_COMPUTER_LIMIT for the request because PC doesn't
        # support standard pagination, and we need to detect when we hit the limit
        # to switch to ID-based pagination.
        stac_items = super().search(
            collections=collections,
            bbox=bbox,
            intersects=intersects,
            date_time=date_time,
            ids=ids,
            limit=PLANETARY_COMPUTER_LIMIT,
            query=query,
            sortby=sortby,
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


class PlanetaryComputer(StacDataSource, TileStore):
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

        super().__init__(
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

        self.asset_bands = asset_bands
        self.timeout = timeout
        self.skip_items_missing_assets = skip_items_missing_assets

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
        item_groups: list[list[SourceItem]],
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
