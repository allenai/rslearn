"""Data on EarthDaily."""

import copy
import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import affine
import numpy as np
import numpy.typing as npt
import pystac
import pystac_client
import rasterio
import requests
import shapely
from earthdaily import EDSClient, EDSConfig
from rasterio.enums import Resampling
from upath import UPath

from rslearn.config import DType, LayerConfig, QueryConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import DataSource, DataSourceContext, Item
from rslearn.data_sources.utils import MatchedItemGroup, match_candidate_items_to_window
from rslearn.dataset import Window
from rslearn.dataset.materialize import RasterMaterializer
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStore, TileStoreWithLayer
from rslearn.utils.array import nodata_eq, unique_nodata_value
from rslearn.utils.fsspec import join_upath
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry
from rslearn.utils.raster_array import RasterArray, RasterMetadata
from rslearn.utils.raster_format import get_raster_projection_and_bounds

from .copernicus import get_harmonize_callback

logger = get_logger(__name__)


class EarthDailyItem(Item):
    """An item in the EarthDaily data source."""

    def __init__(
        self,
        name: str,
        geometry: STGeometry,
        asset_urls: dict[str, str],
        asset_scale_offsets: dict[str, list[dict[str, float | None]]] | None = None,
        product_id: str | None = None,
    ):
        """Creates a new EarthDailyItem.

        Args:
            name: unique name of the item
            geometry: the spatial and temporal extent of the item
            asset_urls: map from asset key to the asset URL.
            asset_scale_offsets: optional per-asset scale/offset metadata. Each asset key
                maps to a list of dictionaries (one per raster band) with keys
                "scale", "offset", and "nodata".
            product_id: optional Sentinel-2 product ID from STAC
                `properties["sentinel:product_id"]`.
        """
        super().__init__(name, geometry)
        self.asset_urls = asset_urls
        self.asset_scale_offsets = asset_scale_offsets or {}
        self.product_id = product_id

    def serialize(self) -> dict[str, Any]:
        """Serializes the item to a JSON-encodable dictionary."""
        d = super().serialize()
        d["asset_urls"] = self.asset_urls
        d["asset_scale_offsets"] = self.asset_scale_offsets
        d["product_id"] = self.product_id
        return d

    @staticmethod
    def deserialize(d: dict[str, Any]) -> "EarthDailyItem":
        """Deserializes an item from a JSON-decoded dictionary."""
        item = super(EarthDailyItem, EarthDailyItem).deserialize(d)
        return EarthDailyItem(
            name=item.name,
            geometry=item.geometry,
            asset_urls=d["asset_urls"],
            asset_scale_offsets=d.get("asset_scale_offsets") or {},
            product_id=d.get("product_id"),
        )


class EarthDaily(DataSource, TileStore):
    """A data source for EarthDaily data.

    This requires the following environment variables to be set:
    - EDS_CLIENT_ID
    - EDS_SECRET
    - EDS_AUTH_URL
    - EDS_API_URL
    """

    METADATA_ASSET_KEYS = frozenset({"product_metadata", "granule_metadata"})

    def __init__(
        self,
        collection_name: str,
        asset_bands: dict[str, list[str]],
        query: dict[str, Any] | None = None,
        sort_by: str | None = None,
        sort_ascending: bool = True,
        cloud_cover_max: float | None = None,
        search_max_items: int | None = None,
        sort_items_by: Literal["cloud_cover", "datetime"] | None = None,
        timeout: timedelta = timedelta(seconds=10),
        skip_items_missing_assets: bool = False,
        cache_dir: str | None = None,
        max_retries: int = 3,
        retry_backoff_factor: float = 5.0,
        read_scale_offsets: bool = True,
        context: DataSourceContext = DataSourceContext(),
    ):
        """Initialize a new EarthDaily instance.

        Args:
            collection_name: the STAC collection name on EarthDaily.
            asset_bands: assets to ingest, mapping from asset name to the list of bands
                in that asset.
            query: optional query argument to STAC searches.
            sort_by: sort by this property in the STAC items.
            sort_ascending: whether to sort ascending (or descending).
            cloud_cover_max: max cloud cover (%) injected into the STAC query.
            search_max_items: max STAC items fetched per geometry. None means no limit.
            sort_items_by: secondary sort applied after search: "cloud_cover", "datetime",
                or None.
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
            read_scale_offsets: whether to parse per-band `scale`/`offset` metadata from
                STAC `raster:bands`.
            context: the data source context.
        """
        self.collection_name = collection_name
        self.asset_bands = asset_bands
        self.query = query
        self.sort_by = sort_by
        self.sort_ascending = sort_ascending
        self.cloud_cover_max = cloud_cover_max
        self.search_max_items = search_max_items
        self.sort_items_by = sort_items_by
        self.timeout = timeout
        self.skip_items_missing_assets = skip_items_missing_assets
        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor
        self.read_scale_offsets = read_scale_offsets

        if cache_dir is not None:
            # Use dataset path as root if provided.
            if context.ds_path is not None:
                self.cache_dir = join_upath(context.ds_path, cache_dir)
            else:
                self.cache_dir = UPath(cache_dir)

            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

        self.eds_client: EDSClient | None = None
        self.client: pystac_client.Client | None = None
        self.collection: pystac_client.CollectionClient | None = None
        self._nodata_cache: dict[str, float | None] = {}

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

        self.client = self.eds_client.platform.pystac_client
        self.collection = self.client.get_collection(self.collection_name)

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
        asset_urls: dict[str, str] = {}
        asset_scale_offsets: dict[str, list[dict[str, float | None]]] = {}
        for asset_key, asset_obj in stac_item.assets.items():
            is_metadata_asset = asset_key in self.METADATA_ASSET_KEYS
            if asset_key not in self.asset_bands and not is_metadata_asset:
                continue

            href: str | None = None
            alt = asset_obj.extra_fields.get("alternate")
            if isinstance(alt, dict):
                download = alt.get("download")
                if isinstance(download, dict):
                    raw_href = download.get("href")
                    if isinstance(raw_href, str) and raw_href:
                        href = raw_href

            if href is None:
                if is_metadata_asset and isinstance(asset_obj.href, str):
                    href = asset_obj.href
                else:
                    raise ValueError(
                        f"item {stac_item.id} asset {asset_key} is missing "
                        "alternate.download.href"
                    )

            asset_urls[asset_key] = href

            if is_metadata_asset or not self.read_scale_offsets:
                continue

            raster_bands = asset_obj.extra_fields.get("raster:bands", [])
            if not isinstance(raster_bands, list) or not raster_bands:
                continue
            scale_offsets: list[dict[str, float | None]] = []
            for band_meta in raster_bands:
                if not isinstance(band_meta, dict):
                    continue
                raw_scale = band_meta.get("scale")
                raw_offset = band_meta.get("offset")
                try:
                    scale = float(raw_scale) if raw_scale is not None else 1.0
                except (TypeError, ValueError):
                    scale = 1.0
                try:
                    offset = float(raw_offset) if raw_offset is not None else 0.0
                except (TypeError, ValueError):
                    offset = 0.0
                raw_nodata = band_meta.get("nodata")
                try:
                    nodata = float(raw_nodata) if raw_nodata is not None else None
                except (TypeError, ValueError):
                    nodata = None
                parsed: dict[str, float | None] = {
                    "scale": scale,
                    "offset": offset,
                    "nodata": nodata,
                }
                scale_offsets.append(parsed)
            if scale_offsets:
                asset_scale_offsets[asset_key] = scale_offsets

        product_id = None
        for property_key in ("sentinel:product_id", "s2:product_uri", "s2:product_id"):
            raw_product_id = stac_item.properties.get(property_key)
            if isinstance(raw_product_id, str) and raw_product_id:
                product_id = raw_product_id
                break

        return EarthDailyItem(
            stac_item.id,
            geom,
            asset_urls,
            asset_scale_offsets=asset_scale_offsets,
            product_id=product_id,
        )

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
                cached = json.load(f)
            if isinstance(cached, dict):
                return EarthDailyItem.deserialize(cached)

        # No cache or not in cache, so we need to make the STAC request.
        _, _, collection = self._load_client()
        stac_item = collection.get_item(name)
        if stac_item is None:
            raise KeyError(f"EarthDaily item not found: {name}")
        item = self._stac_item_to_item(stac_item)

        # Finally we cache it if cache_dir is set.
        if cache_fname is not None:
            with cache_fname.open("w") as f:
                json.dump(item.serialize(), f)

        return item

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[MatchedItemGroup[EarthDailyItem]]]:
        """Get a list of items in the data source intersecting the given geometries.

        Args:
            geometries: the spatiotemporal geometries
            query_config: the query configuration
        """
        _, client, _ = self._load_client()

        groups = []
        for geometry in geometries:
            wgs84_geometry = geometry.to_wgs84()
            logger.debug("performing STAC search for geometry %s", wgs84_geometry)

            max_cloud_cover = self.cloud_cover_max
            effective_query: dict[str, Any] | None = (
                copy.deepcopy(self.query) if self.query is not None else None
            )
            if max_cloud_cover is not None:
                if effective_query is None:
                    effective_query = {}
                effective_query["eo:cloud_cover"] = {"lt": max_cloud_cover}

            result = client.search(
                collections=[self.collection_name],
                intersects=shapely.to_geojson(wgs84_geometry.shp),
                datetime=wgs84_geometry.time_range,
                query=effective_query,
                max_items=self.search_max_items,
            )
            stac_items = list(result.item_collection())
            logger.debug("STAC search yielded %d items", len(stac_items))

            if self.skip_items_missing_assets:
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
                if self.sort_by == "datetime":
                    stac_items.sort(
                        key=lambda item: (item.datetime is None, item.datetime),
                        reverse=not self.sort_ascending,
                    )
                else:
                    stac_items.sort(
                        key=lambda item: (
                            item.properties.get(self.sort_by) is None,
                            item.properties.get(self.sort_by),
                        ),
                        reverse=not self.sort_ascending,
                    )
            elif self.sort_items_by == "cloud_cover":
                stac_items.sort(
                    key=lambda item: (
                        item.properties.get("eo:cloud_cover") is None,
                        item.properties.get("eo:cloud_cover", 0.0),
                    )
                )
            elif self.sort_items_by == "datetime":
                stac_items.sort(key=lambda item: (item.datetime is None, item.datetime))
            elif self.sort_items_by is not None:
                raise ValueError(
                    f"invalid sort_items_by setting ({self.sort_items_by})"
                )

            candidate_items = [
                self.get_item_by_name(stac_item.id) for stac_item in stac_items
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

    def _download_asset_to_tmp(
        self, asset_url: str, tmp_dir: str, asset_key: str, item_name: str
    ) -> str:
        """Download an asset URL to a temporary GeoTIFF path."""
        local_fname = os.path.join(tmp_dir, f"{item_name}_{asset_key}.tif")
        with requests.get(
            asset_url, stream=True, timeout=self.timeout.total_seconds()
        ) as r:
            r.raise_for_status()
            with open(local_fname, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_fname

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
                if tile_store.is_raster_ready(item, band_names):
                    continue

                asset_url = item.asset_urls[asset_key]
                with tempfile.TemporaryDirectory() as tmp_dir:
                    local_fname = self._download_asset_to_tmp(
                        asset_url, tmp_dir, asset_key, item.name
                    )
                    logger.debug(
                        "EarthDaily download item %s asset %s to %s",
                        item.name,
                        asset_key,
                        local_fname,
                    )

                    logger.debug(
                        "EarthDaily ingest item %s asset %s",
                        item.name,
                        asset_key,
                    )
                    tile_store.write_raster_file(
                        item,
                        band_names,
                        UPath(local_fname),
                        time_range=item.geometry.time_range,
                    )

                logger.debug(
                    "EarthDaily done ingesting item %s asset %s",
                    item.name,
                    asset_key,
                )

    def is_raster_ready(self, layer_name: str, item: Item, bands: list[str]) -> bool:
        """Checks if this raster has been written to the store.

        Args:
            layer_name: the layer name or alias.
            item: the item.
            bands: the list of bands identifying which specific raster to read.

        Returns:
            whether there is a raster in the store matching the source, item, and
                bands.
        """
        # Always ready since we wrap accesses to EarthDaily.
        return True

    def get_raster_bands(self, layer_name: str, item: Item) -> list[list[str]]:
        """Get the sets of bands that have been stored for the specified item.

        Args:
            layer_name: the layer name or alias.
            item: the item.
        """
        if not isinstance(item, EarthDailyItem):
            raise TypeError(f"expected EarthDailyItem, got {type(item)}")
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

    def _apply_scale_offsets(
        self,
        array: npt.NDArray[Any],
        *,
        scale_offsets: list[dict[str, float | None]] | None,
        item_name: str,
        asset_key: str,
        time_range: tuple[datetime, datetime] | None = None,
        metadata: RasterMetadata | None = None,
    ) -> RasterArray:
        if not scale_offsets:
            return RasterArray(
                chw_array=array, time_range=time_range, metadata=metadata
            )

        def _to_float(value: Any, default: float) -> float:
            try:
                return float(value) if value is not None else default
            except (TypeError, ValueError):
                return default

        num_bands = array.shape[0]
        nodata_value: int | float | None
        if len(scale_offsets) == num_bands:
            scales = np.array(
                [_to_float(so.get("scale"), 1.0) for so in scale_offsets],
                dtype=np.float32,
            ).reshape(num_bands, 1, 1)
            offsets = np.array(
                [_to_float(so.get("offset"), 0.0) for so in scale_offsets],
                dtype=np.float32,
            ).reshape(num_bands, 1, 1)
            # For nodata, we don't use a default since a None nodata value is okay.
            # We still use _to_float for type checking but so["nodata"] should never be None.
            nd_vals = [
                _to_float(so["nodata"], 0.0)
                for so in scale_offsets
                if so.get("nodata") is not None
            ]
            nodata_value = unique_nodata_value(nd_vals) if nd_vals else None
        else:
            if len(scale_offsets) != 1:
                logger.debug(
                    "EarthDaily scale/offset band count mismatch for item %s asset %s: "
                    "%d metadata bands vs %d raster bands; using first entry for all bands",
                    item_name,
                    asset_key,
                    len(scale_offsets),
                    num_bands,
                )
            scale = _to_float(scale_offsets[0].get("scale"), 1.0)
            offset = _to_float(scale_offsets[0].get("offset"), 0.0)
            raw_nd = scale_offsets[0].get("nodata")
            nodata_value = float(raw_nd) if raw_nd is not None else None
            scales = np.full((num_bands, 1, 1), scale, dtype=np.float32)
            offsets = np.full((num_bands, 1, 1), offset, dtype=np.float32)

        if metadata is None:
            metadata = RasterMetadata(nodata_value=nodata_value)
        elif nodata_value is not None:
            metadata.nodata_value = nodata_value

        if np.all(scales == 1.0) and np.all(offsets == 0.0):
            return RasterArray(
                chw_array=array, time_range=time_range, metadata=metadata
            )

        array = array.astype(np.float32, copy=False)
        scaled = array * scales + offsets

        if nodata_value is not None:
            nodata_mask = nodata_eq(array, nodata_value)
            if nodata_mask.any():
                scaled[nodata_mask] = array[nodata_mask]

        return RasterArray(chw_array=scaled, time_range=time_range, metadata=metadata)

    def get_raster_metadata(
        self, layer_name: str, item: Item, bands: list[str]
    ) -> RasterMetadata:
        """Read nodata from the remote file header (cached per asset key)."""
        if not isinstance(item, EarthDailyItem):
            return RasterMetadata()
        asset_key = self._get_asset_by_band(bands)
        if asset_key not in item.asset_urls:
            return RasterMetadata()
        if asset_key not in self._nodata_cache:
            with rasterio.open(item.asset_urls[asset_key]) as src:
                self._nodata_cache[asset_key] = src.nodata

        nodata_value = self._nodata_cache[asset_key]
        return RasterMetadata(nodata_value=nodata_value)

    def get_raster_bounds(
        self, layer_name: str, item: Item, bands: list[str], projection: Projection
    ) -> PixelBounds:
        """Get the bounds of the raster in the specified projection.

        Args:
            layer_name: the layer name or alias.
            item: the item to check.
            bands: the list of bands identifying which specific raster to read. These
                bands must match the bands of a stored raster.
            projection: the projection to get the raster's bounds in.

        Returns:
            the bounds of the raster in the projection.
        """
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
        item: Item,
        bands: list[str],
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
    ) -> RasterArray:
        """Read raster data from the store.

        Args:
            layer_name: the layer name or alias.
            item: the item to read.
            bands: the list of bands identifying which specific raster to read. These
                bands must match the bands of a stored raster.
            projection: the projection to read in.
            bounds: the bounds to read.
            resampling: the resampling method to use in case reprojection is needed.

        Returns:
            the raster data
        """
        if not isinstance(item, EarthDailyItem):
            raise TypeError(f"expected EarthDailyItem, got {type(item)}")
        asset_key = self._get_asset_by_band(bands)
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
            src_nodata = src.nodata
            with rasterio.vrt.WarpedVRT(
                src,
                crs=projection.crs,
                transform=wanted_transform,
                width=bounds[2] - bounds[0],
                height=bounds[3] - bounds[1],
                resampling=resampling,
            ) as vrt:
                data = vrt.read()
        raster_metadata = None
        if src_nodata is not None:
            raster_metadata = RasterMetadata(nodata_value=src_nodata)
        return RasterArray(
            chw_array=data,
            time_range=item.geometry.time_range,
            metadata=raster_metadata,
        )

    def materialize(
        self,
        window: Window,
        item_groups: list[list[Item]],
        layer_name: str,
        layer_cfg: LayerConfig,
        group_time_ranges: list[tuple[datetime, datetime] | None] | None = None,
    ) -> None:
        """Materialize data for the window.

        Args:
            window: the window to materialize
            item_groups: the items from get_items
            layer_name: the name of this layer
            layer_cfg: the config of this layer
            group_time_ranges: optional request time range for each item group
        """
        RasterMaterializer().materialize(
            TileStoreWithLayer(self, layer_name),
            window,
            layer_name,
            layer_cfg,
            item_groups,
            group_time_ranges=group_time_ranges,
        )


class Sentinel2C1L2A(EarthDaily):
    """EarthDaily Sentinel-2 Collection 1 L2A (`sentinel-2-c1-l2a`) source.

    Applies per-asset scale/offset from STAC `raster:bands` by default (not
    processing-baseline harmonization). For Planetary Computer-compatible asset keys
    use `Sentinel2L2A`.
    """

    COLLECTION_NAME = "sentinel-2-c1-l2a"

    # EarthDaily Sentinel-2 asset keys to rslearn band names.
    # For spectral bands, we use Sentinel-2 band IDs as rslearn band names.
    ASSET_BANDS: dict[str, list[str]] = {
        "coastal": ["B01"],
        "blue": ["B02"],
        "green": ["B03"],
        "red": ["B04"],
        "rededge1": ["B05"],
        "rededge2": ["B06"],
        "rededge3": ["B07"],
        "nir": ["B08"],
        "nir08": ["B8A"],
        "nir09": ["B09"],
        "swir16": ["B11"],
        "swir22": ["B12"],
        # Derived products.
        "visual": ["R", "G", "B"],
        "scl": ["scl"],
        "aot": ["aot"],
        "wvp": ["wvp"],
    }

    def __init__(
        self,
        apply_scale_offset: bool = True,
        assets: list[str] | None = None,
        cloud_cover_max: float | None = None,
        search_max_items: int = 500,
        sort_items_by: Literal["cloud_cover", "datetime"] | None = "cloud_cover",
        query: dict[str, Any] | None = None,
        sort_by: str | None = None,
        sort_ascending: bool = True,
        timeout: timedelta = timedelta(seconds=10),
        cache_dir: str | None = None,
        max_retries: int = 3,
        retry_backoff_factor: float = 5.0,
        context: DataSourceContext = DataSourceContext(),
    ) -> None:
        """Initialize an EarthDaily Sentinel-2 C1 L2A data source.

        Args:
            assets: optional list of EarthDaily Sentinel-2 asset keys (e.g. ["red",
                "green", "blue", "nir", "swir16"]). If omitted and a LayerConfig is
                provided via context, assets are inferred from that layer's band sets.
            cloud_cover_max: max cloud cover (%) applied in searches. If set, injects
                an `eo:cloud_cover` upper bound into the STAC query.
            search_max_items: max number of STAC items to fetch per window before
                rslearn's grouping/matching logic runs.
            sort_items_by: optional ordering applied before grouping; useful when
                using `SpaceMode.COMPOSITE` with `CompositingMethod.FIRST_VALID`.
            query: optional STAC API `query` filter passed to searches.
            sort_by: optional STAC item property to sort by before grouping/matching.
                If set, it takes precedence over sort_items_by.
            sort_ascending: whether to sort ascending when sort_by is set.
            timeout: timeout for HTTP asset downloads (when ingesting).
            cache_dir: optional directory to cache item metadata by item id.
            max_retries: max retries for EarthDaily API client (search/get item).
            retry_backoff_factor: backoff factor for EarthDaily API client retries.
            context: rslearn data source context.
            apply_scale_offset: apply per-asset scale/offset metadata from STAC
                `raster:bands` during read/materialization (defaults to True).
                This decodes C1 COG storage values to physical values; it is not
                Sentinel-2 processing-baseline harmonization. Set to False to use the
                raw integer DN/sample values from the COG.
        """
        if apply_scale_offset and context.layer_config is not None:
            invalid_band_sets = [
                band_set
                for band_set in context.layer_config.band_sets
                if band_set.dtype != DType.FLOAT32
            ]
            if invalid_band_sets:
                invalid_str = ", ".join(
                    f"bands={band_set.bands} dtype={band_set.dtype.value}"
                    for band_set in invalid_band_sets
                )
                raise ValueError(
                    "EarthDaily Sentinel-2 with apply_scale_offset=True requires "
                    "band_sets dtype=float32 because scale/offset outputs physical "
                    f"float values. Invalid band sets: {invalid_str}"
                )

        asset_bands: dict[str, list[str]]
        if context.layer_config is not None and assets is None:
            asset_bands = {}
            wanted_bands: set[str] = set()
            for band_set in context.layer_config.band_sets:
                wanted_bands.update(band_set.bands)
            for asset_key, band_names in self.ASSET_BANDS.items():
                if wanted_bands.intersection(set(band_names)):
                    asset_bands[asset_key] = band_names
        elif assets is not None:
            unknown_assets = [
                asset_key for asset_key in assets if asset_key not in self.ASSET_BANDS
            ]
            if unknown_assets:
                raise ValueError(
                    f"unknown EarthDaily Sentinel-2 assets {unknown_assets}; "
                    f"supported assets are {sorted(self.ASSET_BANDS.keys())}"
                )
            asset_bands = {
                asset_key: self.ASSET_BANDS[asset_key] for asset_key in assets
            }
        else:
            asset_bands = dict(self.ASSET_BANDS)

        super().__init__(
            collection_name=self.COLLECTION_NAME,
            asset_bands=asset_bands,
            query=query,
            sort_by=sort_by,
            sort_ascending=sort_ascending,
            cloud_cover_max=cloud_cover_max,
            search_max_items=search_max_items,
            sort_items_by=sort_items_by,
            timeout=timeout,
            skip_items_missing_assets=True,
            cache_dir=cache_dir,
            max_retries=max_retries,
            retry_backoff_factor=retry_backoff_factor,
            read_scale_offsets=apply_scale_offset,
            context=context,
        )

        self.apply_scale_offset = apply_scale_offset

    def read_raster(
        self,
        layer_name: str,
        item: Item,
        bands: list[str],
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
    ) -> RasterArray:
        """Read raster data from the store.

        Applies per-asset scale/offset metadata when apply_scale_offset=True.
        """
        raster = super().read_raster(
            layer_name, item, bands, projection, bounds, resampling=resampling
        )
        if not self.apply_scale_offset:
            return raster

        if not isinstance(item, EarthDailyItem):
            raise TypeError(f"expected EarthDailyItem, got {type(item)}")
        asset_key = self._get_asset_by_band(bands)
        return self._apply_scale_offsets(
            raster.get_chw_array(),
            scale_offsets=item.asset_scale_offsets.get(asset_key),
            item_name=item.name,
            asset_key=asset_key,
            time_range=item.geometry.time_range,
            metadata=raster.metadata,
        )

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[EarthDailyItem],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest Sentinel-2 items into the provided tile store.

        Applies per-asset scale/offset metadata (when present).
        """
        for item in items:
            with tempfile.TemporaryDirectory() as tmp_dir:
                for asset_key, band_names in self.asset_bands.items():
                    asset_url = item.asset_urls.get(asset_key)
                    if asset_url is None:
                        continue
                    if tile_store.is_raster_ready(item, band_names):
                        continue

                    local_fname = self._download_asset_to_tmp(
                        asset_url, tmp_dir, asset_key, item.name
                    )

                    if not self.apply_scale_offset:
                        tile_store.write_raster_file(
                            item,
                            band_names,
                            UPath(local_fname),
                            time_range=item.geometry.time_range,
                        )
                        continue

                    with rasterio.open(local_fname) as src:
                        array = src.read()
                        src_nodata = src.nodata
                        src_metadata = (
                            RasterMetadata(nodata_value=src_nodata)
                            if src_nodata is not None
                            else None
                        )
                        raster = self._apply_scale_offsets(
                            array,
                            scale_offsets=item.asset_scale_offsets.get(asset_key),
                            item_name=item.name,
                            asset_key=asset_key,
                            time_range=item.geometry.time_range,
                            metadata=src_metadata,
                        )

                        projection, bounds = get_raster_projection_and_bounds(src)
                    tile_store.write_raster(
                        item,
                        band_names,
                        projection,
                        bounds,
                        raster,
                    )


class Sentinel2L2A(EarthDaily):
    """EarthDaily Sentinel-2 `sentinel-2-l2a` source with Planetary Computer-compatible asset keys.

    For EarthDaily Collection 1 (`sentinel-2-c1-l2a`), use `Sentinel2C1L2A`.
    """

    COLLECTION_NAME = "sentinel-2-l2a"

    ASSET_BANDS = {
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
        "SCL": ["SCL"],
        "visual": ["R", "G", "B"],
    }
    NON_REFLECTANCE_ASSETS = frozenset({"SCL", "visual"})
    PROCESSING_BASELINE_PATTERN = re.compile(r"(?:^|_)N(?P<baseline>\d{4})(?:_|$)")
    HARMONIZE_PROCESSING_BASELINE = 400
    HARMONIZE_CUTOFF = datetime(2022, 1, 25)
    HARMONIZE_OFFSET = 1000

    def __init__(
        self,
        harmonize: bool = False,
        assets: list[str] | None = None,
        cloud_cover_max: float | None = None,
        search_max_items: int = 500,
        sort_items_by: Literal["cloud_cover", "datetime"] | None = "cloud_cover",
        query: dict[str, Any] | None = None,
        sort_by: str | None = None,
        sort_ascending: bool = True,
        timeout: timedelta = timedelta(seconds=10),
        cache_dir: str | None = None,
        max_retries: int = 3,
        retry_backoff_factor: float = 5.0,
        context: DataSourceContext = DataSourceContext(),
    ) -> None:
        """Initialize a new EarthDaily Sentinel2L2A data source.

        Args:
            harmonize: apply processing-baseline harmonization to reflectance values.
            assets: optional list of asset keys to ingest. If omitted and a LayerConfig
                is provided via context, assets are inferred from that layer's band sets.
            cloud_cover_max: max cloud cover (%) applied in searches. If set, injects
                an ``eo:cloud_cover`` upper bound into the STAC query.
            search_max_items: max number of STAC items to fetch per window.
            sort_items_by: optional ordering applied before grouping.
            query: optional STAC API ``query`` filter passed to searches.
            sort_by: optional STAC item property to sort by.
            sort_ascending: whether to sort ascending when sort_by is set.
            timeout: timeout for HTTP asset downloads.
            cache_dir: optional directory to cache item metadata by item id.
            max_retries: max retries for EarthDaily API client.
            retry_backoff_factor: backoff factor for EarthDaily API client retries.
            context: rslearn data source context.
        """
        self.harmonize = harmonize
        self._harmonize_callback_cache: dict[
            str, Callable[[npt.NDArray[Any]], npt.NDArray[Any]] | None
        ] = {}

        if context.layer_config is not None:
            asset_bands: dict[str, list[str]] = {}
            for asset_key, band_names in self.ASSET_BANDS.items():
                for band_set in context.layer_config.band_sets:
                    if set(band_set.bands).intersection(set(band_names)):
                        asset_bands[asset_key] = band_names
                        break
        elif assets is not None:
            unknown_assets = [
                asset_key for asset_key in assets if asset_key not in self.ASSET_BANDS
            ]
            if unknown_assets:
                raise ValueError(
                    f"unknown EarthDaily Sentinel-2 L2A assets {unknown_assets}; "
                    f"supported assets are {sorted(self.ASSET_BANDS.keys())}"
                )
            asset_bands = {
                asset_key: self.ASSET_BANDS[asset_key] for asset_key in assets
            }
        else:
            asset_bands = dict(self.ASSET_BANDS)

        super().__init__(
            collection_name=self.COLLECTION_NAME,
            asset_bands=asset_bands,
            query=query,
            sort_by=sort_by,
            sort_ascending=sort_ascending,
            cloud_cover_max=cloud_cover_max,
            search_max_items=search_max_items,
            sort_items_by=sort_items_by,
            timeout=timeout,
            skip_items_missing_assets=True,
            cache_dir=cache_dir,
            max_retries=max_retries,
            retry_backoff_factor=retry_backoff_factor,
            read_scale_offsets=False,
            context=context,
        )

    def _normalize_dt(self, dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt
        return dt.astimezone(UTC).replace(tzinfo=None)

    def _resolve_metadata_url(self, item: EarthDailyItem) -> str:
        for asset_key in self.METADATA_ASSET_KEYS:
            if asset_key in item.asset_urls:
                return item.asset_urls[asset_key]
        raise KeyError(
            "missing metadata asset URL (expected one of: "
            "product_metadata, granule_metadata)"
        )

    def _get_product_xml(self, item: EarthDailyItem) -> ET.Element:
        asset_url = self._resolve_metadata_url(item)
        response = requests.get(asset_url, timeout=self.timeout.total_seconds())
        response.raise_for_status()
        return ET.fromstring(response.content)

    def _get_processing_baseline(self, item_name: str) -> int | None:
        match = self.PROCESSING_BASELINE_PATTERN.search(item_name)
        if match is None:
            return None
        return int(match.group("baseline"))

    def _fallback_harmonize_callback(
        self, item: EarthDailyItem
    ) -> Callable[[npt.NDArray[Any]], npt.NDArray[Any]] | None:
        processing_baseline = None
        if item.product_id is not None:
            processing_baseline = self._get_processing_baseline(item.product_id)
        if processing_baseline is not None:
            if processing_baseline < self.HARMONIZE_PROCESSING_BASELINE:
                return None
        else:
            if item.geometry.time_range is None:
                return None
            start_time = self._normalize_dt(item.geometry.time_range[0])
            if start_time < self.HARMONIZE_CUTOFF:
                return None

        offset = self.HARMONIZE_OFFSET

        def callback(array: npt.NDArray[Any]) -> npt.NDArray[Any]:
            if array.dtype != np.uint16:
                return array
            was_valid = array > 0
            result = np.clip(array, offset, None) - offset
            result[(result == 0) & was_valid] = 1
            return result

        return callback

    def _get_harmonize_callback_for_item(
        self, item: EarthDailyItem, asset_key: str
    ) -> Callable[[npt.NDArray[Any]], npt.NDArray[Any]] | None:
        if not self.harmonize or asset_key in self.NON_REFLECTANCE_ASSETS:
            return None
        if item.name in self._harmonize_callback_cache:
            return self._harmonize_callback_cache[item.name]

        callback = get_harmonize_callback(self._get_product_xml(item))
        if callback is None:
            callback = self._fallback_harmonize_callback(item)
            if callback is not None:
                logger.debug(
                    "EarthDaily Sentinel2L2A using product-id/item-id/date-based harmonization fallback for %s",
                    item.name,
                )

        self._harmonize_callback_cache[item.name] = callback
        return callback

    def read_raster(
        self,
        layer_name: str,
        item: Item,
        bands: list[str],
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
    ) -> RasterArray:
        """Read raster data for an item and apply harmonization when configured."""
        if not isinstance(item, EarthDailyItem):
            raise TypeError(f"expected EarthDailyItem, got {type(item)}")
        raster = super().read_raster(
            layer_name, item, bands, projection, bounds, resampling=resampling
        )

        asset_key = self._get_asset_by_band(bands)
        harmonize_callback = self._get_harmonize_callback_for_item(item, asset_key)
        if harmonize_callback is None:
            return raster

        return RasterArray(
            chw_array=harmonize_callback(raster.get_chw_array()),
            time_range=item.geometry.time_range,
            metadata=raster.metadata,
        )

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[EarthDailyItem],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest Sentinel-2 L2A items with optional harmonization."""
        for item in items:
            for asset_key, band_names in self.asset_bands.items():
                asset_url = item.asset_urls.get(asset_key)
                if asset_url is None:
                    continue
                if tile_store.is_raster_ready(item, band_names):
                    continue

                with tempfile.TemporaryDirectory() as tmp_dir:
                    local_fname = self._download_asset_to_tmp(
                        asset_url, tmp_dir, asset_key, item.name
                    )

                    harmonize_callback = self._get_harmonize_callback_for_item(
                        item, asset_key
                    )
                    if harmonize_callback is None:
                        tile_store.write_raster_file(
                            item,
                            band_names,
                            UPath(local_fname),
                            time_range=item.geometry.time_range,
                        )
                        continue

                    with rasterio.open(local_fname) as src:
                        array = src.read()
                        src_nodata = src.nodata
                        projection, bounds = get_raster_projection_and_bounds(src)
                    array = harmonize_callback(array)
                    raster_metadata = RasterMetadata(nodata_value=src_nodata)
                    tile_store.write_raster(
                        item,
                        band_names,
                        projection,
                        bounds,
                        RasterArray(
                            chw_array=array,
                            time_range=item.geometry.time_range,
                            metadata=raster_metadata,
                        ),
                    )


class Sentinel2EDACloudMask(EarthDaily):
    """EarthDaily Sentinel-2 EDA cloud mask (`sentinel-2-eda-cloud-mask`) source.

    The `cloud-mask` STAC asset contains two thematic bands. They are exposed as
    `cloud-mask` (0 nodata, 1 clear, 2 cloud, 3 cloud shadow, 4 thin cloud) and
    `cirrus-mask` (0 nodata, 1 non-cirrus, 2 cirrus).
    """

    COLLECTION_NAME = "sentinel-2-eda-cloud-mask"
    ASSET_KEY = "cloud-mask"
    DEFAULT_BAND_NAMES = ["cloud-mask", "cirrus-mask"]

    ASSET_BANDS = {
        ASSET_KEY: DEFAULT_BAND_NAMES,
    }

    def __init__(
        self,
        assets: list[str] | None = None,
        band_names: list[str] | None = None,
        cloud_cover_max: float | None = None,
        search_max_items: int = 500,
        sort_items_by: Literal["cloud_cover", "datetime"] | None = "cloud_cover",
        query: dict[str, Any] | None = None,
        sort_by: str | None = None,
        sort_ascending: bool = True,
        timeout: timedelta = timedelta(seconds=10),
        cache_dir: str | None = None,
        max_retries: int = 3,
        retry_backoff_factor: float = 5.0,
        context: DataSourceContext = DataSourceContext(),
    ) -> None:
        """Initialize an EarthDaily Sentinel-2 EDA cloud mask data source.

        Args:
            assets: optional list of EarthDaily cloud-mask STAC asset keys. The only
                supported asset is `cloud-mask`. If omitted and a LayerConfig is
                provided via context, assets are inferred from that layer's band sets.
            band_names: ordered rslearn names for bands to expose from the
                `cloud-mask` GeoTIFF asset. The default is `["cloud-mask",
                "cirrus-mask"]`. The first name maps to GeoTIFF band 1, the second
                name maps to GeoTIFF band 2.
            cloud_cover_max: max cloud cover (%) applied in searches. If set, injects
                an `eo:cloud_cover` upper bound into the STAC query.
            search_max_items: max number of STAC items to fetch per window before
                rslearn's grouping/matching logic runs.
            sort_items_by: optional ordering applied before grouping; useful when
                using `SpaceMode.COMPOSITE` with `CompositingMethod.FIRST_VALID`.
            query: optional STAC API `query` filter passed to searches.
            sort_by: optional STAC item property to sort by before grouping/matching.
                If set, it takes precedence over sort_items_by.
            sort_ascending: whether to sort ascending when sort_by is set.
            timeout: timeout for HTTP asset downloads (when ingesting).
            cache_dir: optional directory to cache item metadata by item id.
            max_retries: max retries for EarthDaily API client (search/get item).
            retry_backoff_factor: backoff factor for EarthDaily API client retries.
            context: rslearn data source context.
        """
        if band_names is not None:
            if not band_names:
                raise ValueError("band_names must contain at least one band")
            if len(set(band_names)) != len(band_names):
                raise ValueError(f"band_names must be unique, got {band_names}")

        if assets is not None:
            unknown_assets = [
                asset_key for asset_key in assets if asset_key != self.ASSET_KEY
            ]
            if unknown_assets:
                raise ValueError(
                    f"unknown EarthDaily Sentinel-2 EDA cloud mask assets "
                    f"{unknown_assets}; supported assets are [{self.ASSET_KEY!r}]"
                )

        if context.layer_config is not None and assets is None and band_names is None:
            supported_bands = set(self.DEFAULT_BAND_NAMES)
            wanted_bands: list[str] = []
            for band_set in context.layer_config.band_sets:
                for band in band_set.bands:
                    if band not in supported_bands:
                        continue
                    if band in wanted_bands:
                        continue
                    wanted_bands.append(band)
            source_ordered_bands = [
                band for band in self.DEFAULT_BAND_NAMES if band in wanted_bands
            ]
            asset_bands = (
                {self.ASSET_KEY: source_ordered_bands} if source_ordered_bands else {}
            )
        else:
            if assets == []:
                asset_bands = {}
            else:
                asset_bands = {
                    self.ASSET_KEY: list(band_names or self.DEFAULT_BAND_NAMES)
                }

        super().__init__(
            collection_name=self.COLLECTION_NAME,
            asset_bands=asset_bands,
            query=query,
            sort_by=sort_by,
            sort_ascending=sort_ascending,
            cloud_cover_max=cloud_cover_max,
            search_max_items=search_max_items,
            sort_items_by=sort_items_by,
            timeout=timeout,
            skip_items_missing_assets=True,
            cache_dir=cache_dir,
            max_retries=max_retries,
            retry_backoff_factor=retry_backoff_factor,
            read_scale_offsets=False,
            context=context,
        )

        self.cloud_mask_source_band_names = list(band_names or self.DEFAULT_BAND_NAMES)

    def _get_asset_by_band(self, bands: list[str]) -> str:
        """Get the cloud-mask asset for exact or subset band requests."""
        configured_bands = self.asset_bands.get(self.ASSET_KEY)
        if configured_bands is not None and all(
            band in configured_bands for band in bands
        ):
            return self.ASSET_KEY
        return super()._get_asset_by_band(bands)

    def _source_band_indexes_for_bands(self, bands: list[str]) -> list[int]:
        """Return 0-based source GeoTIFF band indexes for configured rslearn bands."""
        indexes: list[int] = []
        for band in bands:
            try:
                indexes.append(self.cloud_mask_source_band_names.index(band))
            except ValueError as e:
                raise ValueError(
                    f"band {band!r} is not configured for EarthDaily "
                    f"{self.ASSET_KEY} asset bands {self.cloud_mask_source_band_names}"
                ) from e
        return indexes

    def _validate_asset_band_count(
        self,
        item: EarthDailyItem,
        asset_key: str,
        raster_band_count: int,
        source_band_indexes: list[int],
    ) -> None:
        required_count = max(source_band_indexes, default=-1) + 1
        if raster_band_count < required_count:
            raise ValueError(
                f"EarthDaily item {item.name} asset {asset_key} has "
                f"{raster_band_count} raster bands but configured bands "
                f"{self.asset_bands[asset_key]} require source band {required_count}"
            )

    def read_raster(
        self,
        layer_name: str,
        item: Item,
        bands: list[str],
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
    ) -> RasterArray:
        """Read configured bands from the cloud-mask asset."""
        if not isinstance(item, EarthDailyItem):
            raise TypeError(f"expected EarthDailyItem, got {type(item)}")
        raster = super().read_raster(
            layer_name, item, bands, projection, bounds, resampling=resampling
        )
        asset_key = self._get_asset_by_band(bands)
        source_band_indexes = self._source_band_indexes_for_bands(bands)
        self._validate_asset_band_count(
            item, asset_key, raster.array.shape[0], source_band_indexes
        )
        return RasterArray(
            array=raster.array[source_band_indexes, :, :, :],
            timestamps=raster.timestamps,
            metadata=raster.metadata,
        )

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[EarthDailyItem],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest configured bands from the cloud-mask asset."""
        for item in items:
            band_names = self.asset_bands.get(self.ASSET_KEY)
            if band_names is None:
                continue
            asset_url = item.asset_urls.get(self.ASSET_KEY)
            if asset_url is None:
                continue
            if tile_store.is_raster_ready(item, band_names):
                continue

            with tempfile.TemporaryDirectory() as tmp_dir:
                local_fname = self._download_asset_to_tmp(
                    asset_url, tmp_dir, self.ASSET_KEY, item.name
                )
                with rasterio.open(local_fname) as src:
                    source_band_indexes = self._source_band_indexes_for_bands(
                        band_names
                    )
                    self._validate_asset_band_count(
                        item, self.ASSET_KEY, src.count, source_band_indexes
                    )
                    indexes = [index + 1 for index in source_band_indexes]
                    array = src.read(indexes=indexes)
                    src_nodata = src.nodata
                    projection, bounds = get_raster_projection_and_bounds(src)

                tile_store.write_raster(
                    item,
                    band_names,
                    projection,
                    bounds,
                    RasterArray(
                        chw_array=array,
                        time_range=item.geometry.time_range,
                        metadata=RasterMetadata(nodata_value=src_nodata),
                    ),
                )


class Biophysical(EarthDaily):
    """Biophysical variables on EarthDaily platform (EDAgro layers).

    Supported variables (each maps to its own EarthDaily STAC collection + asset key):
    - `lai` → collection `lai-layer-edagro`, asset `lai`
    - `fapar` → collection `fapar-layer-edagro`, asset `fapar`
    - `fcover` → collection `fcover-layer-edagro`, asset `fcover`
    """

    VARIABLES: dict[str, dict[str, str]] = {
        "lai": {"collection": "lai-layer-edagro", "asset": "lai"},
        "fapar": {"collection": "fapar-layer-edagro", "asset": "fapar"},
        "fcover": {"collection": "fcover-layer-edagro", "asset": "fcover"},
    }

    def __init__(
        self,
        variable: Literal["lai", "fapar", "fcover"],
        *,
        query: dict[str, Any] | None = None,
        sort_by: str | None = None,
        sort_ascending: bool = True,
        timeout: timedelta = timedelta(seconds=10),
        cache_dir: str | None = None,
        max_retries: int = 3,
        retry_backoff_factor: float = 5.0,
        context: DataSourceContext = DataSourceContext(),
    ) -> None:
        """Initialize an EarthDaily biophysical variable data source.

        Args:
            variable: one of `lai`, `fapar`, or `fcover`.
            query: optional STAC API `query` filter passed to searches.
            sort_by: optional STAC item property to sort by before grouping/matching.
            sort_ascending: whether to sort ascending when sort_by is set.
            timeout: timeout for HTTP asset downloads (when ingesting).
            cache_dir: optional directory to cache item metadata by item id.
            max_retries: max retries for EarthDaily API client (search/get item).
            retry_backoff_factor: backoff factor for EarthDaily API client retries.
            context: rslearn data source context.
        """
        if variable not in self.VARIABLES:
            raise ValueError(
                f"unknown biophysical variable {variable}; supported variables are "
                f"{sorted(self.VARIABLES.keys())}"
            )
        cfg = self.VARIABLES[variable]
        self.variable = variable

        asset_key = cfg["asset"]
        super().__init__(
            collection_name=cfg["collection"],
            asset_bands={asset_key: [asset_key]},
            query=query,
            sort_by=sort_by,
            sort_ascending=sort_ascending,
            timeout=timeout,
            skip_items_missing_assets=True,
            cache_dir=cache_dir,
            max_retries=max_retries,
            retry_backoff_factor=retry_backoff_factor,
            context=context,
        )
