"""Data source for OpenLandMap-SoilDB STAC-hosted rasters.

SoilDB collections are published as a static STAC catalog (collection.json + a single
item JSON per collection) with many GeoTIFF/COG assets per item (e.g., different
depths, resolutions, and summary statistics).
"""

import json
import os
import tempfile
from datetime import timedelta
from typing import Any
from urllib.parse import urljoin

import requests
import shapely
from upath import UPath

from rslearn.config import QueryConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.data_source import DataSourceContext
from rslearn.data_sources.direct_materialize_data_source import (
    DirectMaterializeDataSource,
)
from rslearn.data_sources.stac import SourceItem
from rslearn.data_sources.utils import match_candidate_items_to_window
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils import STGeometry
from rslearn.utils.fsspec import join_upath, open_atomic

logger = get_logger(__name__)


SOILDB_COLLECTIONS: dict[str, dict[str, str]] = {
    "bd.core_iso.11272.2017.g.cm3": {
        "title": "OpenLandMap-soildb: Bulk density fine earth [kg/m3]",
    },
    "oc_iso.10694.1995.wpml": {
        "title": "OpenLandMap-soildb: Soil organic carbon [g/kg]",
    },
    "oc_iso.10694.1995.mg.cm3": {
        "title": "OpenLandMap-soildb: Soil organic carbon density [kg/m3]",
    },
    "ph.h2o_iso.10390.2021.index": {
        "title": "OpenLandMap-soildb: Soil pH in H2O",
    },
    "clay.tot_iso.11277.2020.wpct": {
        "title": "OpenLandMap-soildb: Soil texture fraction clay [%]",
    },
    "sand.tot_iso.11277.2020.wpct": {
        "title": "OpenLandMap-soildb: Soil texture fraction sand [%]",
    },
    "silt.tot_iso.11277.2020.wpct": {
        "title": "OpenLandMap-soildb: Soil texture fraction silt [%]",
    },
    "soil.types_ensemble_probabilities": {
        "title": "OpenLandMap-soildb: Soil type probability",
    },
}

# Default STAC asset keys to use when asset_key is not provided.
#
# For most SoilDB collections, the STAC item exposes a consistent set of assets and
# we default to the mean, 30m, 0–30cm depth GeoTIFF asset.
#
# For other collections (including `soil.types_ensemble_probabilities`), users must
# specify asset_key explicitly.
SOILDB_DEFAULT_ASSET_KEY_CANDIDATES: dict[str, list[str]] = {
    "bd.core_iso.11272.2017.g.cm3": ["bd.core_iso.11272.2017.g.cm3_m_30m_b0cm..30cm"],
    "oc_iso.10694.1995.wpml": ["oc_iso.10694.1995.wpml_m_30m_b0cm..30cm"],
    "oc_iso.10694.1995.mg.cm3": ["oc_iso.10694.1995.mg.cm3_m_30m_b0cm..30cm"],
    "ph.h2o_iso.10390.2021.index": ["ph.h2o_iso.10390.2021.index_m_30m_b0cm..30cm"],
    "clay.tot_iso.11277.2020.wpct": ["clay.tot_iso.11277.2020.wpct_m_30m_b0cm..30cm"],
    "sand.tot_iso.11277.2020.wpct": ["sand.tot_iso.11277.2020.wpct_m_30m_b0cm..30cm"],
    "silt.tot_iso.11277.2020.wpct": ["silt.tot_iso.11277.2020.wpct_m_30m_b0cm..30cm"],
}


class SoilDB(DirectMaterializeDataSource[SourceItem]):
    """Read SoilDB rasters from the OpenLandMap static STAC catalog."""

    DEFAULT_CATALOG_URL = (
        "https://s3.eu-central-1.wasabisys.com/stac/openlandmap/catalog.json"
    )
    AUTO_ASSET_KEY = "__auto__"

    def __init__(
        self,
        collection_id: str,
        asset_key: str | None = None,
        band_name: str = "value",
        catalog_url: str = DEFAULT_CATALOG_URL,
        collection_url: str | None = None,
        validate_collection_id: bool = False,
        timeout: timedelta = timedelta(seconds=30),
        cache_dir: str | None = None,
        context: DataSourceContext = DataSourceContext(),
    ) -> None:
        """Initialize a new SoilDB data source.

        Args:
            collection_id: SoilDB STAC collection id (e.g. "clay.tot_iso.11277.2020.wpct").
            asset_key: STAC asset key to read. If not set, chooses a default
                GeoTIFF/COG asset for collections that define one (typically mean,
                30m, 0–30cm). Otherwise this must be set explicitly.
            band_name: band name to use if the layer config is missing from context.
            catalog_url: OpenLandMap static STAC catalog.json URL.
            collection_url: optional override for the collection.json URL.
            validate_collection_id: if True, require collection_id to be one of the
                known SoilDB collections listed in this module.
            timeout: timeout for HTTP requests.
            cache_dir: optional directory to cache the resolved SourceItem JSON.
            context: the data source context.
        """
        self.collection_id = collection_id
        self.catalog_url = catalog_url
        self.collection_url = collection_url
        self.timeout = timeout
        self._explicit_asset_key = asset_key

        if validate_collection_id and collection_id not in SOILDB_COLLECTIONS:
            raise ValueError(
                f"unknown SoilDB collection_id {collection_id!r}; known={sorted(SOILDB_COLLECTIONS.keys())}"
            )
        if collection_id not in SOILDB_COLLECTIONS:
            logger.debug(
                "SoilDB collection_id %s not in SOILDB_COLLECTIONS registry",
                collection_id,
            )

        if context.layer_config is not None:
            if len(context.layer_config.band_sets) != 1:
                raise ValueError("expected a single band set")
            if len(context.layer_config.band_sets[0].bands) != 1:
                raise ValueError("expected band set to have a single band")
            band_name = context.layer_config.band_sets[0].bands[0]

        # Initialize with a single-band mapping under a stable internal asset key.
        super().__init__(asset_bands={self.AUTO_ASSET_KEY: [band_name]})

        # Cache resolved item (optional).
        if cache_dir is not None:
            if context.ds_path is not None:
                self.cache_dir = join_upath(context.ds_path, cache_dir)
            else:
                self.cache_dir = UPath(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

        self._item: SourceItem | None = None

    # --- DirectMaterializeDataSource implementation ---

    def get_asset_url(self, item_name: str, asset_key: str) -> str:
        """Get the URL to read the requested asset from."""
        item = self.get_item_by_name(item_name)
        if asset_key not in item.asset_urls:
            raise KeyError(
                f"asset_key {asset_key!r} not available for item {item_name!r}; "
                f"available={sorted(item.asset_urls.keys())}"
            )
        return item.asset_urls[asset_key]

    # --- DataSource implementation ---

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[SourceItem]]]:
        """Get the SoilDB item for each requested window geometry."""
        item = self._get_or_load_item()
        groups: list[list[list[SourceItem]]] = []
        for geometry in geometries:
            cur_groups = match_candidate_items_to_window(geometry, [item], query_config)
            groups.append(cur_groups)
        return groups

    def deserialize_item(self, serialized_item: dict) -> SourceItem:
        """Deserialize a previously-serialized SoilDB SourceItem."""
        return SourceItem.deserialize(serialized_item)

    def get_item_by_name(self, name: str) -> SourceItem:
        """Get the single STAC item exposed by this SoilDB collection."""
        item = self._get_or_load_item()
        if name != item.name:
            raise ValueError(
                f"unknown item {name!r}; SoilDB collection {self.collection_id!r} exposes a single item {item.name!r}"
            )
        return item

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[SourceItem],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Download the configured GeoTIFF asset and write it into the tile store."""
        for item in items:
            for asset_key, band_names in self.asset_bands.items():
                if tile_store.is_raster_ready(item.name, band_names):
                    continue

                asset_url = self.get_asset_url(item.name, asset_key)

                with tempfile.TemporaryDirectory() as tmp_dir:
                    local_fname = os.path.join(tmp_dir, f"{asset_key}.tif")
                    logger.debug(
                        "SoilDB download item %s asset %s to %s",
                        item.name,
                        asset_key,
                        local_fname,
                    )
                    with requests.get(
                        asset_url, stream=True, timeout=self.timeout.total_seconds()
                    ) as r:
                        r.raise_for_status()
                        with open(local_fname, "wb") as f:
                            for chunk in r.iter_content(chunk_size=1024 * 1024):
                                f.write(chunk)

                    tile_store.write_raster_file(
                        item.name,
                        band_names,
                        UPath(local_fname),
                        time_range=item.geometry.time_range,
                    )

    # --- STAC loading helpers ---

    def _cache_path(self) -> UPath | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{self.collection_id}.json"

    def _get_or_load_item(self) -> SourceItem:
        if self._item is not None:
            return self._item

        cache_path = self._cache_path()
        if cache_path is not None and cache_path.exists():
            with cache_path.open() as f:
                payload = json.load(f)
            self._item = SourceItem.deserialize(payload["item"])
            return self._item

        item_url, item_dict = self._load_stac_item_dict()
        item = self._stac_item_dict_to_item(item_url, item_dict)

        if cache_path is not None:
            with open_atomic(cache_path, "w") as f:
                json.dump(
                    {
                        "collection_id": self.collection_id,
                        "item": item.serialize(),
                    },
                    f,
                )

        self._item = item
        return item

    def _collection_url_from_catalog(self) -> str:
        return urljoin(self.catalog_url, f"./{self.collection_id}/collection.json")

    def _load_stac_item_dict(self) -> tuple[str, dict[str, Any]]:
        collection_url = self.collection_url or self._collection_url_from_catalog()
        logger.debug("SoilDB loading collection.json from %s", collection_url)
        coll = self._fetch_json(collection_url)

        item_link = None
        for link in coll.get("links", []):
            if link.get("rel") != "item":
                continue
            item_link = link.get("href")
            break
        if not item_link:
            raise ValueError(
                f"collection {self.collection_id!r} has no rel='item' link at {collection_url!r}"
            )

        item_url = urljoin(collection_url, item_link)
        logger.debug("SoilDB loading item from %s", item_url)
        return item_url, self._fetch_json(item_url)

    def _fetch_json(self, url: str) -> dict[str, Any]:
        resp = requests.get(url, timeout=self.timeout.total_seconds())
        resp.raise_for_status()
        return resp.json()

    def _stac_item_dict_to_item(
        self, item_url: str, item_dict: dict[str, Any]
    ) -> SourceItem:
        if item_dict.get("geometry") is not None:
            shp = shapely.geometry.shape(item_dict["geometry"])
        elif item_dict.get("bbox") is not None and len(item_dict["bbox"]) == 4:
            shp = shapely.box(*item_dict["bbox"])
        else:
            raise ValueError("STAC item missing both geometry and bbox")

        # SoilDB is a static product for our purposes; ignore item timestamps so it
        # matches any window.
        geom = STGeometry(WGS84_PROJECTION, shp, None)

        assets: dict[str, Any] = item_dict.get("assets", {}) or {}
        if not assets:
            raise ValueError(f"STAC item {item_dict.get('id')!r} has no assets")

        if self._explicit_asset_key is not None:
            stac_asset_key = self._explicit_asset_key
        else:
            stac_asset_key = self._pick_stac_asset_key(assets)

        if stac_asset_key not in assets:
            raise ValueError(
                f"asset_key {stac_asset_key!r} not found in STAC item; "
                f"available_count={len(assets)}"
            )

        href = assets[stac_asset_key].get("href")
        if not isinstance(href, str) or not href:
            raise ValueError(f"asset {stac_asset_key!r} has no href")

        return SourceItem(
            name=item_dict["id"],
            geometry=geom,
            asset_urls={self.AUTO_ASSET_KEY: href},
            properties={
                "stac_collection": self.collection_id,
                "stac_item_url": item_url,
                "stac_asset_key": stac_asset_key,
            },
        )

    def _pick_stac_asset_key(self, assets: dict[str, Any]) -> str:
        """Pick a STAC asset key for this collection.

        This method implements SoilDB-specific selection logic:
        - Some collections have a per-collection default asset key (or short list).
        - Otherwise, an explicit asset key is required.
        """
        candidates = SOILDB_DEFAULT_ASSET_KEY_CANDIDATES.get(self.collection_id)
        if not candidates:
            raise ValueError(
                f"no default asset_key available for collection {self.collection_id!r}; "
                "set asset_key explicitly"
            )

        for k in candidates:
            if k in assets:
                return k

        raise ValueError(
            f"default asset_key candidates not found in STAC item for collection {self.collection_id!r}; "
            f"candidates={candidates!r}; set asset_key explicitly"
        )
