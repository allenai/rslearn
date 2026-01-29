"""Global SRTM void-filled elevation data from USGS, mirrored on Hugging Face by AI2.

This module provides:
1. A bulk download utility to fetch SRTM data from USGS EarthExplorer via the M2M API.
   This can be used to initialize a mirror of the data.
2. A data source that will pull from the AI2 Hugging Face mirror.

The SRTM dataset in USGS EarthExplorer is "srtm_v2" which contains void-filled elevation
data from the Shuttle Radar Topography Mission. The bulk download fetches the highest
resolution available: 1 arc-second (~30m) in the US and 3 arc-second (~90m) globally.

See https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-shuttle-radar-topography-mission-srtm
for details.
"""

import functools
import json
import math
import multiprocessing
import os
import re
import shutil
import tempfile
from datetime import timedelta
from typing import Any

import requests
import shapely
from upath import UPath

from rslearn.config import QueryConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import DataSource, DataSourceContext, Item
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.fsspec import join_upath, open_atomic
from rslearn.utils.geometry import STGeometry
from rslearn.utils.m2m_api import M2MAPIClient
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.retry_session import create_retry_session

logger = get_logger(__name__)

# SRTM dataset name in USGS EarthExplorer M2M API
SRTM_DATASET_NAME = "srtm_v2"

# Product names in order of preference (highest resolution first)
# 1 Arc-second is only available in the US, 3 Arc-second is global
SRTM_PRODUCT_NAMES = ["GeoTIFF 1 Arc-second", "GeoTIFF 3 Arc-second"]

# SRTM covers latitude -60 to 60
SRTM_LAT_MIN = -60
SRTM_LAT_MAX = 60

# Cache filename for scene list
SCENE_CACHE_FILENAME = "scenes.json"


class SRTM(DataSource):
    """Data source for SRTM elevation data from the AI2 Hugging Face mirror.

    The data is split into 1x1-degree tiles, with filenames like:
    N05/SRTM1N05W163V2.tif (1 arc-second, ~30m resolution)
    N05/SRTM3N05W163V2.tif (3 arc-second, ~90m resolution)

    SRTM1 (1 arc-second) is available for some regions (primarily US territories),
    while SRTM3 (3 arc-second) is available globally. By default, SRTM1 is preferred
    when available for higher resolution. Set always_use_3arcsecond=True to always
    use the lower resolution SRTM3 data for consistency.

    Items from this data source do not come with a time range. The band name will match
    that specified in the band set, which should have a single band (e.g. "dem").
    """

    BASE_URL = (
        "https://huggingface.co/datasets/allenai/srtm-global-void-filled/resolve/main/"
    )
    FILE_LIST_FILENAME = "file_list.json"
    FILENAME_SUFFIX = "V2.tif"

    def __init__(
        self,
        timeout: timedelta = timedelta(seconds=10),
        cache_dir: str | None = None,
        always_use_3arcsecond: bool = False,
        context: DataSourceContext = DataSourceContext(),
    ):
        """Initialize a new SRTM instance.

        Args:
            timeout: timeout for requests.
            cache_dir: optional directory to cache the file list.
            always_use_3arcsecond: if True, always use 3 arc-second (SRTM3) data even
                when 1 arc-second (SRTM1) is available. Defaults to False, which
                prefers SRTM1 for higher resolution when available.
            context: the data source context.
        """
        # Get band name from context if possible, falling back to "dem".
        if context.layer_config is not None:
            if len(context.layer_config.band_sets) != 1:
                raise ValueError("expected a single band set")
            if len(context.layer_config.band_sets[0].bands) != 1:
                raise ValueError("expected band set to have a single band")
            self.band_name = context.layer_config.band_sets[0].bands[0]
        else:
            self.band_name = "dem"

        self.timeout = timeout
        self.session = requests.session()
        self.always_use_3arcsecond = always_use_3arcsecond

        # Set the cache path if a cache_dir is provided.
        self.file_list_cache_path: UPath | None = None
        if cache_dir is not None:
            if context.ds_path is not None:
                cache_root = join_upath(context.ds_path, cache_dir)
            else:
                cache_root = UPath(cache_dir)
            cache_root.mkdir(parents=True, exist_ok=True)
            self.file_list_cache_path = join_upath(cache_root, self.FILE_LIST_FILENAME)

        self._basename_to_item, self._tile_to_item = self._load_file_index()

    def _load_file_index(
        self,
    ) -> tuple[dict[str, Item], dict[tuple[int, int], Item]]:
        """Load the file list and build indices for lookups."""
        file_list = self._load_file_list_json()
        if not isinstance(file_list, list):
            raise ValueError("expected file_list.json to be a list of filenames")

        basename_to_item: dict[str, Item] = {}
        tile_to_item: dict[tuple[int, int], Item] = {}

        for entry in file_list:
            if not isinstance(entry, str):
                raise ValueError(
                    "expected file_list.json to contain only string filenames"
                )
            basename = os.path.basename(entry)

            # Check if this is SRTM1 or SRTM3 based on filename prefix
            is_srtm1 = basename.startswith("SRTM1")

            # Skip SRTM1 files if always_use_3arcsecond is enabled
            if self.always_use_3arcsecond and is_srtm1:
                continue

            lat_min, lon_min = self._parse_tile_basename(basename)
            geometry = STGeometry(
                WGS84_PROJECTION,
                shapely.box(lon_min, lat_min, lon_min + 1, lat_min + 1),
                None,
            )
            item = Item(entry, geometry)

            key = (lon_min, lat_min)

            # For tile_to_item, prefer SRTM1 over SRTM3 when not using always_use_3arcsecond
            if key in tile_to_item:
                existing_is_srtm1 = os.path.basename(tile_to_item[key].name).startswith(
                    "SRTM1"
                )
                # Only replace if current is SRTM1 and existing is not
                if is_srtm1 and not existing_is_srtm1:
                    tile_to_item[key] = item
                # Keep existing if it's SRTM1 and current is not
                elif existing_is_srtm1 and not is_srtm1:
                    pass
                else:
                    # Same type, keep the existing one
                    pass
            else:
                tile_to_item[key] = item

            if basename not in basename_to_item:
                basename_to_item[basename] = item

        logger.info(
            f"Loaded {len(tile_to_item)} SRTM tiles from Hugging Face file list"
        )
        return basename_to_item, tile_to_item

    def _load_file_list_json(self) -> list[Any]:
        """Load file list JSON, optionally from cache."""
        if self.file_list_cache_path is not None and self.file_list_cache_path.exists():
            with self.file_list_cache_path.open() as f:
                file_list = json.load(f)
            logger.info(f"Loaded SRTM file list cache from {self.file_list_cache_path}")
            return file_list

        response = self.session.get(
            self.BASE_URL + self.FILE_LIST_FILENAME,
            timeout=self.timeout.total_seconds(),
        )
        response.raise_for_status()
        file_list = response.json()
        if self.file_list_cache_path is not None:
            with open_atomic(self.file_list_cache_path, "w") as f:
                json.dump(file_list, f)
        return file_list

    def _parse_tile_basename(self, basename: str) -> tuple[int, int]:
        """Parse a tile basename into (lat_min, lon_min)."""
        match = re.match(
            r"^SRTM\d([NS])(\d{2})([EW])(\d{3})V2\.tif$",
            basename,
        )
        if match is None:
            raise ValueError(f"invalid SRTM tile filename: {basename}")

        lat_sign, lat_str, lon_sign, lon_str = match.groups()
        lat_degrees = int(lat_str)
        lon_degrees = int(lon_str)

        if lat_sign == "N":
            lat_min = lat_degrees
        elif lat_sign == "S":
            lat_min = -lat_degrees
        else:
            raise ValueError(f"invalid SRTM tile latitude for filename: {basename}")

        if lon_sign == "E":
            lon_min = lon_degrees
        elif lon_sign == "W":
            lon_min = -lon_degrees
        else:
            raise ValueError(f"invalid SRTM tile longitude for filename: {basename}")

        return lat_min, lon_min

    def get_item_by_name(self, name: str) -> Item:
        """Gets an item by name.

        Args:
            name: the name of the item to get. For SRTM, the item name is the filename
                of the GeoTIFF tile.

        Returns:
            the Item object
        """
        basename = os.path.basename(name)
        item = self._basename_to_item.get(basename)
        if item is not None:
            return item
        raise ValueError(f"unknown SRTM tile name: {name}")

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[Item]]]:
        """Get a list of items in the data source intersecting the given geometries.

        Args:
            geometries: the spatiotemporal geometries
            query_config: the query configuration

        Returns:
            List of groups of items that should be retrieved for each geometry.
        """
        # It only makes sense to create mosaic from the SRTM data since it is one
        # global layer (but spatially split up into items).
        if query_config.space_mode != SpaceMode.MOSAIC or query_config.max_matches != 1:
            raise ValueError(
                "expected mosaic with max_matches=1 for the query configuration"
            )

        groups = []
        for geometry in geometries:
            wgs84_geometry = geometry.to_projection(WGS84_PROJECTION)
            shp_bounds = wgs84_geometry.shp.bounds
            cell_bounds = (
                math.floor(shp_bounds[0]),
                math.floor(shp_bounds[1]),
                math.ceil(shp_bounds[2]),
                math.ceil(shp_bounds[3]),
            )
            items = []
            for lon_min in range(cell_bounds[0], cell_bounds[2]):
                for lat_min in range(cell_bounds[1], cell_bounds[3]):
                    if (lon_min, lat_min) not in self._tile_to_item:
                        continue
                    items.append(self._tile_to_item[(lon_min, lat_min)])

            logger.debug(f"Got {len(items)} items (grid cells) for geometry")
            groups.append([items])

        return groups

    def deserialize_item(self, serialized_item: Any) -> Item:
        """Deserializes an item from JSON-decoded data."""
        assert isinstance(serialized_item, dict)
        return Item.deserialize(serialized_item)

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[Item],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        Args:
            tile_store: the tile store to ingest into
            items: the items to ingest
            geometries: a list of geometries needed for each item
        """
        for item in items:
            if tile_store.is_raster_ready(item.name, [self.band_name]):
                continue

            url = self.BASE_URL + item.name
            logger.debug(f"Downloading SRTM data for {item.name} from {url}")
            response = self.session.get(
                url, stream=True, timeout=self.timeout.total_seconds()
            )

            if response.status_code == 404:
                logger.warning(
                    f"Skipping item {item.name} because there is no data at that cell"
                )
                continue
            response.raise_for_status()

            with tempfile.TemporaryDirectory() as tmp_dir:
                local_fname = os.path.join(tmp_dir, "data.tif")
                with open(local_fname, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                logger.debug(f"Ingesting data for {item.name}")
                tile_store.write_raster_file(
                    item.name, [self.band_name], UPath(local_fname)
                )


# The code below is for creating the mirror by downloading SRTM GeoTIFFs from the USGS
# EarthExplorer M2M API.


@functools.cache
def _get_cached_m2m_client(timeout: timedelta) -> M2MAPIClient:
    """Get a cached M2M API client for this process.

    The client is cached per process, so each worker reuses its client across
    multiple download tasks.

    Args:
        timeout: timeout for API requests

    Returns:
        M2M API client
    """
    session = create_retry_session()
    return M2MAPIClient(timeout=timeout, session=session)


def _worker_download_scene(
    scene: dict[str, Any],
    output_path: str,
    timeout: timedelta,
) -> str:
    """Worker function for downloading a single SRTM GeoTIFF file.

    This function gets the download URL from the M2M API and downloads the file.
    The M2M client is cached per worker process to avoid repeated logins.

    Args:
        scene: scene metadata from scene_search
        output_path: path to save the downloaded file
        timeout: timeout for API requests and download

    Returns:
        the output path of the downloaded file
    """
    entity_id = scene["entityId"]
    display_id = scene["displayId"]

    logger.debug(f"Starting download for {display_id}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Get cached M2M client for this worker process
    client = _get_cached_m2m_client(timeout)

    # Get downloadable products for this scene
    logger.debug(f"Getting products for {display_id}")
    products = client.get_downloadable_products(SRTM_DATASET_NAME, entity_id)

    # Build a map of available products
    available_products = {}
    for product in products:
        if product.get("available", False) and product.get("id"):
            available_products[product["productName"]] = product["id"]

    # Try products in order of preference (highest resolution first)
    download_url = None
    for product_name in SRTM_PRODUCT_NAMES:
        if product_name in available_products:
            product_id = available_products[product_name]
            logger.debug(f"Getting download URL for {display_id} ({product_name})")
            download_url = client.get_download_url(entity_id, product_id)
            break

    if download_url is None:
        raise ValueError(
            f"No GeoTIFF product found for scene {display_id}. "
            f"Available products: {list(available_products.keys())}"
        )

    # Download with atomic write using retry session
    logger.debug(f"Downloading file for {display_id}")
    with client.session.get(
        download_url, stream=True, timeout=timeout.total_seconds()
    ) as r:
        r.raise_for_status()
        with open_atomic(UPath(output_path), "wb") as f:
            shutil.copyfileobj(r.raw, f)

    return output_path


def _fetch_all_scenes(
    client: M2MAPIClient,
    cache_dir: str,
) -> dict[str, dict[str, Any]]:
    """Fetch all SRTM scenes, using a cached scene list if available.

    SRTM data is organized in 1x1 degree tiles covering latitude -60 to 60.
    We iterate over 10x10 degree boxes to avoid timeout issues with global search.

    This fetches both SRTM1 (1 arc-second, US only) and SRTM3 (3 arc-second, global)
    scenes. The download function will select the highest resolution available.

    The scene list is cached in scenes.json in the cache directory. If the cache
    exists, it is loaded instead of querying the API.

    Args:
        client: M2M API client
        cache_dir: directory where cache file is stored

    Returns:
        dict mapping display_id to scene metadata
    """
    cache_path = os.path.join(cache_dir, SCENE_CACHE_FILENAME)
    scenes: dict[str, dict[str, Any]] = {}

    # Try to load from cache
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            scenes = json.load(f)
        logger.info(f"Loaded {len(scenes)} scenes from cache")
        return scenes

    # Fetch from API by iterating over 10x10 degree boxes
    logger.info("No cached scene list found, fetching from API...")
    box_size = 10
    total_boxes = ((SRTM_LAT_MAX - SRTM_LAT_MIN) // box_size) * (360 // box_size)

    box_idx = 0
    for lat in range(SRTM_LAT_MIN, SRTM_LAT_MAX, box_size):
        for lon in range(-180, 180, box_size):
            box_idx += 1
            bbox = (lon, lat, lon + box_size, lat + box_size)

            results = client.scene_search(SRTM_DATASET_NAME, bbox=bbox)
            for scene in results:
                display_id = scene["displayId"]
                if display_id not in scenes:
                    scenes[display_id] = scene

            logger.info(
                f"Searched {box_idx}/{total_boxes} boxes, "
                f"found {len(scenes)} unique scenes so far"
            )

    # Save to cache
    with open(cache_path, "w") as f:
        json.dump(scenes, f)
    logger.info(f"Cached {len(scenes)} scenes to {SCENE_CACHE_FILENAME}")

    return scenes


def bulk_download_srtm(
    output_dir: str,
    num_workers: int = 4,
    timeout: timedelta = timedelta(minutes=5),
) -> None:
    """Bulk download SRTM data from USGS EarthExplorer.

    Downloads all SRTM tiles to the specified output directory. Uses atomic
    rename to ensure partially downloaded files are not included. Files that
    already exist in the output directory are skipped.

    The scene list is cached in scenes.json in the output directory to avoid
    re-querying on subsequent runs.

    Requires M2M_USERNAME and M2M_TOKEN environment variables to be set.

    Args:
        output_dir: directory to save downloaded files
        num_workers: number of parallel download workers
        timeout: timeout for API requests and downloads
    """
    os.makedirs(output_dir, exist_ok=True)

    client = _get_cached_m2m_client(timeout)
    scenes = _fetch_all_scenes(client, output_dir)

    # Filter out scenes that are already downloaded
    download_tasks = []
    skipped = 0
    for display_id, scene in scenes.items():
        # Use display_id as the filename with .tif extension for GeoTIFF
        output_path = os.path.join(output_dir, f"{display_id}.tif")
        if os.path.exists(output_path):
            logger.debug(f"Skipping {display_id} - already downloaded")
            skipped += 1
        else:
            download_tasks.append(
                {
                    "scene": scene,
                    "output_path": output_path,
                    "timeout": timeout,
                }
            )

    logger.info(
        f"Need to download {len(download_tasks)} scenes ({skipped} already downloaded)"
    )

    if not download_tasks:
        logger.info("All scenes already downloaded!")
        return

    # Download in parallel using multiprocessing.Pool
    # Each worker creates its own M2M API client to get download URLs
    logger.info(f"Starting downloads with {num_workers} workers...")
    with multiprocessing.Pool(num_workers) as pool:
        for output_path in star_imap_unordered(
            pool, _worker_download_scene, download_tasks
        ):
            logger.info(f"Downloaded {output_path}")


def main() -> None:
    """Command-line entry point for bulk SRTM download.

    Requires M2M_USERNAME and M2M_TOKEN environment variables to be set.
    """
    import argparse
    import logging

    parser = argparse.ArgumentParser(
        description="Bulk download SRTM data from USGS EarthExplorer. "
        "Requires M2M_USERNAME and M2M_TOKEN environment variables."
    )
    parser.add_argument(
        "output_dir",
        help="Directory to save downloaded SRTM files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout in seconds for API requests and downloads (default: 30)",
    )

    args = parser.parse_args()

    # Configure logging based on RSLEARN_LOGLEVEL
    log_level = os.environ.get("RSLEARN_LOGLEVEL", "INFO")
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-6s %(name)s %(message)s",
    )
    # Enable urllib3 logging to see retry information at DEBUG level
    logging.getLogger("urllib3").setLevel(log_level)

    bulk_download_srtm(
        output_dir=args.output_dir,
        num_workers=args.workers,
        timeout=timedelta(seconds=args.timeout),
    )


if __name__ == "__main__":
    main()
