"""Data source for ESA WorldCover 2021 via AWS S3.

The data is available as Cloud-Optimized GeoTIFFs on the ESA WorldCover S3 bucket.
See https://registry.opendata.aws/esa-worldcover-vito/ for details.
"""

import json
import os
import tempfile
from datetime import timedelta

import boto3
import botocore
import botocore.client
import requests
import shapely
from upath import UPath

import rslearn.data_sources.utils
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.data_source import (
    DataSourceContext,
    Item,
    QueryConfig,
)
from rslearn.data_sources.direct_materialize_data_source import (
    DirectMaterializeDataSource,
)
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.fsspec import join_upath
from rslearn.utils.geometry import STGeometry
from rslearn.utils.grid_index import GridIndex

logger = get_logger(__name__)

# These correspond to the bucket name/region for https://registry.opendata.aws/esa-worldcover-vito/
# along with expected paths within the bucket.
BUCKET_NAME = "esa-worldcover"
BUCKET_REGION = "eu-central-1"
GRID_GEOJSON_KEY = "v200/2021/esa_worldcover_grid.geojson"
TILE_PREFIX = "v200/2021/map"
HTTP_BASE = f"https://{BUCKET_NAME}.s3.{BUCKET_REGION}.amazonaws.com"

GRID_INDEX_CELL_SIZE = 3.0


class WorldCover(DirectMaterializeDataSource[Item]):
    """A data source for the ESA WorldCover 2021 land cover map.

    The data is served as Cloud-Optimized GeoTIFFs from the public AWS S3 bucket
    ``s3://esa-worldcover``. The bucket includes a GeoJSON index that we use for the
    prepare stage.

    See https://registry.opendata.aws/esa-worldcover-vito/ for details about the
    dataset.
    """

    def __init__(
        self,
        metadata_cache_dir: str,
        timeout: timedelta = timedelta(seconds=60),
        context: DataSourceContext = DataSourceContext(),
    ) -> None:
        """Create a new WorldCover instance.

        Args:
            metadata_cache_dir: directory to cache the tile grid GeoJSON.
            timeout: timeout for HTTP requests.
            context: the data source context.
        """
        super().__init__(asset_bands={"map": ["B1"]})
        self.timeout = timeout

        if context.ds_path is not None:
            self._cache_dir = join_upath(context.ds_path, metadata_cache_dir)
        else:
            self._cache_dir = UPath(metadata_cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._grid_index: GridIndex | None = None
        self._items_by_name: dict[str, Item] | None = None

    def _load_index(self) -> tuple[GridIndex, dict[str, Item]]:
        """Load the tile grid GeoJSON, downloading from S3 if not cached.

        Returns:
            Tuple of (grid_index, items_by_name dict).
        """
        if self._grid_index is not None and self._items_by_name is not None:
            return self._grid_index, self._items_by_name

        cache_file = self._cache_dir / "esa_worldcover_grid.geojson"
        if not cache_file.exists():
            logger.info("downloading WorldCover grid index to %s", cache_file)
            s3 = boto3.client(
                "s3",
                region_name=BUCKET_REGION,
                config=botocore.client.Config(
                    signature_version=botocore.UNSIGNED,
                ),
            )
            response = s3.get_object(Bucket=BUCKET_NAME, Key=GRID_GEOJSON_KEY)
            content = response["Body"].read()
            with cache_file.open("wb") as f:
                f.write(content)

        with cache_file.open() as f:
            fc = json.load(f)

        grid_index = GridIndex(GRID_INDEX_CELL_SIZE)
        items_by_name: dict[str, Item] = {}

        for feature in fc["features"]:
            ll_tile = feature["properties"]["ll_tile"]
            shp = shapely.geometry.shape(feature["geometry"])
            geometry = STGeometry(WGS84_PROJECTION, shp, None)
            item = Item(name=ll_tile, geometry=geometry)
            grid_index.insert(shp.bounds, item)
            items_by_name[ll_tile] = item

        self._grid_index = grid_index
        self._items_by_name = items_by_name
        return grid_index, items_by_name

    # --- DataSource implementation ---

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[Item]]]:
        """Get items intersecting the given geometries.

        Args:
            geometries: the spatiotemporal geometries.
            query_config: the query configuration.

        Returns:
            list of groups of items for each geometry.
        """
        grid_index, _ = self._load_index()

        groups = []
        for geometry in geometries:
            wgs84_geometry = geometry.to_projection(WGS84_PROJECTION)
            cur_items = []
            for item in grid_index.query(wgs84_geometry.shp.bounds):
                if not wgs84_geometry.shp.intersects(item.geometry.shp):
                    continue
                cur_items.append(item)

            cur_groups: list[list[Item]] = (
                rslearn.data_sources.utils.match_candidate_items_to_window(
                    geometry, cur_items, query_config
                )
            )
            groups.append(cur_groups)

        return groups

    def get_item_by_name(self, name: str) -> Item:
        """Gets an item by name.

        Args:
            name: the tile name (ll_tile value, e.g. "N30E060").

        Returns:
            the Item.
        """
        _, items_by_name = self._load_index()
        if name not in items_by_name:
            raise ValueError(f"WorldCover tile {name} not found")
        return items_by_name[name]

    def deserialize_item(self, serialized_item: dict) -> Item:
        """Deserializes an item from JSON-decoded data."""
        return Item.deserialize(serialized_item)

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[Item],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        Downloads individual GeoTIFF tiles on demand rather than bulk zip files.

        Args:
            tile_store: the tile store to ingest into.
            items: the items to ingest.
            geometries: a list of geometries needed for each item.
        """
        for item in items:
            band_names = self.asset_bands["map"]
            if tile_store.is_raster_ready(item.name, band_names):
                continue

            url = self._tile_http_url(item.name)
            logger.info("downloading WorldCover tile %s", url)

            with tempfile.TemporaryDirectory() as tmp_dir:
                local_fname = os.path.join(tmp_dir, f"{item.name}.tif")
                with requests.get(
                    url, stream=True, timeout=self.timeout.total_seconds()
                ) as r:
                    r.raise_for_status()
                    with open(local_fname, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)

                tile_store.write_raster_file(
                    item.name,
                    band_names,
                    UPath(local_fname),
                )

    # --- DirectMaterializeDataSource implementation ---

    def get_asset_url(self, item_name: str, asset_key: str) -> str:
        """Get the URL to read a WorldCover tile COG.

        Args:
            item_name: the tile name (ll_tile, e.g. "N30E060").
            asset_key: ignored (always "map").

        Returns:
            a /vsicurl/ URL readable by rasterio.
        """
        return self._tile_http_url(item_name)

    @staticmethod
    def _tile_http_url(ll_tile: str) -> str:
        return (
            f"{HTTP_BASE}/{TILE_PREFIX}/ESA_WorldCover_10m_2021_v200_{ll_tile}_Map.tif"
        )
