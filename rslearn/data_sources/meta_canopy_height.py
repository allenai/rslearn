"""Data source for the Meta/WRI Canopy Height Maps v2 (CHMv2) via AWS S3.

The data is a global 1-meter resolution canopy height map derived from high-resolution
satellite imagery using a DINOv3-based model. About 80% of the source imagery spans
2018 to 2020, so the map roughly represents canopy height over that period. It is
served as Cloud-Optimized GeoTIFFs from the public AWS Open Data bucket
``s3://dataforgood-fb-data``.

See https://registry.opendata.aws/dataforgood-fb-forestsv2/ and
https://arxiv.org/abs/2603.06382 for details.
"""

import json
from datetime import timedelta

import boto3
import botocore
import botocore.client
import shapely
from typing_extensions import override
from upath import UPath

import rslearn.data_sources.utils
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.data_source import (
    DataSourceContext,
    Item,
    ItemLookupDataSource,
    QueryConfig,
)
from rslearn.data_sources.direct_materialize_data_source import (
    DirectMaterializeDataSource,
)
from rslearn.data_sources.utils import MatchedItemGroup
from rslearn.log_utils import get_logger
from rslearn.utils.fsspec import join_upath, open_atomic
from rslearn.utils.geometry import STGeometry
from rslearn.utils.grid_index import GridIndex

logger = get_logger(__name__)

# Bucket name/region/prefixes for https://registry.opendata.aws/dataforgood-fb-forestsv2/.
BUCKET_NAME = "dataforgood-fb-data"
BUCKET_REGION = "us-east-1"
DATASET_PREFIX = "forests/v2/global/dinov3_global_chm_v2_ml3"
TILES_GEOJSON_KEY = f"{DATASET_PREFIX}/tiles.geojson"
CHM_PREFIX = f"{DATASET_PREFIX}/chm"
HTTP_BASE = f"https://{BUCKET_NAME}.s3.{BUCKET_REGION}.amazonaws.com"

# The quadkey (tile filename) is stored under this property in tiles.geojson.
QUADKEY_PROPERTY = "tile"

# CHMv2 tiles are zoom-10 web mercator tiles (~0.35 degrees at the equator), so a
# 1-degree grid index cell keeps lookups efficient.
GRID_INDEX_CELL_SIZE = 1.0

# The canopy height band name exposed by this data source.
CANOPY_HEIGHT_BAND = "canopy_height"


class MetaCanopyHeightV2(DirectMaterializeDataSource[Item], ItemLookupDataSource[Item]):
    """A data source for the Meta/WRI Canopy Height Maps v2 (CHMv2).

    About 80% of the source imagery spans 2018 to 2020, so the map roughly
    represents canopy height over that period.

    The data is served as Cloud-Optimized GeoTIFFs (single band, canopy height in
    meters above ground) from the public AWS S3 bucket ``s3://dataforgood-fb-data``.
    Tiles are named by their zoom-10 web mercator quadkey and stored in EPSG:3857;
    reprojection to the window projection happens on read.

    The bucket includes ``tiles.geojson`` mapping each tile extent to its quadkey,
    which we use as the authoritative index for the prepare stage. This means we only
    ever create items for tiles that actually exist (no items for ocean/no-data areas).

    Data is read directly from the COGs at materialize time; this data source does not
    support ingest.

    See https://registry.opendata.aws/dataforgood-fb-forestsv2/ for details.
    """

    def __init__(
        self,
        metadata_cache_dir: str,
        timeout: timedelta = timedelta(seconds=60),
        context: DataSourceContext = DataSourceContext(),
    ) -> None:
        """Create a new MetaCanopyHeightV2 instance.

        Args:
            metadata_cache_dir: directory to cache the tiles.geojson index.
            timeout: timeout for HTTP requests.
            context: the data source context.
        """
        super().__init__(asset_bands={"chm": [CANOPY_HEIGHT_BAND]})
        self.timeout = timeout

        if context.ds_path is not None:
            self._cache_dir = join_upath(context.ds_path, metadata_cache_dir)
        else:
            self._cache_dir = UPath(metadata_cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._grid_index: GridIndex | None = None
        self._items_by_name: dict[str, Item] | None = None

    def _load_index(self) -> tuple[GridIndex, dict[str, Item]]:
        """Load the tiles.geojson index, downloading from S3 if not cached.

        Returns:
            Tuple of (grid_index, items_by_name dict).
        """
        if self._grid_index is not None and self._items_by_name is not None:
            return self._grid_index, self._items_by_name

        cache_file = self._cache_dir / "chmv2_tiles.geojson"
        if not cache_file.exists():
            logger.info("downloading CHMv2 tiles index to %s", cache_file)
            s3 = boto3.client(
                "s3",
                region_name=BUCKET_REGION,
                config=botocore.client.Config(
                    signature_version=botocore.UNSIGNED,
                ),
            )
            response = s3.get_object(Bucket=BUCKET_NAME, Key=TILES_GEOJSON_KEY)
            content = response["Body"].read()
            with open_atomic(cache_file, "wb") as f:
                f.write(content)

        with cache_file.open() as f:
            fc = json.load(f)

        grid_index = GridIndex(GRID_INDEX_CELL_SIZE)
        items_by_name: dict[str, Item] = {}

        for feature in fc["features"]:
            quadkey = str(feature["properties"][QUADKEY_PROPERTY])
            shp = shapely.geometry.shape(feature["geometry"])
            geometry = STGeometry(WGS84_PROJECTION, shp, None)
            item = Item(name=quadkey, geometry=geometry)
            grid_index.insert(shp.bounds, item)
            items_by_name[quadkey] = item

        self._grid_index = grid_index
        self._items_by_name = items_by_name
        return grid_index, items_by_name

    # --- DataSource implementation ---

    @override
    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[MatchedItemGroup[Item]]]:
        grid_index, _ = self._load_index()

        groups = []
        for geometry in geometries:
            wgs84_geometry = geometry.to_projection(WGS84_PROJECTION)
            cur_items = []
            for item in grid_index.query(wgs84_geometry.shp.bounds):
                if not wgs84_geometry.shp.intersects(item.geometry.shp):
                    continue
                cur_items.append(item)

            cur_groups: list[MatchedItemGroup[Item]] = (
                rslearn.data_sources.utils.match_candidate_items_to_window(
                    geometry, cur_items, query_config
                )
            )
            groups.append(cur_groups)

        return groups

    @override
    def get_item_by_name(self, name: str) -> Item:
        """Gets an item by name.

        Args:
            name: the tile quadkey (e.g. "0022222122").

        Returns:
            the Item.
        """
        _, items_by_name = self._load_index()
        if name not in items_by_name:
            raise ValueError(f"CHMv2 tile {name} not found")
        return items_by_name[name]

    @override
    def deserialize_item(self, serialized_item: dict) -> Item:
        return Item.deserialize(serialized_item)

    # --- DirectMaterializeDataSource implementation ---

    @override
    def get_asset_url(self, item: Item, asset_key: str) -> str:
        assert asset_key == "chm", f"Unknown asset key: {asset_key}"
        # Item name is the quadkey that goes into the filename.
        return f"{HTTP_BASE}/{CHM_PREFIX}/{item.name}.tif"
