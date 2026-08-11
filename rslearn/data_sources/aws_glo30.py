"""Copernicus GLO-30 DEM (30m) from the original public S3 bucket.

This module provides a data source for the Copernicus Global 30m Digital Elevation
Model (GLO-30), reading directly from the public AWS S3 bucket
``copernicus-dem-30m`` (no credentials required).

The dataset is split into 1x1-degree Cloud-Optimized GeoTIFF tiles, with filenames
like::

    Copernicus_DSM_COG_10_N47_00_W123_00_DEM/Copernicus_DSM_COG_10_N47_00_W123_00_DEM.tif

Only tiles containing land are published, so the bucket includes a ``tileList.txt``
index that we download and cache to avoid creating items for tiles that do not exist.

This data source serves the raw ``elevation`` band. Derived terrain products (slope
and aspect) are computed on the fly at training/inference time; see
``rslearn.train.transforms.terrain.ElevationToSlopeAspect``.

The underlying elevation data were acquired by the TanDEM-X mission between 2011 and
2015. Items from this data source do not come with a time range (the DEM is static).

Note: the AWS ``copernicus-dem-30m`` bucket serves the Copernicus DEM 2021 release.
Newer releases (e.g. 2024_1) are only available through the Copernicus Data Space
Ecosystem API, not this public bucket. See also
``rslearn.data_sources.planetary_computer.CopDemGlo30`` for the same dataset via
Microsoft Planetary Computer.
"""

from __future__ import annotations

import math

import boto3
import botocore
import botocore.client
import shapely
from typing_extensions import override
from upath import UPath

from rslearn.config import QueryConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import DataSourceContext, Item
from rslearn.data_sources.data_source import ItemLookupDataSource
from rslearn.data_sources.direct_materialize_data_source import (
    DirectMaterializeDataSource,
)
from rslearn.data_sources.utils import MatchedItemGroup
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.fsspec import join_upath, open_atomic
from rslearn.utils.geometry import STGeometry, flatten_shape

logger = get_logger(__name__)

GLO30_BUCKET = "copernicus-dem-30m"
GLO30_REGION = "eu-central-1"
GLO30_BASE_URL = f"https://{GLO30_BUCKET}.s3.{GLO30_REGION}.amazonaws.com/"
TILE_LIST_KEY = "tileList.txt"

DEFAULT_BAND_NAME = "elevation"
DATA_ASSET = "dem"


def _tile_name(lat: int, lon: int) -> str:
    """Return the GLO-30 tile directory/file name for a 1x1-degree cell.

    Args:
        lat: integer latitude of the cell's southern edge.
        lon: integer longitude of the cell's western edge.

    Returns:
        tile name like ``Copernicus_DSM_COG_10_N47_00_W123_00_DEM``.
    """
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


def _tile_url(lat: int, lon: int, base_url: str = GLO30_BASE_URL) -> str:
    """Return the full HTTPS URL for a GLO-30 COG tile."""
    name = _tile_name(lat, lon)
    return f"{base_url}{name}/{name}.tif"


class CopernicusGLO30(DirectMaterializeDataSource[Item], ItemLookupDataSource[Item]):
    """Data source for Copernicus GLO-30 DEM from the public S3 bucket.

    The data is split into 1x1-degree Cloud-Optimized GeoTIFFs, so it is read
    on-demand at materialize time; there is no ingest step. The underlying elevation
    was acquired by the TanDEM-X mission between 2011 and 2015. Items from this data
    source do not carry a time range (the DEM is static).

    Only land tiles are published. We use the bucket's ``tileList.txt`` index to
    ensure we only create items for tiles that exist, so windows over ocean simply
    match fewer (or no) tiles rather than failing to read.

    The layer's ``band_sets`` should contain a single band set with a single band
    (the elevation band). Use
    ``rslearn.train.transforms.terrain.ElevationToSlopeAspect`` to derive slope and
    aspect from the elevation band at training/inference time.

    Example config::

        {
            "class_path": "rslearn.data_sources.aws_glo30.CopernicusGLO30",
            "init_args": {"metadata_cache_dir": "cache/glo30"},
            "query_config": {"space_mode": "MOSAIC", "max_matches": 1},
            "ingest": false
        }
    """

    BASE_URL = GLO30_BASE_URL

    def __init__(
        self,
        metadata_cache_dir: str,
        band_name: str = DEFAULT_BAND_NAME,
        context: DataSourceContext = DataSourceContext(),
    ):
        """Initialize a new CopernicusGLO30 instance.

        Args:
            metadata_cache_dir: directory to cache the tileList.txt index.
            band_name: band name to use if the layer config is missing from the
                context.
            context: the data source context.
        """
        if context.layer_config is not None:
            if len(context.layer_config.band_sets) != 1:
                raise ValueError("expected a single band set")
            bands = context.layer_config.band_sets[0].bands
            if len(bands) != 1:
                raise ValueError(
                    "expected band set to have a single band (the elevation band); "
                    "slope/aspect are computed by "
                    "rslearn.train.transforms.terrain.ElevationToSlopeAspect"
                )
            band_name = bands[0]

        super().__init__(asset_bands={DATA_ASSET: [band_name]})
        self.band_name = band_name

        if context.ds_path is not None:
            self._cache_dir = join_upath(context.ds_path, metadata_cache_dir)
        else:
            self._cache_dir = UPath(metadata_cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._tile_names: set[str] | None = None

    def _load_tile_names(self) -> set[str]:
        """Load the set of published tile names, downloading the index if needed.

        Returns:
            the set of tile names available in the bucket.
        """
        if self._tile_names is not None:
            return self._tile_names

        cache_file = self._cache_dir / TILE_LIST_KEY
        if not cache_file.exists():
            logger.info("downloading GLO-30 tile list to %s", cache_file)
            s3 = boto3.client(
                "s3",
                region_name=GLO30_REGION,
                config=botocore.client.Config(
                    signature_version=botocore.UNSIGNED,
                ),
            )
            response = s3.get_object(Bucket=GLO30_BUCKET, Key=TILE_LIST_KEY)
            content = response["Body"].read()
            with open_atomic(cache_file, "wb") as f:
                f.write(content)

        with cache_file.open() as f:
            self._tile_names = {line.strip() for line in f if line.strip()}
        return self._tile_names

    def _build_item(self, lon_min: int, lat_min: int) -> Item:
        """Create an Item for a 1x1-degree GLO-30 tile."""
        name = _tile_name(lat_min, lon_min)
        geometry = STGeometry(
            WGS84_PROJECTION,
            shapely.box(lon_min, lat_min, lon_min + 1, lat_min + 1),
            None,
        )
        return Item(name, geometry)

    # ------------------------------------------------------------------
    # DataSource interface
    # ------------------------------------------------------------------

    @override
    def get_item_by_name(self, name: str) -> Item:
        """Gets an item by name.

        Args:
            name: the tile name (e.g. ``Copernicus_DSM_COG_10_N47_00_W123_00_DEM``).

        Returns:
            the Item object.
        """
        lat, lon = self._parse_tile_name(name)
        return self._build_item(lon, lat)

    @staticmethod
    def _parse_tile_name(name: str) -> tuple[int, int]:
        """Parse a tile name to (lat_min, lon_min).

        Args:
            name: tile name like ``Copernicus_DSM_COG_10_N47_00_W123_00_DEM``.

        Returns:
            tuple of (latitude, longitude) of the tile's southwest corner.

        Raises:
            ValueError: if the name is not a valid GLO-30 tile name.
        """
        # Copernicus_DSM_COG_10_N47_00_W123_00_DEM
        parts = name.split("_")
        # parts: [Copernicus, DSM, COG, 10, N47, 00, W123, 00, DEM]
        if len(parts) != 9:
            raise ValueError(f"invalid GLO-30 tile name {name}")

        ns_part = parts[4]  # e.g. "N47" or "S03"
        ew_part = parts[6]  # e.g. "W123" or "E010"
        if ns_part[:1] not in ("N", "S") or ew_part[:1] not in ("E", "W"):
            raise ValueError(f"invalid GLO-30 tile name {name}")

        try:
            lat = int(ns_part[1:])
            lon = int(ew_part[1:])
        except ValueError:
            raise ValueError(f"invalid GLO-30 tile name {name}") from None

        if ns_part[0] == "S":
            lat = -lat
        if ew_part[0] == "W":
            lon = -lon

        return lat, lon

    @override
    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[MatchedItemGroup[Item]]]:
        """Get a list of items intersecting the given geometries.

        Args:
            geometries: the spatiotemporal geometries.
            query_config: the query configuration.

        Returns:
            list of groups of items for each geometry.
        """
        if query_config.space_mode != SpaceMode.MOSAIC or query_config.max_matches != 1:
            raise ValueError(
                "expected mosaic with max_matches=1 for the query configuration"
            )
        if query_config.min_matches != 0:
            raise ValueError(
                "min_matches is not supported for CopernicusGLO30; set min_matches=0"
            )

        tile_names = self._load_tile_names()

        groups = []
        for geometry in geometries:
            wgs84_geometry = geometry.to_wgs84()
            items: list[Item] = []
            seen: set[tuple[int, int]] = set()
            for shp in flatten_shape(wgs84_geometry.shp):
                shp_bounds = shp.bounds
                cell_bounds = (
                    math.floor(shp_bounds[0]),
                    math.floor(shp_bounds[1]),
                    math.ceil(shp_bounds[2]),
                    math.ceil(shp_bounds[3]),
                )
                for lon_min in range(cell_bounds[0], cell_bounds[2]):
                    for lat_min in range(cell_bounds[1], cell_bounds[3]):
                        key = (lon_min, lat_min)
                        if key in seen:
                            continue
                        seen.add(key)
                        # Only land tiles are published; skip ocean cells so that
                        # direct materialization never reads a missing tile.
                        if _tile_name(lat_min, lon_min) not in tile_names:
                            continue
                        items.append(self._build_item(lon_min, lat_min))

            logger.debug(f"Got {len(items)} GLO-30 items for geometry")
            groups.append([MatchedItemGroup(items, geometry.time_range)])

        return groups

    @override
    def deserialize_item(self, serialized_item: dict) -> Item:
        """Deserializes an item from JSON-decoded data."""
        return Item.deserialize(serialized_item)

    @override
    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[Item],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Not supported; the tiles are COGs that are read at materialize time."""
        raise NotImplementedError(
            "CopernicusGLO30 only supports direct materialization; "
            'set "ingest": false on the layer'
        )

    # ------------------------------------------------------------------
    # DirectMaterializeDataSource interface
    # ------------------------------------------------------------------

    @override
    def get_asset_url(self, item: Item, asset_key: str) -> str:
        """Return the URL of the COG backing the given item."""
        if asset_key != DATA_ASSET:
            raise ValueError(f"Unknown asset key: {asset_key}")
        lat, lon = self._parse_tile_name(item.name)
        return _tile_url(lat, lon, base_url=self.BASE_URL)
