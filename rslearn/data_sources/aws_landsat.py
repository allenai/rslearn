"""Data source for raster data in Registry of Open Data on AWS."""

import io
import json
import os
import urllib.request
import zipfile
from collections.abc import Generator
from datetime import timedelta
from typing import Any, BinaryIO, Optional

import boto3
import dateutil.parser
import fiona
import fiona.transform
import pytimeparse
import rasterio
import shapely
import tqdm

import rslearn.data_sources.utils
import rslearn.utils.mgrs
from rslearn.config import LayerConfig, RasterLayerConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.tile_stores import PrefixedTileStore, TileStore
from rslearn.utils import STGeometry, open_atomic

from .data_source import DataSource, Item, QueryConfig
from .raster_source import get_needed_projections, ingest_raster


class LandsatOliTirsItem(Item):
    """An item in the LandsatOliTirs data source."""

    def __init__(
        self, name: str, geometry: STGeometry, blob_path: str, cloud_cover: float
    ):
        """Creates a new LandsatOliTirsItem.

        Args:
            name: unique name of the item
            geometry: the spatial and temporal extent of the item
            blob_path: path in bucket, e.g.,
                collection02/level-1/standard/oli-tirs/2024/032/028/LC09_L1GT_032028_20240214_20240214_02_T2/LC09_L1GT_032028_20240214_20240214_02_T2_
            cloud_cover: the scene's cloud cover (between 0 and 100)
        """
        super().__init__(name, geometry)
        self.blob_path = blob_path
        self.cloud_cover = cloud_cover

    def serialize(self) -> dict:
        """Serializes the item to a JSON-encodable dictionary."""
        d = super().serialize()
        d["blob_path"] = self.blob_path
        d["cloud_cover"] = self.cloud_cover
        return d

    @staticmethod
    def deserialize(d: dict) -> Item:
        """Deserializes an item from a JSON-decoded dictionary."""
        if "name" not in d:
            d["name"] = d["blob_path"].split("/")[-1].split(".tif")[0]
        item = super(LandsatOliTirsItem, LandsatOliTirsItem).deserialize(d)
        return LandsatOliTirsItem(
            name=item.name,
            geometry=item.geometry,
            blob_path=d["blob_path"],
            cloud_cover=d["cloud_cover"],
        )


class LandsatOliTirs(DataSource):
    """A data source for Landsat 8/9 OLI-TIRS imagery on AWS.

    Specifically, uses the usgs-landsat S3 bucket maintained by USGS. The data includes
    Tier 1/2 scenes but does not seem to include Real-Time scenes.

    See https://aws.amazon.com/marketplace/pp/prodview-ivr4jeq6flk7u for details about
    the bucket.
    """

    bucket_name = "usgs-landsat"
    bucket_prefix = "collection02/level-1/standard/oli-tirs"
    bands = [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B9",
        "B10",
        "B11",
    ]

    wrs2_url = "https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/WRS2_descending_0.zip"  # noqa
    """URL to download shapefile specifying polygon of each (path, row)."""

    def __init__(
        self,
        config: LayerConfig,
        metadata_cache_dir: str,
        max_time_delta: timedelta = timedelta(days=30),
        sort_by: Optional[str] = None,
    ) -> None:
        """Initialize a new LandsatOliTirs instance.

        Args:
            config: configuration of this layer
            metadata_cache_dir: local directory to cache product metadata files.
            max_time_delta: maximum time before a query start time or after a
                query end time to look for products. This is required due to the large
                number of available products, and defaults to 30 days.
            sort_by: can be "cloud_cover", default arbitrary order; only has effect for
                SpaceMode.WITHIN.
        """
        self.config = config
        self.metadata_cache_dir = metadata_cache_dir
        self.max_time_delta = max_time_delta
        self.sort_by = sort_by

        self.bucket = boto3.resource("s3").Bucket(self.bucket_name)

    @staticmethod
    def from_config(config: LayerConfig, root_dir: str = ".") -> "LandsatOliTirs":
        """Creates a new LandsatOliTirs instance from a configuration dictionary."""
        assert isinstance(config, RasterLayerConfig)
        d = config.data_source.config_dict
        if "max_time_delta" in d:
            max_time_delta = timedelta(seconds=pytimeparse.parse(d["max_time_delta"]))
        else:
            max_time_delta = timedelta(days=30)
        return LandsatOliTirs(
            config=config,
            metadata_cache_dir=os.path.join(root_dir, d["metadata_cache_dir"]),
            max_time_delta=max_time_delta,
            sort_by=d.get("sort_by"),
        )

    def _read_products(
        self, needed_year_pathrows: set[tuple[int, str, str]]
    ) -> Generator[LandsatOliTirsItem, None, None]:
        """Read _MTL.json files and yield relevant LandsatOliTirsItems.

        Args:
            needed_year_pathrows: set of (year, path, row) where we need to search for
                images.
        """
        for year, path, row in tqdm.tqdm(
            needed_year_pathrows, desc="Reading product infos"
        ):
            assert len(path) == 3
            assert len(row) == 3
            local_fname = os.path.join(
                self.metadata_cache_dir, f"{year}_{path}_{row}.json"
            )

            if not os.path.exists(local_fname):
                prefix = f"{self.bucket_prefix}/{year}/{path}/{row}/"
                items = []
                for obj in self.bucket.objects.filter(
                    Prefix=prefix, RequestPayer="requester"
                ):
                    if not obj.key.endswith("_MTL.json"):
                        continue
                    # Load JSON data.
                    buf = io.BytesIO()
                    self.bucket.download_fileobj(
                        obj.key, buf, ExtraArgs={"RequestPayer": "requester"}
                    )
                    buf.seek(0)
                    product = json.load(buf)
                    metadata = product["LANDSAT_METADATA_FILE"]
                    image_attributes = metadata["IMAGE_ATTRIBUTES"]
                    projection_attributes = metadata["PROJECTION_ATTRIBUTES"]

                    # Get polygon coordinates.
                    coordinates = []
                    for corner_id in ["UL", "UR", "LR", "LL"]:
                        lon = projection_attributes[f"CORNER_{corner_id}_LON_PRODUCT"]
                        lat = projection_attributes[f"CORNER_{corner_id}_LAT_PRODUCT"]
                        coordinates.append((lon, lat))

                    # Get datetime.
                    date_str = image_attributes["DATE_ACQUIRED"]
                    time_str = image_attributes["SCENE_CENTER_TIME"]
                    ts = dateutil.parser.isoparse(date_str + "T" + time_str)

                    blob_path = obj.key.split("MTL.json")[0]
                    geometry = STGeometry(
                        WGS84_PROJECTION,
                        shapely.Polygon(coordinates),
                        [ts, ts],
                    )
                    items.append(
                        LandsatOliTirsItem(
                            name=metadata["PRODUCT_CONTENTS"]["LANDSAT_PRODUCT_ID"],
                            geometry=geometry,
                            blob_path=blob_path,
                            cloud_cover=image_attributes["CLOUD_COVER"],
                        )
                    )

                with open_atomic(local_fname, "w") as f:
                    json.dump([item.serialize() for item in items], f)

            else:
                with open(local_fname) as f:
                    items = [
                        LandsatOliTirsItem.deserialize(item_dict)
                        for item_dict in json.load(f)
                    ]

            yield from items

    def get_wrs2_polygons(self) -> list[tuple[shapely.Geometry, str, str]]:
        """Get polygons for each (path, row) in the WRS2 grid.

        Returns:
            List of (polygon, path, row).
        """
        local_fname = os.path.join(self.metadata_cache_dir, "WRS2_descending.shp")
        if not os.path.exists(local_fname):
            # Download and extract zip to cache dir.
            zip_fname = os.path.join(self.metadata_cache_dir, "WRS2_descending_0.zip")
            os.makedirs(self.metadata_cache_dir, exist_ok=True)
            urllib.request.urlretrieve(self.wrs2_url, zip_fname)
            with zipfile.ZipFile(zip_fname, "r") as zipf:
                zipf.extractall(self.metadata_cache_dir)

        with fiona.open(local_fname) as src:
            polygons = []
            for feat in src:
                shp = shapely.geometry.shape(feat["geometry"])
                path = str(feat["properties"]["PATH"]).zfill(3)
                row = str(feat["properties"]["ROW"]).zfill(3)
                polygons.append((shp, path, row))
            return polygons

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
        wrs2_polygons = self.get_wrs2_polygons()
        needed_year_pathrows = set()
        wgs84_geometries = [
            geometry.to_projection(WGS84_PROJECTION) for geometry in geometries
        ]
        for wgs84_geometry in wgs84_geometries:
            if wgs84_geometry.time_range is None:
                raise ValueError(
                    "Landsat on AWS requires geometry time ranges to be set"
                )
            cur_pathrows = set()
            for polygon, path, row in wrs2_polygons:
                if wgs84_geometry.shp.intersects(polygon):
                    cur_pathrows.add((path, row))
            for path, row in cur_pathrows:
                for year in range(
                    (wgs84_geometry.time_range[0] - self.max_time_delta).year,
                    (wgs84_geometry.time_range[1] + self.max_time_delta).year + 1,
                ):
                    needed_year_pathrows.add((year, path, row))

        items = list(self._read_products(needed_year_pathrows))

        groups = []
        for geometry, wgs84_geometry in zip(geometries, wgs84_geometries):
            cur_items = []
            for item in items:
                if not wgs84_geometry.shp.intersects(item.geometry.shp):
                    continue
                cur_items.append(item)

            if self.sort_by == "cloud_cover":
                items.sort(key=lambda item: item.cloud_cover)
            elif self.sort_by is not None:
                raise ValueError(f"invalid sort_by setting ({self.sort_by})")

            cur_groups = rslearn.data_sources.utils.match_candidate_items_to_window(
                geometry, cur_items, query_config
            )
            groups.append(cur_groups)

        return groups

    def get_item_by_name(self, name: str) -> Item:
        """Gets an item by name."""
        # Product name is like LC08_L1TP_046027_20230715_20230724_02_T1.
        # We want to use _read_products so we need to extract:
        # - year: 2023
        # - path: 046
        # - row: 027
        parts = name.split("_")
        assert len(parts[2]) == 6
        assert len(parts[3]) == 8
        year = int(parts[3][0:4])
        path = parts[2][0:3]
        row = parts[2][3:6]
        for item in self._read_products({(year, path, row)}):
            if item.name == name:
                return item
        raise ValueError(f"item {name} not found")

    def deserialize_item(self, serialized_item: Any) -> Item:
        """Deserializes an item from JSON-decoded data."""
        assert isinstance(serialized_item, dict)
        return LandsatOliTirsItem.deserialize(serialized_item)

    def retrieve_item(self, item: Item) -> Generator[tuple[str, BinaryIO], None, None]:
        """Retrieves the rasters corresponding to an item as file streams."""
        for band in self.bands:
            buf = io.BytesIO()
            self.bucket.download_fileobj(
                item.blob_path + f"{band}.TIF",
                buf,
                ExtraArgs={"RequestPayer": "requester"},
            )
            buf.seek(0)
            fname = item.blob_path.split("/")[-1] + f"_{band}.TIF"
            yield (fname, buf)

    def ingest(
        self,
        tile_store: TileStore,
        items: list[Item],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        Args:
            tile_store: the tile store to ingest into
            items: the items to ingest
            geometries: a list of geometries needed for each item
        """
        for item, cur_geometries in zip(items, geometries):
            for band in self.bands:
                band_names = [band]
                cur_tile_store = PrefixedTileStore(
                    tile_store, (item.name, "_".join(band_names))
                )
                needed_projections = get_needed_projections(
                    cur_tile_store, band_names, self.config.band_sets, cur_geometries
                )
                if not needed_projections:
                    continue

                buf = io.BytesIO()
                self.bucket.download_fileobj(
                    item.blob_path + f"{band}.TIF",
                    buf,
                    ExtraArgs={"RequestPayer": "requester"},
                )
                buf.seek(0)
                with rasterio.open(buf) as raster:
                    for projection in needed_projections:
                        ingest_raster(
                            tile_store=cur_tile_store,
                            raster=raster,
                            projection=projection,
                            time_range=item.geometry.time_range,
                            layer_config=self.config,
                        )
