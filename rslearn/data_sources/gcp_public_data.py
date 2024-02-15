"""Data source for raster data on public Cloud Storage buckets."""

import csv
import gzip
import os
from typing import Generator, Optional

import dateutil.parser
import shapely
from google.cloud import storage
from rasterio.crs import CRS

from rslearn.config import LayerConfig
from rslearn.data_sources import DataSource, Item, QueryConfig
from rslearn.utils.geometry import STGeometry
from rslearn.utils.grid_index import GridIndex


class Sentinel2Item(Item):
    """An item in the Sentinel2 data source."""

    def __init__(self, name: str, geometry: STGeometry, blob_path_tmpl):
        """Creates a new Sentinel2Item.

        Args:
            name: unique name of the item
            geometry: the spatial and temporal extent of the item
            blob_path_tmpl: blob path for the image with band name placeholder
        """
        super().__init__(name, geometry)
        self.blob_path_tmpl = blob_path_tmpl

    def serialize(self) -> dict:
        """Serializes the item to a JSON-encodable dictionary."""
        d = super().serialize()
        d["blob_path_tmpl"] = self.blob_path_tmpl
        return d

    def deserialize(d: dict) -> Item:
        """Deserializes an item from a JSON-decoded dictionary."""
        item = super().deserialize(d)
        return Sentinel2Item(
            name=item.name,
            geometry=item.geometry,
            blob_path_tmpl=d["blob_path_tmpl"],
        )


class Sentinel2(DataSource):
    """A data source for Sentinel-2 data on Google Cloud Storage.

    Sentinel-2 imagery is available on Google Cloud Storage as part of the Google
    Public Cloud Data Program. The images are added with a 1-2 day latency after
    becoming available on Copernicus.

    See https://cloud.google.com/storage/docs/public-datasets/sentinel-2 for details.
    """

    bucket_name = "gcp-public-data-sentinel-2"

    index_fname = "index.csv.gz"

    def __init__(self, config: LayerConfig, index_cache_fname: Optional[str] = None):
        """Initialize a new Sentinel2 instance.

        Args:
            index_cache_fname: local file to cache index.csv.gz. If None, no caching is
                done.
        """
        self.bucket = storage.Client().bucket(self.bucket_name)
        self.index_cache_fname = index_cache_fname

    def _read_index(self) -> Generator[dict[str, str], None, None]:
        if self.index_cache_fname:
            if not os.path.exists(self.index_cache_fname):
                blob = self.bucket.blob(self.index_fname)
                blob.download_to_filename(self.index_cache_fname)

            with gzip.open(self.index_cache_fname, "rt") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    yield row

        else:
            blob = self.bucket.blob(self.index_fname)
            with blob.open("rb") as blob_f:
                with gzip.open(blob_f, "rt") as gzip_f:
                    reader = csv.DictReader(gzip_f)
                    for row in reader:
                        yield row

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[Item]:
        """Get a list of items in the data source intersecting the given geometries.

        Args:
            geometries: the spatiotemporal geometries
            query_config: the query configuration

        Returns:
            List of items that intersect the given geometries.
        """
        wgs84_crs = CRS.from_epsg(4326)
        wgs84_geometries = [geometry.to_crs(wgs84_crs, 1) for geometry in geometries]
        index = GridIndex(0.1)
        for idx, geometry in enumerate(wgs84_geometries):
            index.insert_rect(geometry.bounds, idx)

        items = [[] for _ in geometries]
        for i, row in enumerate(self._read_index()):
            if i % 10000 == 0:
                print(i)

            product_id_parts = row["PRODUCT_ID"].split("_")
            if len(product_id_parts) < 7:
                continue
            product_type = product_id_parts[1]
            if product_type != "MSIL1C":
                continue
            time_str = product_id_parts[2]
            tile_id = product_id_parts[5]
            assert tile_id[0] == "T"

            granule_id = row["GRANULE_ID"]
            base_url = row["BASE_URL"].split("gs://gcp-public-data-sentinel-2/")[1]

            blob_path_tmpl = "{}/GRANULE/{}/IMG_DATA/{}_{}_CHANNEL.jp2".format(
                base_url, granule_id, tile_id, time_str
            )

            # Extract the spatial and temporal bounds of the image.
            bounds = (
                row["EAST_LON"],
                row["SOUTH_LAT"],
                row["WEST_LON"],
                row["NORTH_LAT"],
            )
            shp = shapely.box(*bounds)
            sensing_time = dateutil.parser.isoparse(row["SENSING_TIME"])

            results = index.query(bounds)
            for idx in results:
                geometry = wgs84_geometries[idx]
                if geometry.time_range and (
                    sensing_time < geometry.time_range[0]
                    or sensing_time >= geometry.time_range[1]
                ):
                    continue
                if not geometry.shp.intersects(shp):
                    continue
                items[idx].append(Sentinel2Item(shp, sensing_time, blob_path_tmpl))

        return items
