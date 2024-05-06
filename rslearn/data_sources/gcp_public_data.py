"""Data source for raster data on public Cloud Storage buckets."""

import csv
import gzip
import io
import json
import os
import xml.etree.ElementTree as ET
from datetime import timedelta
from typing import Any, BinaryIO, Generator, Optional

import dateutil.parser
import pytimeparse
import rasterio
import shapely
import tqdm
from google.cloud import storage

import rslearn.utils.mgrs
from rslearn.config import LayerConfig, QueryConfig, RasterLayerConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import DataSource, Item
from rslearn.data_sources.utils import match_candidate_items_to_window
from rslearn.tile_stores import PrefixedTileStore, TileStore
from rslearn.utils import STGeometry, open_atomic
from rslearn.utils.rtree_index import RtreeIndex

from .raster_source import get_needed_projections, ingest_raster


class Sentinel2Item(Item):
    """An item in the Sentinel2 data source."""

    def __init__(
        self, name: str, geometry: STGeometry, blob_prefix: str, cloud_cover: float
    ):
        """Creates a new Sentinel2Item.

        Args:
            name: unique name of the item
            geometry: the spatial and temporal extent of the item
            blob_prefix: blob path prefix for the images
            cloud_cover: cloud cover percentage between 0-100
        """
        super().__init__(name, geometry)
        self.blob_prefix = blob_prefix
        self.cloud_cover = cloud_cover

    def serialize(self) -> dict[str, Any]:
        """Serializes the item to a JSON-encodable dictionary."""
        d = super().serialize()
        d["blob_prefix"] = self.blob_prefix
        d["cloud_cover"] = self.cloud_cover
        return d

    @staticmethod
    def deserialize(d: dict[str, Any]) -> Item:
        """Deserializes an item from a JSON-decoded dictionary."""
        item = super(Sentinel2Item, Sentinel2Item).deserialize(d)
        return Sentinel2Item(
            name=item.name,
            geometry=item.geometry,
            blob_prefix=d["blob_prefix"],
            cloud_cover=d["cloud_cover"],
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

    bands = [
        ("B01.jp2", ["B01"]),
        ("B02.jp2", ["B02"]),
        ("B03.jp2", ["B03"]),
        ("B04.jp2", ["B04"]),
        ("B05.jp2", ["B05"]),
        ("B06.jp2", ["B06"]),
        ("B07.jp2", ["B07"]),
        ("B08.jp2", ["B08"]),
        ("B09.jp2", ["B09"]),
        ("B10.jp2", ["B10"]),
        ("B11.jp2", ["B11"]),
        ("B12.jp2", ["B12"]),
        ("B8A.jp2", ["B8A"]),
        ("TCI.jp2", ["R", "G", "B"]),
    ]

    def __init__(
        self,
        config: LayerConfig,
        index_cache_dir: str,
        max_time_delta: timedelta = timedelta(days=30),
        sort_by: Optional[str] = None,
        use_rtree_index: bool = True,
    ):
        """Initialize a new Sentinel2 instance.

        Args:
            index_cache_dir: local directory to cache the index.csv.gz contents, as
                well as individual product metadata files. Defaults to None in which
                case products are looked up from the cloud storage directly.
            max_time_delta: maximum time before a query start time or after a
                query end time to look for products. This is required due to the large
                number of available products, and defaults to 30 days.
            sort_by: can be "cloud_cover", default arbitrary order; only has effect for
                SpaceMode.WITHIN.
        """
        self.config = config
        self.index_cache_dir = index_cache_dir
        self.max_time_delta = max_time_delta
        self.sort_by = sort_by

        self.bucket = storage.Client().bucket(self.bucket_name)

        if use_rtree_index:
            rtree_fname = os.path.join(self.index_cache_dir, "rtree_index")
            needs_building = not os.path.exists(rtree_fname + ".dat")
            self.rtree_index = RtreeIndex(rtree_fname)
            if needs_building:
                self._build_index()
        else:
            self.rtree_index = None

    @staticmethod
    def from_config(config: LayerConfig) -> "Sentinel2":
        """Creates a new Sentinel2 instance from a configuration dictionary."""
        assert isinstance(config, RasterLayerConfig)
        d = config.data_source.config_dict
        if "max_time_delta" in d:
            max_time_delta = timedelta(seconds=pytimeparse.parse(d["max_time_delta"]))
        else:
            max_time_delta = timedelta(days=30)
        return Sentinel2(
            config=config,
            index_cache_dir=d["index_cache_dir"],
            max_time_delta=max_time_delta,
            sort_by=d.get("sort_by"),
            use_rtree_index=d.get("use_rtree_index", True),
        )

    def _read_index(self, desc: str) -> Generator[dict[str, str], None, None]:
        """Read the index.csv.gz in the Cloud Storage bucket.

        The CSV only contains the bounding box of each image and not the exact
        geometry, which can be retrieved from individual product metadata
        (MTD_MSIL1C.xml) files.
        """
        blob = self.bucket.blob(self.index_fname)
        with blob.open("rb") as blob_f:
            with gzip.open(blob_f, "rt") as gzip_f:
                reader = csv.DictReader(gzip_f)
                for row in tqdm.tqdm(reader, desc=desc):
                    if not row["BASE_URL"]:
                        continue
                    product_id = row["PRODUCT_ID"]
                    product_id_parts = product_id.split("_")
                    if len(product_id_parts) < 7:
                        continue
                    product_type = product_id_parts[1]
                    if product_type != "MSIL1C":
                        continue
                    time_str = product_id_parts[2]
                    tile_id = product_id_parts[5]
                    assert tile_id[0] == "T"

                    granule_id = row["GRANULE_ID"]
                    base_url = row["BASE_URL"].split(
                        "gs://gcp-public-data-sentinel-2/"
                    )[1]

                    blob_prefix = "{}/GRANULE/{}/IMG_DATA/{}_{}_".format(
                        base_url, granule_id, tile_id, time_str
                    )

                    # Extract the spatial and temporal bounds of the image.
                    bounds = (
                        float(row["EAST_LON"]),
                        float(row["SOUTH_LAT"]),
                        float(row["WEST_LON"]),
                        float(row["NORTH_LAT"]),
                    )
                    shp = shapely.box(*bounds)
                    sensing_time = dateutil.parser.isoparse(row["SENSING_TIME"])
                    geometry = STGeometry(
                        WGS84_PROJECTION, shp, (sensing_time, sensing_time)
                    )

                    cloud_cover = float(row["CLOUD_COVER"])

                    yield Sentinel2Item(product_id, geometry, blob_prefix, cloud_cover)

    def _build_index(self):
        """Build the RtreeIndex from items in the data source."""
        for item in self._read_index(desc="Building rtree index"):
            self.rtree_index.insert(
                item.geometry.shp.bounds, json.dumps(item.serialize())
            )

    def get_item_by_name(self, name: str) -> Item:
        """Gets an item by name.

        Reads the individual product metadata file (MTD_MSIL1C.xml) to get both the
        expected blob path where images are stored as well as the detailed geometry of
        the product (not just the bounding box).

        Args:
            name: the name of the item to get

        Returns:
            the item object
        """
        parts = name.split("_")
        assert len(parts[5]) == 6
        assert parts[5][0] == "T"
        cell_id = parts[5][1:]
        base_url = f"tiles/{cell_id[0:2]}/{cell_id[2:3]}/{cell_id[3:5]}/{name}.SAFE/"

        local_xml_fname = os.path.join(self.index_cache_dir, name + ".xml")
        if not os.path.exists(local_xml_fname):
            metadata_blob_path = base_url + "MTD_MSIL1C.xml"
            blob = self.bucket.blob(metadata_blob_path)
            tmp_local_xml_fname = local_xml_fname + ".tmp." + str(os.getpid())
            blob.download_to_filename(tmp_local_xml_fname)
            os.rename(tmp_local_xml_fname, local_xml_fname)

        tree = ET.parse(local_xml_fname)

        # The EXT_POS_LIST tag has flat list of polygon coordinates.
        elements = list(tree.iter("EXT_POS_LIST"))
        assert len(elements) == 1
        coords = elements[0].text.strip().split(" ")
        # Convert flat list of lat1 lon1 lat2 lon2 ...
        # into (lon1, lat1), (lon2, lat2), ...
        # Then we can get the shapely geometry.
        coords = [
            [float(coords[i + 1]), float(coords[i])] for i in range(0, len(coords), 2)
        ]
        shp = shapely.Polygon(coords)

        # Get blob prefix which is a subfolder of the base_url
        elements = list(tree.iter("IMAGE_FILE"))
        elements = [el for el in elements if el.text.endswith("_B01")]
        assert len(elements) == 1
        blob_prefix = base_url + elements[0].text.split("B01")[0]

        elements = list(tree.iter("PRODUCT_START_TIME"))
        assert len(elements) == 1
        start_time = dateutil.parser.isoparse(elements[0].text)

        elements = list(tree.iter("Cloud_Coverage_Assessment"))
        assert len(elements) == 1
        cloud_cover = float(elements[0].text)

        return Sentinel2Item(
            name,
            STGeometry(WGS84_PROJECTION, shp, (start_time, start_time)),
            blob_prefix,
            cloud_cover,
        )

    def _read_products(
        self, needed_cell_years: set[tuple[str, int]]
    ) -> Generator[Sentinel2Item, None, None]:
        """Read files and yield relevant Sentinel2Items.

        Args:
            needed_cell_years: set of (mgrs grid cell, year) where we need to search
                for images.
        """
        for cell_id, year in tqdm.tqdm(needed_cell_years, desc="Reading product infos"):
            assert len(cell_id) == 5
            local_fname = os.path.join(self.index_cache_dir, f"{cell_id}_{year}.json")

            if not os.path.exists(local_fname):
                cell_part1 = cell_id[0:2]
                cell_part2 = cell_id[2:3]
                cell_part3 = cell_id[3:5]

                items = []

                for product_prefix in ["S2A_MSIL1C", "S2B_MSIL1C"]:
                    blob_prefix = (
                        f"tiles/{cell_part1}/{cell_part2}/{cell_part3}/"
                        + f"{product_prefix}_{year}"
                    )
                    blobs = self.bucket.list_blobs(prefix=blob_prefix, delimiter="/")
                    for blob in blobs:
                        if not blob.name.endswith(".SAFE_$folder$"):
                            continue
                        item_name = blob.name.split("/")[-1].split(".SAFE_$folder$")[0]
                        item = self.get_item_by_name(item_name)
                        items.append(item)

                with open_atomic(local_fname, "w") as f:
                    json.dump([item.serialize() for item in items], f)

            else:
                with open(local_fname) as f:
                    items = [Sentinel2Item.deserialize(d) for d in json.load(f)]

            for item in items:
                yield item

    def _get_candidate_items_index(
        self, wgs84_geometries: list[STGeometry]
    ) -> list[list[list[Item]]]:
        """List relevant items using rtree index."""
        candidates = [[] for _ in wgs84_geometries]
        for idx, geometry in enumerate(wgs84_geometries):
            time_range = None
            if geometry.time_range:
                time_range = (
                    geometry.time_range[0] - self.max_time_delta,
                    geometry.time_range[1] + self.max_time_delta,
                )

            encoded_items = self.rtree_index.query(geometry.shp.bounds)
            for encoded_item in encoded_items:
                item = Sentinel2Item.deserialize(json.loads(encoded_item))
                if not item.geometry.intersects_time_range(time_range):
                    continue
                if not item.geometry.shp.intersects(geometry.shp):
                    continue
                item = self.get_item_by_name(item.name)
                if not item.geometry.shp.intersects(geometry.shp):
                    continue
                candidates[idx].append(item)
        return candidates

    def _get_candidate_items_direct(
        self, wgs84_geometries: list[STGeometry]
    ) -> list[list[list[Item]]]:
        """Use _read_products to list relevant items."""
        needed_cell_years = set()
        for wgs84_geometry in wgs84_geometries:
            if wgs84_geometry.time_range is None:
                raise ValueError(
                    "Sentinel2 on GCP requires geometry time ranges to be set"
                )
            for cell_id in rslearn.utils.mgrs.for_each_cell(wgs84_geometry.shp.bounds):
                for year in range(
                    (wgs84_geometry.time_range[0] - self.max_time_delta).year,
                    (wgs84_geometry.time_range[1] + self.max_time_delta).year,
                ):
                    needed_cell_years.add((cell_id, year))

        items_by_cell = {}
        for item in self._read_products(needed_cell_years):
            cell_id = "".join(item.blob_path.split("/")[1:4])
            if cell_id not in items_by_cell:
                items_by_cell[cell_id] = []
            items_by_cell[cell_id].append(item)

        candidates = [[] for _ in wgs84_geometries]
        for idx, geometry in enumerate(wgs84_geometries):
            for cell_id in rslearn.utils.mgrs.for_each_cell(geometry.shp.bounds):
                for item in items_by_cell.get(cell_id, []):
                    if not geometry.shp.intersects(item.geometry.shp):
                        continue
                    candidates[idx].append(item)

        return candidates

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
        wgs84_geometries = [
            geometry.to_projection(WGS84_PROJECTION) for geometry in geometries
        ]

        if self.rtree_index:
            candidates = self._get_candidate_items_index(wgs84_geometries)
        else:
            candidates = self._get_candidate_items_direct(wgs84_geometries)

        groups = []
        for geometry, item_list in zip(wgs84_geometries, candidates):
            if self.sort_by == "cloud_cover":
                item_list.sort(key=lambda item: item.cloud_cover)
            elif self.sort_by is not None:
                raise ValueError(f"invalid sort_by setting ({self.sort_by})")
            cur_groups = match_candidate_items_to_window(
                geometry, item_list, query_config
            )
            groups.append(cur_groups)
        return groups

    def deserialize_item(self, serialized_item: Any) -> Item:
        """Deserializes an item from JSON-decoded data."""
        assert isinstance(serialized_item, dict)
        return Sentinel2Item.deserialize(serialized_item)

    def retrieve_item(self, item: Item) -> Generator[tuple[str, BinaryIO], None, None]:
        """Retrieves the rasters corresponding to an item as file streams."""
        for suffix, _ in self.bands:
            blob_path = item.blob_prefix + suffix
            fname = blob_path.split("/")[-1]
            buf = io.BytesIO()
            blob = self.bucket.blob(item.blob_prefix + suffix)
            if not blob.exists():
                continue
            blob.download_to_file(buf)
            buf.seek(0)
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
            for suffix, band_names in self.bands:
                cur_tile_store = PrefixedTileStore(
                    tile_store, (item.name, "_".join(band_names))
                )
                needed_projections = get_needed_projections(
                    cur_tile_store, band_names, self.config.band_sets, cur_geometries
                )
                if not needed_projections:
                    continue

                buf = io.BytesIO()
                blob = self.bucket.blob(item.blob_prefix + suffix)
                if not blob.exists():
                    continue
                blob.download_to_file(buf)
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
