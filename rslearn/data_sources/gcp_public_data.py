"""Data source for raster data on public Cloud Storage buckets."""

import io
import json
import random
import xml.etree.ElementTree as ET
from collections.abc import Generator
from datetime import datetime, timedelta
from typing import Any, BinaryIO

import dateutil.parser
import pytimeparse
import rasterio
import shapely
import tqdm
from google.cloud import bigquery, storage
from upath import UPath

from rslearn.config import QueryConfig, RasterLayerConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import DataSource, Item
from rslearn.data_sources.utils import match_candidate_items_to_window
from rslearn.log_utils import get_logger
from rslearn.tile_stores import PrefixedTileStore, TileStore
from rslearn.utils.fsspec import join_upath, open_atomic
from rslearn.utils.geometry import STGeometry, flatten_shape, split_at_prime_meridian

from .copernicus import get_harmonize_callback, get_sentinel2_tiles
from .raster_source import get_needed_projections, ingest_raster

logger = get_logger(__name__)


# TODO: this is a copy of the Sentinel2Item class in aws_open_data.py
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
    def deserialize(d: dict[str, Any]) -> "Sentinel2Item":
        """Deserializes an item from a JSON-decoded dictionary."""
        item = super(Sentinel2Item, Sentinel2Item).deserialize(d)
        return Sentinel2Item(
            name=item.name,
            geometry=item.geometry,
            blob_prefix=d["blob_prefix"],
            cloud_cover=d["cloud_cover"],
        )


# TODO: Distinguish between AWS and GCP data sources in class names.
class Sentinel2(DataSource):
    """A data source for Sentinel-2 data on Google Cloud Storage.

    Sentinel-2 imagery is available on Google Cloud Storage as part of the Google
    Public Cloud Data Program. The images are added with a 1-2 day latency after
    becoming available on Copernicus.

    See https://cloud.google.com/storage/docs/public-datasets/sentinel-2 for details.

    The bucket is public and free so no credentials are needed.
    """

    BUCKET_NAME = "gcp-public-data-sentinel-2"

    # Name of BigQuery table containing index of Sentinel-2 scenes in the bucket.
    TABLE_NAME = "bigquery-public-data.cloud_storage_geo_index.sentinel_2_index"

    BANDS = [
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
        config: RasterLayerConfig,
        index_cache_dir: UPath,
        max_time_delta: timedelta = timedelta(days=30),
        sort_by: str | None = None,
        use_rtree_index: bool = True,
        harmonize: bool = False,
        rtree_time_range: tuple[datetime, datetime] | None = None,
        rtree_cache_dir: UPath | None = None,
    ):
        """Initialize a new Sentinel2 instance.

        Args:
            config: the LayerConfig of the layer containing this data source.
            index_cache_dir: local directory to cache the index contents, as well as
                individual product metadata files.
            max_time_delta: maximum time before a query start time or after a
                query end time to look for products. This is required due to the large
                number of available products, and defaults to 30 days.
            sort_by: can be "cloud_cover", default arbitrary order; only has effect for
                SpaceMode.WITHIN.
            use_rtree_index: whether to create an rtree index to enable faster lookups
                (default true). Note that the rtree is populated from a BigQuery table
                where Google maintains an index, and this requires GCP credentials to
                query; additionally, rtree creation can take several minutes/hours. Use
                use_rtree_index=False to avoid the need for credentials.
            harmonize: harmonize pixel values across different processing baselines,
                see https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
            rtree_time_range: only populate the rtree index with scenes within this
                time range. Restricting to a few months significantly speeds up rtree
                creation time.
            rtree_cache_dir: by default, if use_rtree_index is enabled, the rtree is
                stored in index_cache_dir (where product XML files are also stored). If
                rtree_cache_dir is set, then the rtree is stored here instead (so
                index_cache_dir is only used to cache product XML files).
        """
        self.config = config
        self.index_cache_dir = index_cache_dir
        self.max_time_delta = max_time_delta
        self.sort_by = sort_by
        self.harmonize = harmonize

        self.index_cache_dir.mkdir(parents=True, exist_ok=True)

        self.bucket = storage.Client.create_anonymous_client().bucket(self.BUCKET_NAME)
        self.rtree_index: Any | None = None
        if use_rtree_index:
            from rslearn.utils.rtree_index import RtreeIndex, get_cached_rtree

            if rtree_cache_dir is None:
                rtree_cache_dir = self.index_cache_dir
            rtree_cache_dir.mkdir(parents=True, exist_ok=True)

            def build_fn(index: RtreeIndex) -> None:
                """Build the RtreeIndex from items in the data source."""
                for item in self._read_index(
                    desc="Building rtree index", time_range=rtree_time_range
                ):
                    for shp in flatten_shape(item.geometry.shp):
                        index.insert(shp.bounds, json.dumps(item.serialize()))

            self.rtree_index = get_cached_rtree(rtree_cache_dir, build_fn)

    @staticmethod
    def from_config(config: RasterLayerConfig, ds_path: UPath) -> "Sentinel2":
        """Creates a new Sentinel2 instance from a configuration dictionary."""
        if config.data_source is None:
            raise ValueError("config.data_source is required")
        d = config.data_source.config_dict
        kwargs = dict(
            config=config,
            index_cache_dir=join_upath(ds_path, d["index_cache_dir"]),
        )

        if "max_time_delta" in d:
            kwargs["max_time_delta"] = timedelta(
                seconds=pytimeparse.parse(d["max_time_delta"])
            )

        if "rtree_time_range" in d:
            kwargs["rtree_time_range"] = (
                datetime.fromisoformat(d["rtree_time_range"][0]),
                datetime.fromisoformat(d["rtree_time_range"][1]),
            )

        if "rtree_cache_dir" in d:
            kwargs["rtree_cache_dir"] = join_upath(ds_path, d["rtree_cache_dir"])

        simple_optionals = ["sort_by", "use_rtree_index", "harmonize"]
        for k in simple_optionals:
            if k in d:
                kwargs[k] = d[k]

        return Sentinel2(**kwargs)

    def _read_index(
        self, desc: str, time_range: tuple[datetime, datetime] | None = None
    ) -> Generator[Sentinel2Item, None, None]:
        """Read Sentinel-2 scenes from BigQuery table.

        The table only contains the bounding box of each image and not the exact
        geometry, which can be retrieved from individual product metadata
        (MTD_MSIL1C.xml) files.

        Args:
            desc: description to include with tqdm progress bar.
            time_range: optional time_range to restrict the reading.
        """
        query_str = f"""
            SELECT  source_url, base_url, product_id, sensing_time, granule_id,
                    east_lon, south_lat, west_lon, north_lat, cloud_cover
            FROM    `{self.TABLE_NAME}`
        """
        if time_range is not None:
            query_str += f"""
                WHERE sensing_time >= "{time_range[0]}" AND sensing_time <= "{time_range[1]}"
            """

        client = bigquery.Client()
        result = client.query(query_str)

        for row in tqdm.tqdm(result, desc=desc):
            # Some entries have source_url correct and others have base_url correct.
            # If base_url is correct, then it seems the source_url always ends in
            # index.csv.gz.
            if row["source_url"] and not row["source_url"].endswith("index.csv.gz"):
                base_url = row["source_url"].split(f"gs://{self.BUCKET_NAME}/")[1]
            elif row["base_url"] is not None and row["base_url"] != "":
                base_url = row["base_url"].split(f"gs://{self.BUCKET_NAME}/")[1]
            else:
                raise ValueError(
                    f"Unexpected value '{row['source_url']}' in column 'source_url'"
                    + f" and '{row['base_url']} in column 'base_url'"
                )

            product_id = row["product_id"]
            product_id_parts = product_id.split("_")
            if len(product_id_parts) < 7:
                continue
            product_type = product_id_parts[1]
            if product_type != "MSIL1C":
                continue
            time_str = product_id_parts[2]
            tile_id = product_id_parts[5]
            assert tile_id[0] == "T"

            granule_id = row["granule_id"]

            blob_prefix = (
                f"{base_url}/GRANULE/{granule_id}/IMG_DATA/{tile_id}_{time_str}_"
            )

            # Extract the spatial and temporal bounds of the image.
            bounds = (
                float(row["east_lon"]),
                float(row["south_lat"]),
                float(row["west_lon"]),
                float(row["north_lat"]),
            )
            shp = shapely.box(*bounds)
            sensing_time = row["sensing_time"]
            geometry = STGeometry(WGS84_PROJECTION, shp, (sensing_time, sensing_time))
            geometry = split_at_prime_meridian(geometry)

            cloud_cover = float(row["cloud_cover"])

            yield Sentinel2Item(product_id, geometry, blob_prefix, cloud_cover)

    def _get_xml_by_name(self, name: str) -> ET.ElementTree:
        """Gets the metadata XML of an item by its name.

        Args:
            name: the name of the item

        Returns:
            the parsed XML ElementTree
        """
        parts = name.split("_")
        assert len(parts[5]) == 6
        assert parts[5][0] == "T"
        cell_id = parts[5][1:]
        base_url = f"tiles/{cell_id[0:2]}/{cell_id[2:3]}/{cell_id[3:5]}/{name}.SAFE/"

        cache_xml_fname = self.index_cache_dir / (name + ".xml")
        if not cache_xml_fname.exists():
            metadata_blob_path = base_url + "MTD_MSIL1C.xml"
            blob = self.bucket.blob(metadata_blob_path)
            with open_atomic(cache_xml_fname, "wb") as f:
                blob.download_to_file(f)

        with cache_xml_fname.open("rb") as f:
            return ET.parse(f)

    def get_item_by_name(self, name: str) -> Sentinel2Item:
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

        tree = self._get_xml_by_name(name)

        # The EXT_POS_LIST tag has flat list of polygon coordinates.
        elements = list(tree.iter("EXT_POS_LIST"))
        assert len(elements) == 1
        if elements[0].text is None:
            raise ValueError(f"EXT_POS_LIST is empty for {name}")
        coords_text = elements[0].text.strip().split(" ")
        # Convert flat list of lat1 lon1 lat2 lon2 ...
        # into (lon1, lat1), (lon2, lat2), ...
        # Then we can get the shapely geometry.
        coords = [
            [float(coords_text[i + 1]), float(coords_text[i])]
            for i in range(0, len(coords_text), 2)
        ]
        shp = shapely.Polygon(coords)

        # Get blob prefix which is a subfolder of the base_url
        elements = list(tree.iter("IMAGE_FILE"))
        elements = [
            el for el in elements if el.text is not None and el.text.endswith("_B01")
        ]
        assert len(elements) == 1
        if elements[0].text is None:
            raise ValueError(f"IMAGE_FILE is empty for {name}")
        blob_prefix = base_url + elements[0].text.split("B01")[0]

        elements = list(tree.iter("PRODUCT_START_TIME"))
        assert len(elements) == 1
        if elements[0].text is None:
            raise ValueError(f"PRODUCT_START_TIME is empty for {name}")
        start_time = dateutil.parser.isoparse(elements[0].text)

        elements = list(tree.iter("Cloud_Coverage_Assessment"))
        assert len(elements) == 1
        if elements[0].text is None:
            raise ValueError(f"Cloud_Coverage_Assessment is empty for {name}")
        cloud_cover = float(elements[0].text)

        geometry = STGeometry(WGS84_PROJECTION, shp, (start_time, start_time))
        geometry = split_at_prime_meridian(geometry)

        return Sentinel2Item(
            name,
            geometry,
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
        # Read the product infos in random order so in case there are multiple jobs
        # reading similar cells, they are more likely to work on different cells/years
        # in parallel.
        needed_cell_years_list = list(needed_cell_years)
        random.shuffle(needed_cell_years_list)

        for cell_id, year in tqdm.tqdm(
            needed_cell_years_list, desc="Reading product infos"
        ):
            assert len(cell_id) == 5
            cache_fname = self.index_cache_dir / f"{cell_id}_{year}.json"

            if not cache_fname.exists():
                cell_part1 = cell_id[0:2]
                cell_part2 = cell_id[2:3]
                cell_part3 = cell_id[3:5]

                items = []

                for product_prefix in ["S2A_MSIL1C", "S2B_MSIL1C"]:
                    cell_folder = f"tiles/{cell_part1}/{cell_part2}/{cell_part3}"
                    blob_prefix = f"{cell_folder}/{product_prefix}_{year}"
                    blobs = self.bucket.list_blobs(prefix=blob_prefix, delimiter="/")

                    # Need to consume the iterator to obtain folder names.
                    # See https://cloud.google.com/storage/docs/samples/storage-list-files-with-prefix#storage_list_files_with_prefix-python # noqa: E501
                    # Previously we checked for .SAFE_$folder$ blobs here, but those do
                    # not exist for some years like 2017.
                    for _ in blobs:
                        pass

                    logger.debug(
                        "under %s, found %d folders to scan",
                        blob_prefix,
                        len(blobs.prefixes),
                    )

                    for prefix in blobs.prefixes:
                        folder_name = prefix.split("/")[-2]
                        expected_suffix = ".SAFE"
                        assert folder_name.endswith(expected_suffix)
                        item_name = folder_name.split(expected_suffix)[0]

                        # Make sure metadata XML blob exists, otherwise we won't be
                        # able to load the item.
                        # (Sometimes there is a .SAFE folder but some files like the
                        # XML file are just missing for whatever reason.)
                        xml_blob_path = f"{cell_folder}/{folder_name}/MTD_MSIL1C.xml"
                        xml_blob = self.bucket.blob(xml_blob_path)
                        if not xml_blob.exists():
                            logger.warning(
                                "no metadata XML for Sentinel-2 folder %s at %s",
                                folder_name,
                                xml_blob_path,
                            )
                            continue

                        item = self.get_item_by_name(item_name)
                        items.append(item)

                with open_atomic(cache_fname, "w") as f:
                    json.dump([item.serialize() for item in items], f)

            else:
                with cache_fname.open() as f:
                    items = [Sentinel2Item.deserialize(d) for d in json.load(f)]

            for item in items:
                yield item

    def _get_candidate_items_index(
        self, wgs84_geometries: list[STGeometry]
    ) -> list[list[Sentinel2Item]]:
        """List relevant items using rtree index."""
        candidates: list[list[Sentinel2Item]] = [[] for _ in wgs84_geometries]
        for idx, geometry in enumerate(wgs84_geometries):
            time_range = None
            if geometry.time_range:
                time_range = (
                    geometry.time_range[0] - self.max_time_delta,
                    geometry.time_range[1] + self.max_time_delta,
                )
            if self.rtree_index is None:
                raise ValueError("rtree_index is required")
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
    ) -> list[list[Sentinel2Item]]:
        """Use _read_products to list relevant items."""
        needed_cell_years = set()
        for wgs84_geometry in wgs84_geometries:
            if wgs84_geometry.time_range is None:
                raise ValueError(
                    "Sentinel2 on GCP requires geometry time ranges to be set"
                )
            for cell_id in get_sentinel2_tiles(wgs84_geometry, self.index_cache_dir):
                for year in range(
                    (wgs84_geometry.time_range[0] - self.max_time_delta).year,
                    (wgs84_geometry.time_range[1] + self.max_time_delta).year + 1,
                ):
                    needed_cell_years.add((cell_id, year))

        items_by_cell: dict[str, list[Sentinel2Item]] = {}
        for item in self._read_products(needed_cell_years):
            cell_id = "".join(item.blob_prefix.split("/")[1:4])
            assert len(cell_id) == 5
            if cell_id not in items_by_cell:
                items_by_cell[cell_id] = []
            items_by_cell[cell_id].append(item)

        candidates: list[list[Sentinel2Item]] = [[] for _ in wgs84_geometries]
        for idx, geometry in enumerate(wgs84_geometries):
            for cell_id in get_sentinel2_tiles(geometry, self.index_cache_dir):
                for item in items_by_cell.get(cell_id, []):
                    if not geometry.shp.intersects(item.geometry.shp):
                        continue
                    candidates[idx].append(item)

        return candidates

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[Sentinel2Item]]]:
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

    def deserialize_item(self, serialized_item: Any) -> Sentinel2Item:
        """Deserializes an item from JSON-decoded data."""
        assert isinstance(serialized_item, dict)
        return Sentinel2Item.deserialize(serialized_item)

    def retrieve_item(
        self, item: Sentinel2Item
    ) -> Generator[tuple[str, BinaryIO], None, None]:
        """Retrieves the rasters corresponding to an item as file streams."""
        for suffix, _ in self.BANDS:
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
        items: list[Sentinel2Item],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        Args:
            tile_store: the tile store to ingest into
            items: the items to ingest
            geometries: a list of geometries needed for each item
        """
        for item, cur_geometries in zip(items, geometries):
            harmonize_callback = None
            if self.harmonize:
                harmonize_callback = get_harmonize_callback(
                    self._get_xml_by_name(item.name)
                )

            for suffix, band_names in self.BANDS:
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
                            array_callback=harmonize_callback,
                        )
