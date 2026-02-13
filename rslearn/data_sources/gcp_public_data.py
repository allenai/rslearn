"""Data source for raster data on public Cloud Storage buckets."""

import io
import json
import os
import random
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, BinaryIO

import dateutil.parser
import rasterio
import shapely
import tqdm
from google.cloud import bigquery, storage
from upath import UPath

from rslearn.config import QueryConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import DataSource, DataSourceContext, Item
from rslearn.data_sources.utils import match_candidate_items_to_window
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.fsspec import join_upath, open_atomic
from rslearn.utils.geometry import STGeometry, flatten_shape, split_at_antimeridian
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import get_raster_projection_and_bounds

from .copernicus import get_harmonize_callback, get_sentinel2_tiles

logger = get_logger(__name__)


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


class CorruptItemException(Exception):
    """A Sentinel-2 scene is corrupted or otherwise unreadable for a known reason."""

    def __init__(self, message: str) -> None:
        """Create a new CorruptItemException.

        Args:
            message: error message.
        """
        self.message = message


class MissingXMLException(Exception):
    """Exception for when an item's XML file does not exist in GCS.

    Some items that appear in the index on BigQuery, or that have a folder, lack an XML
    file, and so in those cases this exception can be ignored.
    """

    def __init__(self, item_name: str):
        """Create a new MissingXMLException.

        Args:
            item_name: the name of the item (Sentinel-2 scene) that is missing its XML
                file in the GCS bucket.
        """
        self.item_name = item_name


@dataclass
class ParsedProductXML:
    """Result of parsing a Sentinel-2 product XML file."""

    blob_prefix: str
    shp: shapely.Polygon
    start_time: datetime
    cloud_cover: float


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

    # Possible prefixes of the product name that may appear on GCS, before the year
    # appears in the product name. For example, a product may start with
    # "S2A_MSIL1C_20230101..." so S2A_MSIL1C appears here. This list is used when
    # enumerating the list of products on GCS that fall in a certain year: because the
    # year comes after this prefix, filtering in the object list operation requires
    # including this prefix first followed by the year.
    VALID_PRODUCT_PREFIXES = ["S2A_MSIL1C", "S2B_MSIL1C", "S2C_MSIL1C"]

    # The name of the L1C product metadata XML file.
    METADATA_FILENAME = "MTD_MSIL1C.xml"

    def __init__(
        self,
        index_cache_dir: str,
        sort_by: str | None = None,
        use_rtree_index: bool = True,
        harmonize: bool = False,
        rtree_time_range: tuple[datetime, datetime] | None = None,
        rtree_cache_dir: str | None = None,
        use_bigquery: bool | None = None,
        bands: list[str] | None = None,
        context: DataSourceContext = DataSourceContext(),
    ):
        """Initialize a new Sentinel2 instance.

        Args:
            index_cache_dir: local directory to cache the index contents, as well as
                individual product metadata files.
            sort_by: can be "cloud_cover", default arbitrary order; only has effect for
                SpaceMode.WITHIN.
            use_rtree_index: whether to create an rtree index to enable faster lookups
                (default true). rtree will take several hours if it is not restricted
                to a short time range using rtree_time_range.
            harmonize: harmonize pixel values across different processing baselines,
                see https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
            rtree_time_range: only populate the rtree index with scenes within this
                time range. Restricting to a few months significantly speeds up rtree
                creation time.
            rtree_cache_dir: by default, if use_rtree_index is enabled, the rtree is
                stored in index_cache_dir (where product XML files are also stored). If
                rtree_cache_dir is set, then the rtree is stored here instead (so
                index_cache_dir is only used to cache product XML files).
            use_bigquery: whether to use the BigQuery index over the scenes in the
                bucket. This must be enabled if use_rtree_index is enabled, since we
                only support populating the rtree index from BigQuery. Note that
                BigQuery requires GCP credentials to be setup; to avoid the need for
                credentials, set use_bigquery=False and use_rtree_index=False. The
                default value is None which enables BigQuery when use_rtree_index=True
                and disables when use_rtree_index=False.
            bands: the bands to download, or None to download all bands. This is only
                used if the layer config is not in the context.
            context: the data source context.
        """
        if use_bigquery is None:
            use_bigquery = use_rtree_index
        if not use_bigquery and use_rtree_index:
            raise ValueError(
                "use_bigquery must be enabled if use_rtree_index is enabled"
            )

        # Resolve index_cache_dir and rtree_cache_dir depending on dataset context.
        if context.ds_path is not None:
            self.index_cache_dir = join_upath(context.ds_path, index_cache_dir)
        else:
            self.index_cache_dir = UPath(index_cache_dir)

        if rtree_cache_dir is None:
            self.rtree_cache_dir = self.index_cache_dir
        elif context.ds_path is not None:
            self.rtree_cache_dir = join_upath(context.ds_path, rtree_cache_dir)
        else:
            self.rtree_cache_dir = UPath(rtree_cache_dir)

        self.sort_by = sort_by
        self.harmonize = harmonize
        self.use_bigquery = use_bigquery

        self.index_cache_dir.mkdir(parents=True, exist_ok=True)

        # Determine the subset of bands that are needed based on the layer config.
        self.needed_bands: list[tuple[str, list[str]]]
        if context.layer_config is not None:
            self.needed_bands = []
            for fname, cur_bands in self.BANDS:
                # See if the bands provided by this file intersect with the bands in at
                # least one configured band set.
                for band_set in context.layer_config.band_sets:
                    if not set(band_set.bands).intersection(cur_bands):
                        continue
                    self.needed_bands.append((fname, cur_bands))
                    break
        elif bands is not None:
            self.needed_bands = []
            for fname, cur_bands in self.BANDS:
                if not set(bands).intersection(cur_bands):
                    continue
                self.needed_bands.append((fname, cur_bands))
        else:
            self.needed_bands = list(self.BANDS)

        self.bucket = storage.Client.create_anonymous_client().bucket(self.BUCKET_NAME)
        self.rtree_index: Any | None = None
        if use_rtree_index:
            from rslearn.utils.rtree_index import RtreeIndex, get_cached_rtree

            self.rtree_cache_dir.mkdir(parents=True, exist_ok=True)

            def build_fn(index: RtreeIndex) -> None:
                """Build the RtreeIndex from items in the data source."""
                for item in self._read_bigquery(
                    desc="Building rtree index", time_range=rtree_time_range
                ):
                    for shp in flatten_shape(item.geometry.shp):
                        index.insert(shp.bounds, json.dumps(item.serialize()))

            self.rtree_index = get_cached_rtree(self.rtree_cache_dir, build_fn)

    def _read_bigquery(
        self,
        desc: str | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        wgs84_bbox: tuple[float, float, float, float] | None = None,
    ) -> Generator[Sentinel2Item, None, None]:
        """Read Sentinel-2 scenes from BigQuery table.

        The table only contains the bounding box of each image and not the exact
        geometry, which can be retrieved from individual product metadata
        (MTD_MSIL1C.xml) files.

        Args:
            desc: description to include with tqdm progress bar.
            time_range: optional time_range to restrict the reading.
            wgs84_bbox: optional bounding box in WGS-84 coordinates to restrict the
                reading.
        """
        query_str = f"""
            SELECT  source_url, base_url, product_id, sensing_time, granule_id,
                    east_lon, south_lat, west_lon, north_lat, cloud_cover
            FROM    `{self.TABLE_NAME}`
        """
        clauses = []
        if time_range is not None:
            clauses.append(f"""(
                sensing_time >= "{time_range[0]}" AND sensing_time <= "{time_range[1]}"
            )""")
        if wgs84_bbox is not None:
            clauses.append(f"""(
                west_lon < {wgs84_bbox[2]} AND
                east_lon > {wgs84_bbox[0]} AND
                south_lat < {wgs84_bbox[3]} AND
                north_lat > {wgs84_bbox[1]}
            )""")
        if clauses:
            query_str += " WHERE " + " AND ".join(clauses)

        client = bigquery.Client()
        result = client.query(query_str)
        if desc is not None:
            result = tqdm.tqdm(result, desc=desc)

        for row in result:
            # Validate product ID has correct number of sections and that it is MSIL1C.
            # Example product IDs:
            # - S2B_MSIL1C_20180210T200549_N0206_R128_T08VPK_20180210T215722
            # - S2A_OPER_PRD_MSIL1C_PDMC_20160315T180002_R091_V20160315T060423_20160315T060423
            # We must do this before checking source_url because we want to skip the
            # products that say OPER instead of MSIL1C (occasionally the OPER products
            # are missing other fields in the CSV).
            # For example, the OPER product above has:
            # - source_url = https://storage.googleapis.com/gcp-public-data-sentinel-2/index.csv.gz
            # - base_url = None
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

            # Figure out what the product folder is for this entry.
            # Some entries have source_url correct and others have base_url correct.
            # If base_url is correct, then it seems the source_url always ends in
            # index.csv.gz.
            # Example 1:
            # - source_url = https://storage.googleapis.com/gcp-public-data-sentinel-2/index.csv.gz
            # - base_url = gs://gcp-public-data-sentinel-2/tiles/54/U/VV/S2A_MSIL1C_20160219T015301_N0201_R017_T54UVV_20160222T152042.SAFE
            # Example 2:
            # - source_url = gs://gcp-public-data-sentinel-2/tiles/15/C/WM/S2B_MSIL1C_20250101T121229_N0511_R080_T15CWM_20250101T150509.SAFE
            # - base_url = None
            if row["source_url"] and not row["source_url"].endswith("index.csv.gz"):
                product_folder = row["source_url"].split(f"gs://{self.BUCKET_NAME}/")[1]
            elif row["base_url"] is not None and row["base_url"] != "":
                product_folder = row["base_url"].split(f"gs://{self.BUCKET_NAME}/")[1]
            else:
                raise ValueError(
                    f"Unexpected value '{row['source_url']}' in column 'source_url'"
                    + f" and '{row['base_url']} in column 'base_url'"
                    + f"for product {row['product_id']}"
                )

            # Build the blob prefix based on the product ID and granule ID.
            # The blob prefix is the prefix to the JP2 image files on GCS.
            granule_id = row["granule_id"]
            blob_prefix = (
                f"{product_folder}/GRANULE/{granule_id}/IMG_DATA/{tile_id}_{time_str}_"
            )

            # Extract the spatial and temporal bounds of the image.
            bounds = (
                float(row["west_lon"]),
                float(row["south_lat"]),
                float(row["east_lon"]),
                float(row["north_lat"]),
            )
            shp = shapely.box(*bounds)
            sensing_time = row["sensing_time"]
            geometry = STGeometry(WGS84_PROJECTION, shp, (sensing_time, sensing_time))
            geometry = split_at_antimeridian(geometry)

            cloud_cover = float(row["cloud_cover"])

            yield Sentinel2Item(product_id, geometry, blob_prefix, cloud_cover)

    def _build_cell_folder_name(self, cell_id: str) -> str:
        """Get the prefix on GCS containing the product files in the provided cell.

        The Sentinel-2 cell ID is based on MGRS and is a way of splitting up the world
        into large tiles.

        Args:
            cell_id: the 5-character cell ID. Note that the product name includes the
                cell ID with a "T" prefix, the T should be removed.

        Returns:
            the path on GCS of the folder corresponding to this Sentinel-2 cell.
        """
        return f"tiles/{cell_id[0:2]}/{cell_id[2:3]}/{cell_id[3:5]}/"

    def _build_product_folder_name(self, item_name: str) -> str:
        """Get the folder containing the given Sentinel-2 scene ID on GCS.

        Args:
            item_name: the item name (Sentinel-2 scene ID).

        Returns:
            the path on GCS of the .SAFE folder corresponding to this item.
        """
        parts = item_name.split("_")
        cell_id_with_prefix = parts[5]
        if len(cell_id_with_prefix) != 6:
            raise ValueError(
                f"cell ID should be 6 characters but got {cell_id_with_prefix}"
            )
        if cell_id_with_prefix[0] != "T":
            raise ValueError(
                f"cell ID should start with T but got {cell_id_with_prefix}"
            )
        cell_id = cell_id_with_prefix[1:]
        return self._build_cell_folder_name(cell_id) + f"{item_name}.SAFE/"

    def _get_xml_by_name(self, name: str) -> "ET.ElementTree[ET.Element[str]]":
        """Gets the metadata XML of an item by its name.

        Args:
            name: the name of the item

        Returns:
            the parsed XML ElementTree
        """
        cache_xml_fname = self.index_cache_dir / (name + ".xml")
        if not cache_xml_fname.exists():
            product_folder = self._build_product_folder_name(name)
            metadata_blob_path = product_folder + self.METADATA_FILENAME
            logger.debug("reading metadata XML from %s", metadata_blob_path)
            blob = self.bucket.blob(metadata_blob_path)
            if not blob.exists():
                raise MissingXMLException(name)
            with open_atomic(cache_xml_fname, "wb") as f:
                blob.download_to_file(f)

        with cache_xml_fname.open("rb") as f:
            return ET.parse(f)

    def _parse_xml(self, name: str) -> ParsedProductXML:
        """Parse a Sentinel-2 product XML file.

        This extracts the blob prefix in the GCS bucket, the polygon extent, sensing
        start time, and cloud cover.

        Args:
            name: the Sentinel-2 scene name.
        """
        # Get the XML. This helper function handles caching the XML file.
        tree = self._get_xml_by_name(name)

        # Now parse the XML, starting with the detailed geometry of the image.
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

        # Get blob prefix which is a subfolder of the product folder.
        # The blob prefix is the prefix to the JP2 image files on GCS.
        product_folder = self._build_product_folder_name(name)
        elements = list(tree.iter("IMAGE_FILE"))
        elements = [
            el for el in elements if el.text is not None and el.text.endswith("_B01")
        ]
        assert len(elements) == 1
        if elements[0].text is None:
            raise ValueError(f"IMAGE_FILE is empty for {name}")
        blob_prefix = product_folder + elements[0].text.split("B01")[0]

        # Get the sensing start time.
        elements = list(tree.iter("PRODUCT_START_TIME"))
        assert len(elements) == 1
        if elements[0].text is None:
            raise ValueError(f"PRODUCT_START_TIME is empty for {name}")
        start_time = dateutil.parser.isoparse(elements[0].text)

        # Get the cloud cover.
        elements = list(tree.iter("Cloud_Coverage_Assessment"))
        assert len(elements) == 1
        if elements[0].text is None:
            raise ValueError(f"Cloud_Coverage_Assessment is empty for {name}")
        cloud_cover = float(elements[0].text)

        return ParsedProductXML(
            blob_prefix=blob_prefix,
            shp=shp,
            start_time=start_time,
            cloud_cover=cloud_cover,
        )

    def _get_item_by_name(self, name: str) -> Sentinel2Item:
        """Gets an item by name.

        This implements the main logic of processing the product metadata file
        without the caching logic in get_item_by_name, see that function for details.

        Args:
            name: the Sentinel-2 scene ID.
        """
        product_xml = self._parse_xml(name)

        # Some Sentinel-2 scenes in the bucket are missing a subset of image files. So
        # here we verify that all the bands we know about are intact.
        expected_suffixes = {t[0] for t in self.BANDS}
        for blob in self.bucket.list_blobs(prefix=product_xml.blob_prefix):
            assert blob.name.startswith(product_xml.blob_prefix)
            suffix = blob.name[len(product_xml.blob_prefix) :]
            if suffix in expected_suffixes:
                expected_suffixes.remove(suffix)
        if len(expected_suffixes) > 0:
            raise CorruptItemException(
                f"item is missing image files: {expected_suffixes}"
            )

        time_range = (product_xml.start_time, product_xml.start_time)
        geometry = STGeometry(WGS84_PROJECTION, product_xml.shp, time_range)
        geometry = split_at_antimeridian(geometry)

        # Sometimes the geometry is not valid.
        # We just apply make_valid on it to correct issues.
        if not geometry.shp.is_valid:
            geometry.shp = shapely.make_valid(geometry.shp)

        # Some rasters have zero-area geometry due to incorrect geometry. For example,
        # S2B_MSIL1C_20190111T193659_N0207_R056_T08MLS_20190111T205033.SAFE.
        # So here we add a check for that and mark it corrupt if so.
        if geometry.shp.area == 0:
            raise CorruptItemException(
                f"XML for item {name} shows geometry with zero area"
            )

        return Sentinel2Item(
            name=name,
            geometry=geometry,
            blob_prefix=product_xml.blob_prefix,
            cloud_cover=product_xml.cloud_cover,
        )

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
        # The main logic for getting the item is implemented in _get_item_by_name.
        # Here, we implement caching logic so that, if we have already seen this item
        # before, then we can just deserialize it from a JSON file.
        # We want to cache the item if it is successful, but also cache the
        # CorruptItemException if it is raised.
        cache_item_fname = self.index_cache_dir / (name + ".json")

        if cache_item_fname.exists():
            with cache_item_fname.open() as f:
                d = json.load(f)

            if "error" in d:
                raise CorruptItemException(d["error"])

            return Sentinel2Item.deserialize(d)

        try:
            item = self._get_item_by_name(name)
        except CorruptItemException as e:
            with open_atomic(cache_item_fname, "w") as f:
                json.dump({"error": e.message}, f)
            raise

        with open_atomic(cache_item_fname, "w") as f:
            json.dump(item.serialize(), f)
        return item

    def _read_products_for_cell_year(
        self, cell_id: str, year: int
    ) -> list[Sentinel2Item]:
        """Read items for the given cell and year directly from the GCS bucket.

        This helper function is used by self._read_products which then caches the
        items together in one file.
        """
        items = []

        for product_prefix in self.VALID_PRODUCT_PREFIXES:
            cell_folder = self._build_cell_folder_name(cell_id)
            blob_prefix = f"{cell_folder}{product_prefix}_{year}"
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

                try:
                    item = self.get_item_by_name(item_name)
                except CorruptItemException as e:
                    logger.warning("skipping corrupt item %s: %s", item_name, e.message)
                    continue
                except MissingXMLException:
                    # Sometimes there is a .SAFE folder but some files like the
                    # XML file are just missing for whatever reason. Since we
                    # know this happens occasionally, we just ignore the error
                    # here.
                    logger.warning(
                        "no metadata XML for Sentinel-2 folder %s/%s",
                        blob_prefix,
                        folder_name,
                    )
                    continue

                items.append(item)

        return items

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
                items = self._read_products_for_cell_year(cell_id, year)
                with open_atomic(cache_fname, "w") as f:
                    json.dump([item.serialize() for item in items], f)

            else:
                with cache_fname.open() as f:
                    items = [Sentinel2Item.deserialize(d) for d in json.load(f)]

            yield from items

    def _get_candidate_items_index(
        self, wgs84_geometries: list[STGeometry]
    ) -> list[list[Sentinel2Item]]:
        """List relevant items using rtree index.

        Args:
            wgs84_geometries: the geometries to query.
        """
        candidates: list[list[Sentinel2Item]] = [[] for _ in wgs84_geometries]
        for idx, geometry in enumerate(wgs84_geometries):
            time_range = None
            if geometry.time_range:
                time_range = (
                    geometry.time_range[0],
                    geometry.time_range[1],
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

                # Get the item from XML to get its exact geometry (the index only
                # knows the bounding box of the item).
                try:
                    item = self.get_item_by_name(item.name)
                except CorruptItemException as e:
                    logger.warning("skipping corrupt item %s: %s", item.name, e.message)
                    continue
                except MissingXMLException:
                    # Sometimes a scene that appears in the BigQuery index does not
                    # actually have an XML file on GCS. Since we know this happens
                    # occasionally, we ignore the error here.
                    logger.warning(
                        "skipping item %s that is missing XML file", item.name
                    )
                    continue

                if not item.geometry.shp.intersects(geometry.shp):
                    continue
                candidates[idx].append(item)
        return candidates

    def _get_candidate_items_direct(
        self, wgs84_geometries: list[STGeometry]
    ) -> list[list[Sentinel2Item]]:
        """Use _read_products to list matching items directly from the bucket.

        Args:
            wgs84_geometries: the geometries to query.
        """
        needed_cell_years = set()
        for wgs84_geometry in wgs84_geometries:
            if wgs84_geometry.time_range is None:
                raise ValueError(
                    "Sentinel2 on GCP requires geometry time ranges to be set"
                )
            for cell_id in get_sentinel2_tiles(wgs84_geometry, self.index_cache_dir):
                for year in range(
                    wgs84_geometry.time_range[0].year,
                    wgs84_geometry.time_range[1].year + 1,
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

    def _get_candidate_items_bigquery(
        self, wgs84_geometries: list[STGeometry]
    ) -> list[list[Sentinel2Item]]:
        """Use _read_bigquery to list matching items by querying the BigQuery table.

        Args:
            wgs84_geometries: the geometries to query.
        """
        candidates: list[list[Sentinel2Item]] = [[] for _ in wgs84_geometries]
        for idx, geometry in enumerate(wgs84_geometries):
            wgs84_bbox = geometry.shp.bounds
            for item in self._read_bigquery(
                time_range=geometry.time_range, wgs84_bbox=wgs84_bbox
            ):
                # Get the item from XML to get its exact geometry (BigQuery only knows
                # the bounding box of the item).
                try:
                    item = self.get_item_by_name(item.name)
                except CorruptItemException as e:
                    logger.warning("skipping corrupt item %s: %s", item.name, e.message)
                    continue
                except MissingXMLException:
                    # Sometimes a scene that appears in the BigQuery index does not
                    # actually have an XML file on GCS. Since we know this happens
                    # occasionally, we ignore the error here.
                    logger.warning(
                        "skipping item %s that is missing XML file", item.name
                    )
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
        elif self.use_bigquery:
            candidates = self._get_candidate_items_bigquery(wgs84_geometries)
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

    def deserialize_item(self, serialized_item: dict) -> Sentinel2Item:
        """Deserializes an item from JSON-decoded data."""
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
        tile_store: TileStoreWithLayer,
        items: list[Sentinel2Item],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        Args:
            tile_store: the tile store to ingest into
            items: the items to ingest
            geometries: a list of geometries needed for each item
        """
        for item in items:
            for suffix, band_names in self.needed_bands:
                if tile_store.is_raster_ready(item.name, band_names):
                    continue

                with tempfile.TemporaryDirectory() as tmp_dir:
                    fname = os.path.join(tmp_dir, suffix)
                    blob = self.bucket.blob(item.blob_prefix + suffix)
                    logger.debug(
                        "gcp_public_data downloading raster file %s",
                        item.blob_prefix + suffix,
                    )
                    blob.download_to_filename(fname)
                    logger.debug(
                        "gcp_public_data ingesting raster file %s into tile store",
                        item.blob_prefix + suffix,
                    )

                    # Harmonize values if needed.
                    # TCI does not need harmonization.
                    harmonize_callback = None
                    if self.harmonize and suffix != "TCI.jp2":
                        harmonize_callback = get_harmonize_callback(
                            self._get_xml_by_name(item.name)
                        )

                    if harmonize_callback is not None:
                        # In this case we need to read the array, convert the pixel
                        # values, and pass modified array directly to the TileStore.
                        with rasterio.open(fname) as src:
                            array = src.read()
                            projection, bounds = get_raster_projection_and_bounds(src)
                        array = harmonize_callback(array)
                        tile_store.write_raster(
                            item.name,
                            band_names,
                            projection,
                            bounds,
                            RasterArray(
                                chw_array=array,
                                time_range=item.geometry.time_range,
                            ),
                        )

                    else:
                        tile_store.write_raster_file(
                            item.name, band_names, UPath(fname)
                        )

                logger.debug(
                    "gcp_public_data done ingesting raster file %s",
                    item.blob_prefix + suffix,
                )
