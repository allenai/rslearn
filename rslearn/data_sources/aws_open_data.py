"""Data source for raster data in Registry of Open Data on AWS."""

import io
import json
import xml.etree.ElementTree as ET
from collections.abc import Callable, Generator
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, BinaryIO

import boto3
import dateutil.parser
import fiona
import fiona.transform
import numpy.typing as npt
import pytimeparse
import rasterio
import shapely
import tqdm
from rasterio.crs import CRS
from upath import UPath

import rslearn.data_sources.utils
from rslearn.config import RasterLayerConfig
from rslearn.const import SHAPEFILE_AUX_EXTENSIONS, WGS84_EPSG, WGS84_PROJECTION
from rslearn.tile_stores import PrefixedTileStore, TileStore
from rslearn.utils import GridIndex, Projection, STGeometry, daterange
from rslearn.utils.fsspec import get_upath_local, join_upath, open_atomic

from .copernicus import get_harmonize_callback, get_sentinel2_tiles
from .data_source import (
    DataSource,
    Item,
    ItemLookupDataSource,
    QueryConfig,
    RetrieveItemDataSource,
)
from .raster_source import get_needed_projections, ingest_raster


class NaipItem(Item):
    """An item in the Naip data source."""

    def __init__(self, name: str, geometry: STGeometry, blob_path: str):
        """Creates a new NaipItem.

        Args:
            name: unique name of the item
            geometry: the spatial and temporal extent of the item
            blob_path: path in bucket
        """
        super().__init__(name, geometry)
        self.blob_path = blob_path

    def serialize(self) -> dict:
        """Serializes the item to a JSON-encodable dictionary."""
        d = super().serialize()
        d["blob_path"] = self.blob_path
        return d

    @staticmethod
    def deserialize(d: dict) -> "NaipItem":
        """Deserializes an item from a JSON-decoded dictionary."""
        item = super(NaipItem, NaipItem).deserialize(d)
        return NaipItem(
            name=item.name, geometry=item.geometry, blob_path=d["blob_path"]
        )


class Naip(DataSource):
    """A data source for NAIP imagery on AWS.

    Specifically uses the naip-source requester pays bucket maintained by Esri. See
    https://registry.opendata.aws/naip/ for more information.

    AWS credentials must be configured for use with boto3.
    """

    bucket_name = "naip-source"

    manifest_fname = "manifest.txt"

    def __init__(
        self,
        config: RasterLayerConfig,
        index_cache_dir: UPath,
        use_rtree_index: bool = False,
        states: list[str] | None = None,
        years: list[int] | None = None,
    ) -> None:
        """Initialize a new Naip instance.

        Args:
            config: the LayerConfig of the layer containing this data source.
            index_cache_dir: directory to cache index shapefiles.
            use_rtree_index: whether to create an rtree index to enable faster lookups
                (default false)
            states: optional list of states (lowercase two-letter codes) to restrict
                the search. If use_rtree_index is enabled, the rtree will only be
                populated with data from these states.
            years: optional list of years to restrict the search
        """
        self.config = config
        self.index_cache_dir = index_cache_dir
        self.states = states
        self.years = years

        self.bucket = boto3.resource("s3").Bucket(self.bucket_name)
        self.rtree_index: Any | None = None
        if use_rtree_index:
            from rslearn.utils.rtree_index import RtreeIndex, get_cached_rtree

            def build_fn(index: RtreeIndex) -> None:
                for item in self._read_index_shapefiles(desc="Building rtree index"):
                    index.insert(item.geometry.shp.bounds, json.dumps(item.serialize()))

            self.rtree_index = get_cached_rtree(self.index_cache_dir, build_fn)

    @staticmethod
    def from_config(config: RasterLayerConfig, ds_path: UPath) -> "Naip":
        """Creates a new Naip instance from a configuration dictionary."""
        if config.data_source is None:
            raise ValueError(f"data_source is required for config dict {config}")
        d = config.data_source.config_dict
        kwargs = dict(
            config=config,
            index_cache_dir=join_upath(ds_path, d["index_cache_dir"]),
        )
        if "use_rtree_index" in d:
            kwargs["use_rtree_index"] = d["use_rtree_index"]
        return Naip(**kwargs)

    def _download_manifest(self) -> UPath:
        """Download the manifest that enumerates files in the bucket.

        Returns:
            The local filename where the manifest has been downloaded.
        """
        manifest_path = self.index_cache_dir / self.manifest_fname
        if not manifest_path.exists():
            with manifest_path.open("wb") as dst:
                self.bucket.download_fileobj(
                    self.manifest_fname,
                    dst,
                    ExtraArgs={"RequestPayer": "requester"},
                )
        return manifest_path

    def _download_index_shapefiles(self) -> None:
        """Download all index shapefiles that specify image extents."""
        manifest_path = self._download_manifest()
        needed_files: list[tuple[str, UPath]] = []
        with manifest_path.open() as f:
            for line in f:
                blob_path = line.strip()
                if not blob_path:
                    continue
                if "/index/" not in blob_path:
                    continue

                # Data before 2012 doesn't seem to be present even though index files
                # might exist?
                # Also filter state/year here.
                path_parts = blob_path.split("/")
                state_code = path_parts[0]
                year = int(path_parts[1])
                if year < 2012:
                    continue
                if self.states and state_code not in self.states:
                    continue
                if self.years and year not in self.years:
                    continue

                cache_path = self.index_cache_dir / blob_path
                if cache_path.exists():
                    continue
                needed_files.append((blob_path, cache_path))
        for blob_path, cache_path in tqdm.tqdm(
            needed_files, desc="Downloading index files"
        ):
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open_atomic(cache_path, "wb") as dst:
                self.bucket.download_fileobj(
                    blob_path, dst, ExtraArgs={"RequestPayer": "requester"}
                )

    def _read_index_shapefiles(
        self, desc: str | None = None
    ) -> Generator[NaipItem, None, None]:
        """Read the index shapefiles and yield NaipItems corresponding to each image."""
        self._download_index_shapefiles()

        # Sometimes we see files like m_3410458_sw_13_060_20180612_20190426.tif but
        # the actual name is m_3410458_sw_13_060_20180612_20190427.tif (last part, the
        # version date, is different).
        # Also sometimes it is just m_4212362_sw_10_1_20140622.tif (ending in capture
        # date).
        # So here we create dict from just the prefix (i.e. exclude the version date).
        tif_blob_path_dict = {}
        manifest_path = self._download_manifest()
        with manifest_path.open() as f:
            for line in f:
                blob_path = line.strip()
                if not blob_path:
                    continue
                if not blob_path.endswith(".tif"):
                    continue
                parts = blob_path.split(".tif")[0].split("_")
                prefix = "_".join(parts[0:6])
                tif_blob_path_dict[prefix] = blob_path

        shape_files = self.index_cache_dir.glob("**/*.shp")
        if desc:
            shape_files = tqdm.tqdm(shape_files, desc=desc)
        for shp_fname in shape_files:
            # We copy the file to local since it needs to read all the auxiliary files
            # of the shapefile.
            prefix = ".".join(shp_fname.name.split(".")[:-1])
            aux_files: list[UPath] = []
            for ext in SHAPEFILE_AUX_EXTENSIONS:
                aux_files.append(shp_fname.parent / (prefix + ext))

            with get_upath_local(shp_fname, extra_paths=aux_files) as local_fname:
                with fiona.open(local_fname) as f:
                    src_crs = f.crs
                    dst_crs = fiona.crs.from_epsg(WGS84_EPSG)

                    for feature in f:
                        geometry = fiona.transform.transform_geom(
                            src_crs, dst_crs, feature["geometry"]
                        )
                        shp = shapely.geometry.shape(geometry)

                        # Properties specifies TIF filename like:
                        # - m_4212362_sw_10_1_20140622_20140923.tif.
                        # Index fname is like:
                        # - ca/2014/100cm/index/naip_3_14_3_1_ca.shp.
                        # So use that to reconstruct the prefix:
                        # - ca/2014/100cm/index/rgbir/m_4212362_sw_10_1_20140622
                        base_dir = "/".join(
                            shp_fname.parent.parent.path.split("/")[-3:]
                        )
                        tif_fname = feature["properties"]["FileName"]
                        fname_parts = tif_fname.split(".tif")[0].split("_")
                        tile_id = fname_parts[1][0:5]
                        fname_prefix = "_".join(fname_parts[:-1])
                        full_prefix = f"{base_dir}/rgbir/{tile_id}/{fname_prefix}"

                        if full_prefix not in tif_blob_path_dict:
                            print(
                                f"warning: skipping file {tif_fname} seen in "
                                + f"shapefile {shp_fname} but {full_prefix} does not exist "
                                + "in manifest"
                            )
                            continue
                        blob_path = tif_blob_path_dict[full_prefix]

                        # SrcImgDate is either string like "20180905" or int 20180905.
                        # We make sure it is string here.
                        # But it also could not exist in which case we need to fallback to
                        # extracting it from filename.
                        if "SrcImgDate" in feature["properties"]:
                            src_img_date = feature["properties"]["SrcImgDate"]
                            if isinstance(src_img_date, int):
                                src_img_date = str(src_img_date)
                        else:
                            src_img_date = fname_parts[5]
                        time = datetime.strptime(src_img_date, "%Y%m%d").replace(
                            tzinfo=timezone.utc
                        )

                        geometry = STGeometry(WGS84_PROJECTION, shp, (time, time))

                        yield NaipItem(
                            name=blob_path.split("/")[-1].split(".tif")[0],
                            geometry=geometry,
                            blob_path=blob_path,
                        )

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[NaipItem]]]:
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

        items: list = [[] for _ in geometries]
        if self.rtree_index:
            for idx, geometry in enumerate(wgs84_geometries):
                encoded_items = self.rtree_index.query(geometry.shp.bounds)
                for encoded_item in encoded_items:
                    item = NaipItem.deserialize(json.loads(encoded_item))
                    if not item.geometry.shp.intersects(geometry.shp):
                        continue
                    items[idx].append(item)
        else:
            index = GridIndex(0.01)
            for idx, geometry in enumerate(wgs84_geometries):
                index.insert(geometry.shp.bounds, idx)
            for item in self._read_index_shapefiles():
                results = index.query(item.geometry.shp.bounds)
                for idx in results:
                    geometry = wgs84_geometries[idx]
                    if not geometry.shp.intersects(item.geometry.shp):
                        continue
                    items[idx].append(item)

        groups = []
        for geometry, item_list in zip(wgs84_geometries, items):
            cur_groups = rslearn.data_sources.utils.match_candidate_items_to_window(
                geometry, item_list, query_config
            )
            groups.append(cur_groups)
        return groups

    def deserialize_item(self, serialized_item: Any) -> NaipItem:
        """Deserializes an item from JSON-decoded data."""
        assert isinstance(serialized_item, dict)
        return NaipItem.deserialize(serialized_item)

    def ingest(
        self,
        tile_store: TileStore,
        items: list[NaipItem],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        Args:
            tile_store: the tile store to ingest into
            items: the items to ingest
            geometries: a list of geometries needed for each item
        """
        for item, cur_geometries in zip(items, geometries):
            bands = ["R", "G", "B", "IR"]
            cur_tile_store = PrefixedTileStore(tile_store, (item.name, "_".join(bands)))
            needed_projections = get_needed_projections(
                cur_tile_store, bands, self.config.band_sets, cur_geometries
            )
            if not needed_projections:
                continue

            buf = io.BytesIO()
            self.bucket.download_fileobj(
                item.blob_path, buf, ExtraArgs={"RequestPayer": "requester"}
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


class Sentinel2Modality(Enum):
    """Sentinel-2 modality, either L1C or L2A."""

    L1C = "L1C"
    L2A = "L2A"


class Sentinel2Item(Item):
    """An item in the Sentinel2 data source."""

    def __init__(
        self, name: str, geometry: STGeometry, blob_path: str, cloud_cover: float
    ):
        """Creates a new Sentinel2Item.

        Args:
            name: unique name of the item
            geometry: the spatial and temporal extent of the item
            blob_path: path in bucket, e.g. tiles/51/C/WM/2024/2/1/0/
            cloud_cover: the scene's cloud cover
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
    def deserialize(d: dict) -> "Sentinel2Item":
        """Deserializes an item from a JSON-decoded dictionary."""
        if "name" not in d:
            d["name"] = d["blob_path"].split("/")[-1].split(".tif")[0]
        item = super(Sentinel2Item, Sentinel2Item).deserialize(d)
        return Sentinel2Item(
            name=item.name,
            geometry=item.geometry,
            blob_path=d["blob_path"],
            cloud_cover=d["cloud_cover"],
        )


# TODO: Distinguish between AWS and GCP data sources in class names.
class Sentinel2(
    ItemLookupDataSource[Sentinel2Item], RetrieveItemDataSource[Sentinel2Item]
):
    """A data source for Sentinel-2 L1C and L2A imagery on AWS.

    Specifically, uses the sentinel-s2-l1c and sentinel-s2-l2a S3 buckets maintained by
    Sinergise. They state the data is "added regularly, usually within few hours after
    they are available on Copernicus OpenHub".

    See https://aws.amazon.com/marketplace/pp/prodview-2ostsvrguftb2 for details about
    the buckets.

    AWS credentials must be configured for use with boto3.
    """

    bucket_names = {
        Sentinel2Modality.L1C: "sentinel-s2-l1c",
        Sentinel2Modality.L2A: "sentinel-s2-l2a",
    }

    band_fnames = {
        Sentinel2Modality.L1C: [
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
        ],
        Sentinel2Modality.L2A: [
            ("R20m/B01.jp2", ["B01"]),
            ("R10m/B02.jp2", ["B02"]),
            ("R10m/B03.jp2", ["B03"]),
            ("R10m/B04.jp2", ["B04"]),
            ("R20m/B05.jp2", ["B05"]),
            ("R20m/B06.jp2", ["B06"]),
            ("R20m/B07.jp2", ["B07"]),
            ("R10m/B08.jp2", ["B08"]),
            ("R60m/B09.jp2", ["B09"]),
            ("R20m/B11.jp2", ["B11"]),
            ("R20m/B12.jp2", ["B12"]),
            ("R20m/B8A.jp2", ["B8A"]),
            ("R10m/TCI.jp2", ["R", "G", "B"]),
        ],
    }

    def __init__(
        self,
        config: RasterLayerConfig,
        modality: Sentinel2Modality,
        metadata_cache_dir: UPath,
        max_time_delta: timedelta = timedelta(days=30),
        sort_by: str | None = None,
        harmonize: bool = False,
    ) -> None:
        """Initialize a new Sentinel2 instance.

        Args:
            config: the LayerConfig of the layer containing this data source.
            modality: L1C or L2A.
            metadata_cache_dir: directory to cache product metadata files.
            max_time_delta: maximum time before a query start time or after a
                query end time to look for products. This is required due to the large
                number of available products, and defaults to 30 days.
            sort_by: can be "cloud_cover", default arbitrary order; only has effect for
                SpaceMode.WITHIN.
            harmonize: harmonize pixel values across different processing baselines,
                see https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
        """  # noqa: E501
        self.config = config
        self.modality = modality
        self.metadata_cache_dir = metadata_cache_dir
        self.max_time_delta = max_time_delta
        self.sort_by = sort_by
        self.harmonize = harmonize

        bucket_name = self.bucket_names[modality]
        self.bucket = boto3.resource("s3").Bucket(bucket_name)

    @staticmethod
    def from_config(config: RasterLayerConfig, ds_path: UPath) -> "Sentinel2":
        """Creates a new Sentinel2 instance from a configuration dictionary."""
        if config.data_source is None:
            raise ValueError("Sentinel2 data source requires a data source config")
        d = config.data_source.config_dict
        kwargs = dict(
            config=config,
            modality=Sentinel2Modality(d["modality"]),
            metadata_cache_dir=join_upath(ds_path, d["metadata_cache_dir"]),
        )

        if "max_time_delta" in d:
            kwargs["max_time_delta"] = timedelta(
                seconds=pytimeparse.parse(d["max_time_delta"])
            )
        simple_optionals = ["sort_by", "harmonize"]
        for k in simple_optionals:
            if k in d:
                kwargs[k] = d[k]

        return Sentinel2(**kwargs)

    def _read_products(
        self, needed_cell_months: set[tuple[str, int, int]]
    ) -> Generator[Sentinel2Item, None, None]:
        """Read productInfo.json files and yield relevant Sentinel2Items.

        Args:
            needed_cell_months: set of (mgrs grid cell, year, month) where we need
                to search for images.
        """
        for cell_id, year, month in tqdm.tqdm(
            needed_cell_months, desc="Reading product infos"
        ):
            assert len(cell_id) == 5
            cache_fname = self.metadata_cache_dir / f"{cell_id}_{year}_{month}.json"

            if not cache_fname.exists():
                cell_part1 = cell_id[0:2]
                cell_part2 = cell_id[2:3]
                cell_part3 = cell_id[3:5]
                prefix = (
                    f"tiles/{cell_part1}/{cell_part2}/{cell_part3}/"
                    + f"{year}/{month}/"
                )

                products = []
                for obj in self.bucket.objects.filter(
                    Prefix=prefix, RequestPayer="requester"
                ):
                    if not obj.key.endswith("tileInfo.json"):
                        continue
                    buf = io.BytesIO()
                    self.bucket.download_fileobj(
                        obj.key, buf, ExtraArgs={"RequestPayer": "requester"}
                    )
                    buf.seek(0)
                    product = json.load(buf)
                    if "tileDataGeometry" not in product:
                        print(
                            "warning: skipping product missing tileDataGeometry",
                            product,
                        )
                        continue
                    if product["tileDataGeometry"]["type"] != "Polygon":
                        print(
                            "warning: skipping product with non-polygon geometry",
                            product,
                        )
                        continue
                    products.append(product)

                with open_atomic(cache_fname, "w") as f:
                    json.dump(products, f)

            else:
                with cache_fname.open() as f:
                    products = json.load(f)

            for product in products:
                tile_geometry_dict = product["tileDataGeometry"]
                assert tile_geometry_dict["crs"]["type"] == "name"
                crs = CRS.from_string(tile_geometry_dict["crs"]["properties"]["name"])
                ts = dateutil.parser.isoparse(product["timestamp"])
                geometry = STGeometry(
                    Projection(crs, 1, 1),
                    shapely.geometry.shape(tile_geometry_dict),
                    (ts, ts),
                )
                yield Sentinel2Item(
                    name=product["productName"],
                    geometry=geometry,
                    blob_path=product["path"] + "/",
                    cloud_cover=product["cloudyPixelPercentage"],
                )

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
        # Identify all (mgrs grid cell, year, month, day) where we need to search for
        # images.
        # To do so, we iterate over the geometries and figure out all the MGRS cells
        # that they intersect.
        needed_cell_months = set()
        wgs84_geometries = [
            geometry.to_projection(WGS84_PROJECTION) for geometry in geometries
        ]
        for wgs84_geometry in wgs84_geometries:
            if wgs84_geometry.time_range is None:
                raise ValueError(
                    "Sentinel2 on AWS requires geometry time ranges to be set"
                )
            for cell_id in get_sentinel2_tiles(wgs84_geometry, self.metadata_cache_dir):
                for ts in daterange(
                    wgs84_geometry.time_range[0] - self.max_time_delta,
                    wgs84_geometry.time_range[1] + self.max_time_delta,
                ):
                    needed_cell_months.add((cell_id, ts.year, ts.month))

        items_by_cell: dict[str, list[Sentinel2Item]] = {}
        for item in self._read_products(needed_cell_months):
            cell_id = "".join(item.blob_path.split("/")[1:4])
            if cell_id not in items_by_cell:
                items_by_cell[cell_id] = []
            items_by_cell[cell_id].append(item)

        groups = []
        for geometry, wgs84_geometry in zip(geometries, wgs84_geometries):
            items = []
            for cell_id in get_sentinel2_tiles(wgs84_geometry, self.metadata_cache_dir):
                for item in items_by_cell.get(cell_id, []):
                    try:
                        item_geom = item.geometry.to_projection(geometry.projection)
                    except ValueError as e:
                        print(f"error re-projecting item {item.name}: {e}")
                        continue
                    if not geometry.shp.intersects(item_geom.shp):
                        continue
                    items.append(item)

            if self.sort_by == "cloud_cover":
                items.sort(key=lambda item: item.cloud_cover)
            elif self.sort_by is not None:
                raise ValueError(f"invalid sort_by setting ({self.sort_by})")

            cur_groups = rslearn.data_sources.utils.match_candidate_items_to_window(
                geometry, items, query_config
            )
            groups.append(cur_groups)

        return groups

    def get_item_by_name(self, name: str) -> Sentinel2Item:
        """Gets an item by name."""
        # Product name is like:
        #   S2B_MSIL1C_20240201T230819_N0510_R015_T51CWM_20240202T012755.
        # We want to use _read_products so we need to extract:
        # - cell_id: 51CWM
        # - year: 2024
        # - month: 2
        parts = name.split("_")
        assert parts[5][0] == "T"
        cell_id = parts[5][1:]
        assert len(parts[2]) == 15
        year = int(parts[2][0:4])
        month = int(parts[2][4:6])
        for item in self._read_products({(cell_id, year, month)}):
            if item.name == name:
                return item
        raise ValueError(f"item {name} not found")

    def deserialize_item(self, serialized_item: Any) -> Sentinel2Item:
        """Deserializes an item from JSON-decoded data."""
        assert isinstance(serialized_item, dict)
        return Sentinel2Item.deserialize(serialized_item)

    def retrieve_item(
        self, item: Sentinel2Item
    ) -> Generator[tuple[str, BinaryIO], None, None]:
        """Retrieves the rasters corresponding to an item as file streams."""
        for fname, _ in self.band_fnames[self.modality]:
            buf = io.BytesIO()
            self.bucket.download_fileobj(
                item.blob_path + fname, buf, ExtraArgs={"RequestPayer": "requester"}
            )
            buf.seek(0)
            yield (fname, buf)

    def _get_harmonize_callback(
        self, item: Sentinel2Item
    ) -> Callable[[npt.NDArray], npt.NDArray] | None:
        """Gets the harmonization callback for the given item.

        Args:
            item: the item

        Returns:
            None if no callback is needed, or the callback to subtract the new offset
        """
        if not self.harmonize:
            return None
        # Search metadata XML for the RADIO_ADD_OFFSET tag.
        # This contains the per-band offset, but we assume all bands have the same offset.
        if item.geometry.time_range is None:
            raise ValueError("Sentinel2 on AWS requires geometry time ranges to be set")
        ts = item.geometry.time_range[0]
        metadata_fname = (
            f"products/{ts.year}/{ts.month}/{ts.day}/{item.name}/metadata.xml"
        )
        buf = io.BytesIO()
        self.bucket.download_fileobj(
            metadata_fname, buf, ExtraArgs={"RequestPayer": "requester"}
        )
        buf.seek(0)
        tree = ET.ElementTree(ET.fromstring(buf.getvalue()))
        return get_harmonize_callback(tree)

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
            harmonize_callback = self._get_harmonize_callback(item)

            for fname, band_names in self.band_fnames[self.modality]:
                cur_tile_store = PrefixedTileStore(
                    tile_store, (item.name, "_".join(band_names))
                )
                needed_projections = get_needed_projections(
                    cur_tile_store, band_names, self.config.band_sets, cur_geometries
                )
                if not needed_projections:
                    continue

                buf = io.BytesIO()
                try:
                    self.bucket.download_fileobj(
                        item.blob_path + fname,
                        buf,
                        ExtraArgs={"RequestPayer": "requester"},
                    )
                except Exception as e:
                    # TODO: sometimes for some reason object doesn't exist
                    # we should probably investigate further why it happens
                    # and then should create the layer here and mark it completed
                    print(
                        f"warning: got error {e} downloading {item.blob_path + fname}"
                    )
                    continue
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
