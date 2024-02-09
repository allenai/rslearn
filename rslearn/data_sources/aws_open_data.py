"""Data source for raster data in Registry of Open Data on AWS."""

import glob
import io
import json
import os
from datetime import datetime, timezone
from typing import Any, Generator, Optional

import boto3
import fiona
import fiona.transform
import rasterio
import shapely
import tqdm
from rasterio.crs import CRS

import rslearn.data_sources.utils
from rslearn.tile_stores import TileStore
from rslearn.utils import WGS84_EPSG, GridIndex, STGeometry

from .data_source import DataSource, Item, QueryConfig
from .raster_source import RasterOptions, ingest_from_rasters


class NaipItem(Item):
    """An item in the Naip data source."""

    def __init__(
        self, name: str, shp: shapely.Geometry, time: datetime, blob_path: str
    ):
        """Creates a new NaipItem.

        Args:
            name: unique name of the item
            shp: the geometry of the item
            time: the time of the item
            blob_path: path in bucket
        """
        super().__init__(name, shp, time)
        self.blob_path = blob_path

    def serialize(self) -> dict:
        """Serializes the item to a JSON-encodable dictionary."""
        d = super().serialize()
        d["blob_path"] = self.blob_path
        return d

    @staticmethod
    def deserialize(d: dict) -> Item:
        """Deserializes an item from a JSON-decoded dictionary."""
        if "name" not in d:
            d["name"] = d["blob_path"].split("/")[-1].split(".tif")[0]
        item = super(NaipItem, NaipItem).deserialize(d)
        return NaipItem(
            name=item.name,
            shp=item.shp,
            time=item.time,
            blob_path=d["blob_path"],
        )


class Naip(DataSource):
    """A data source for NAIP imagery on AWS.

    Specifically uses the naip-source requester pays bucket maintained by Esri. See
    https://registry.opendata.aws/naip/ for more information.
    """

    bucket_name = "naip-source"

    manifest_fname = "manifest.txt"

    def __init__(
        self,
        index_cache_dir: str,
        raster_options: Optional[RasterOptions] = RasterOptions(),
        use_rtree_index=False,
    ) -> None:
        """Initialize a new Naip instance.

        Args:
            index_cache_dir: local directory to cache index shapefiles.
        """
        self.index_cache_dir = index_cache_dir
        self.raster_options = raster_options

        self.bucket = boto3.resource("s3").Bucket(self.bucket_name)

        if use_rtree_index:
            from rslearn.utils.rtree_index import RtreeIndex

            rtree_fname = os.path.join(self.index_cache_dir, "rtree_index")
            needs_building = not os.path.exists(rtree_fname + ".dat")
            self.rtree_index = RtreeIndex(rtree_fname)
            if needs_building:
                self._build_index()
        else:
            self.rtree_index = None

    def _download_manifest(self) -> str:
        """Download the manifest that enumerates files in the bucket.

        Returns:
            The local filename where the manifest has been downloaded.
        """
        local_fname = os.path.join(self.index_cache_dir, self.manifest_fname)
        if not os.path.exists(local_fname):
            self.bucket.download_file(
                self.manifest_fname,
                local_fname,
                ExtraArgs={"RequestPayer": "requester"},
            )
        return local_fname

    def _download_index_shapefiles(self) -> None:
        """Download all index shapefiles that specify image extents."""
        manifest_fname = self._download_manifest()
        needed_files = []
        with open(manifest_fname) as f:
            for line in f:
                blob_path = line.strip()
                if not blob_path:
                    continue
                if "/index/" not in blob_path:
                    continue

                # Data before 2012 doesn't seem to be present even though index files
                # might exist?
                path_parts = blob_path.split("/")
                if int(path_parts[1]) < 2012:
                    continue

                local_fname = os.path.join(self.index_cache_dir, blob_path)
                if os.path.exists(local_fname):
                    continue
                needed_files.append((blob_path, local_fname))
        for blob_path, local_fname in tqdm.tqdm(
            needed_files, desc="Downloading index files"
        ):
            os.makedirs(os.path.dirname(local_fname), exist_ok=True)
            self.bucket.download_file(
                blob_path, local_fname, ExtraArgs={"RequestPayer": "requester"}
            )

    def _read_index_shapefiles(self, desc=None) -> Generator[NaipItem, None, None]:
        """Read the index shapefiles and yield NaipItems corresponding to each image."""
        self._download_index_shapefiles()

        # Sometimes we see files like m_3410458_sw_13_060_20180612_20190426.tif but
        # the actual name is m_3410458_sw_13_060_20180612_20190427.tif (last part, the
        # version date, is different).
        # Also sometimes it is just m_4212362_sw_10_1_20140622.tif (ending in capture
        # date).
        # So here we create dict from just the prefix (i.e. exclude the version date).
        tif_blob_path_dict = {}
        manifest_fname = self._download_manifest()
        with open(manifest_fname) as f:
            for line in f:
                blob_path = line.strip()
                if not blob_path:
                    continue
                if not blob_path.endswith(".tif"):
                    continue
                parts = blob_path.split(".tif")[0].split("_")
                prefix = "_".join(parts[0:6])
                tif_blob_path_dict[prefix] = blob_path

        shape_files = glob.glob(
            "**/*.shp", recursive=True, root_dir=self.index_cache_dir
        )
        if desc:
            shape_files = tqdm.tqdm(shape_files, desc=desc)
        for shp_fname in shape_files:
            with fiona.open(os.path.join(self.index_cache_dir, shp_fname)) as f:
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
                    base_dir = os.path.dirname(os.path.dirname(shp_fname))
                    tif_fname = feature["properties"]["FileName"]
                    fname_parts = tif_fname.split(".tif")[0].split("_")
                    tile_id = fname_parts[1][0:5]
                    fname_prefix = "_".join(fname_parts[:-1])
                    full_prefix = "{}/rgbir/{}/{}".format(
                        base_dir, tile_id, fname_prefix
                    )

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

                    yield NaipItem(
                        name=blob_path.split("/")[-1].split(".tif")[0],
                        shp=shp,
                        time=time,
                        blob_path=blob_path,
                    )

    def _build_index(self):
        """Build the RtreeIndex from items in the data source."""
        for item in self._read_index_shapefiles(desc="Building rtree index"):
            self.rtree_index.insert(item.shp.bounds, json.dumps(item.serialize()))

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
        wgs84_crs = CRS.from_epsg(WGS84_EPSG)
        wgs84_geometries = [geometry.to_crs(wgs84_crs, 1) for geometry in geometries]

        items = [[] for _ in geometries]
        if self.rtree_index:
            for idx, geometry in enumerate(wgs84_geometries):
                encoded_items = self.rtree_index.query(geometry.shp.bounds)
                for encoded_item in encoded_items:
                    item = NaipItem.deserialize(json.loads(encoded_item))
                    if not item.shp.intersects(geometry.shp):
                        continue
                    items[idx].append(item)
        else:
            index = GridIndex(0.01)
            for idx, geometry in wgs84_geometries:
                index.insert(geometry.bounds, idx)
            for i, item in enumerate(self._read_index_shapefiles()):
                results = index.query(item.shp.bounds)
                for idx in results:
                    geometry = wgs84_geometries[idx]
                    if not geometry.shp.intersects(item.shp):
                        continue
                    items[idx].append(item)

        groups = []
        for geometry, item_list in zip(wgs84_geometries, items):
            cur_groups = rslearn.data_sources.utils.match_candidate_items_to_window(
                geometry, item_list, query_config
            )
            groups.append(cur_groups)
        return groups

    def deserialize_item(self, serialized_item: Any) -> Item:
        """Deserializes an item from JSON-decoded data."""
        assert isinstance(serialized_item, dict)
        return NaipItem.deserialize(serialized_item)

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
            # Get distinct CRS/resolution pairs.
            projections = set()
            for geometry in cur_geometries:
                projections.add((geometry.crs, geometry.resolution))
            projections = list(projections)

            # Check if this item was already ingested for all the projections needed.
            layer_prefix = (item.name,)
            any_needed = False
            for crs, resolution in projections:
                ts_layer = tile_store.get_layer(
                    layer_prefix + (crs.to_string(), str(resolution))
                )
                any_needed = (
                    any_needed
                    or ts_layer is None
                    or not ts_layer.get_metadata().properties.get("completed")
                )
            if not any_needed:
                continue

            buf = io.BytesIO()
            self.bucket.download_fileobj(
                item.blob_path, buf, ExtraArgs={"RequestPayer": "requester"}
            )
            buf.seek(0)
            raster = rasterio.open(buf)
            rasters = [(raster, ["R", "G", "B", "IR"])]
            ingest_from_rasters(
                tile_store, layer_prefix, rasters, projections, self.raster_options
            )
