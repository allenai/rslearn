"""Data source for raster or vector data in local files."""

import csv
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from typing import Any

import ee
import ee.ee_types
import rasterio
import rasterio.merge
import shapely
import tqdm
from google.cloud import storage
from upath import UPath

import rslearn.data_sources.utils
from rslearn.config import DType, RasterLayerConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils import STGeometry
from rslearn.utils.fsspec import join_upath
from rslearn.utils.raster_format import get_raster_projection_and_bounds_from_transform
from rslearn.utils.rtree_index import RtreeIndex, get_cached_rtree

from .data_source import DataSource, Item, QueryConfig

logger = get_logger(__name__)


class GEE(DataSource):
    """A data source for ingesting images from Google Earth Engine."""

    def __init__(
        self,
        collection_name: str,
        gcs_bucket_name: str,
        bands: list[str],
        index_cache_dir: UPath,
        service_account_name: str,
        service_account_credentials: str,
        filters: list[tuple[str, Any]] | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize a new GEE instance.

        Args:
            collection_name: the Earth Engine ImageCollection to ingest images from
            gcs_bucket_name: the Cloud Storage bucket to export GEE images to
            bands: the list of bands to ingest
            index_cache_dir: cache directory to store rtree index
            service_account_name: name of the service account to use for authentication
            service_account_credentials: service account credentials filename
            filters: optional list of tuples (property_name, property_value) to filter
                images (using ee.Filter.eq)
            dtype: optional desired array data type. If the data obtained from GEE does
                not match this type, then it is converted.
        """
        self.collection_name = collection_name
        self.gcs_bucket_name = gcs_bucket_name
        self.bands = bands
        self.index_cache_dir = index_cache_dir
        self.filters = filters
        self.dtype = dtype

        self.bucket = storage.Client().bucket(self.gcs_bucket_name)

        credentials = ee.ServiceAccountCredentials(
            service_account_name, service_account_credentials
        )
        ee.Initialize(credentials)

        self.index_cache_dir.mkdir(parents=True, exist_ok=True)
        self.rtree_index = get_cached_rtree(self.index_cache_dir, self._build_index)

    @staticmethod
    def from_config(config: RasterLayerConfig, ds_path: UPath) -> "GEE":
        """Creates a new GEE instance from a configuration dictionary."""
        if config.data_source is None:
            raise ValueError("data_source is required in config")
        d = config.data_source.config_dict
        bands = [band for band_set in config.band_sets for band in band_set.bands]
        kwargs = {
            "collection_name": d["collection_name"],
            "gcs_bucket_name": d["gcs_bucket_name"],
            "bands": bands,
            "service_account_name": d["service_account_name"],
            "service_account_credentials": d["service_account_credentials"],
            "filters": d.get("filters"),
            "index_cache_dir": join_upath(ds_path, d["index_cache_dir"]),
        }
        if "dtype" in d:
            kwargs["dtype"] = DType(d["dtype"])

        return GEE(**kwargs)

    def get_collection(self) -> ee.ImageCollection:
        """Returns the Earth Engine image collection for this data source."""
        image_collection = ee.ImageCollection(self.collection_name)
        if self.filters is None:
            return image_collection

        for k, v in self.filters:
            cur_filter = ee.Filter.eq(k, v)
            image_collection = image_collection.filter(cur_filter)
        return image_collection

    def _build_index(self, rtree_index: RtreeIndex) -> None:
        csv_blob = self.bucket.blob(f"{self.collection_name}/index.csv")

        if not csv_blob.exists():
            # Export feature collection of image metadata to GCS.
            def image_to_feature(image: ee.Image) -> ee.Feature:
                geometry = image.geometry().transform(proj="EPSG:4326", maxError=0.001)
                return ee.Feature(geometry, {"time": image.date().format()})

            fc = self.get_collection().map(image_to_feature)
            task = ee.batch.Export.table.toCloudStorage(
                collection=fc,
                description="rslearn GEE index export task",
                bucket=self.gcs_bucket_name,
                fileNamePrefix=f"{self.collection_name}/index",
                fileFormat="CSV",
            )
            task.start()
            logger.info(
                "Started task to export GEE index for image collection %s",
                self.collection_name,
            )
            while True:
                time.sleep(10)
                status_dict = task.status()
                logger.debug(
                    "Waiting for export task to complete, current status is %s",
                    status_dict,
                )
                if status_dict["state"] in ["UNSUBMITTED", "READY", "RUNNING"]:
                    continue
                assert status_dict["state"] == "COMPLETED"
                break

        # Read the CSV and add rows into the rtree index.
        with csv_blob.open() as f:
            reader = csv.DictReader(f)
            for row in tqdm.tqdm(reader, desc="Building index"):
                shp = shapely.geometry.shape(json.loads(row[".geo"]))
                if "E" in row["time"]:
                    unix_time = float(row["time"]) / 1000
                    ts = datetime.fromtimestamp(unix_time, tz=timezone.utc)
                else:
                    ts = datetime.fromisoformat(row["time"]).replace(
                        tzinfo=timezone.utc
                    )
                geometry = STGeometry(WGS84_PROJECTION, shp, (ts, ts))
                item = Item(row["system:index"], geometry)
                rtree_index.insert(shp.bounds, json.dumps(item.serialize()))

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

        groups = []
        for geometry in wgs84_geometries:
            cur_items = []
            encoded_items = self.rtree_index.query(geometry.shp.bounds)
            for encoded_item in encoded_items:
                item = Item.deserialize(json.loads(encoded_item))
                if not item.geometry.shp.intersects(geometry.shp):
                    continue
                cur_items.append(item)

            cur_items.sort(key=lambda item: item.geometry.time_range[0])  # type: ignore

            cur_groups = rslearn.data_sources.utils.match_candidate_items_to_window(
                geometry, cur_items, query_config
            )
            groups.append(cur_groups)

        return groups

    def deserialize_item(self, serialized_item: Any) -> Item:
        """Deserializes an item from JSON-decoded data."""
        assert isinstance(serialized_item, dict)
        return Item.deserialize(serialized_item)

    def item_to_image(self, item: Item) -> ee.image.Image:
        """Get the Image corresponding to the Item.

        This function is separated so it can be overriden if subclasses want to add
        modifications to the image.
        """
        filtered = self.get_collection().filter(ee.Filter.eq("system:index", item.name))
        image = filtered.first()
        image = image.select(self.bands)
        return image

    def export_item(self, item: Item, blob_prefix: str) -> None:
        """Export the item to the specified folder.

        Args:
            item: the item to export.
            blob_prefix: the prefix (folder) to use.
        """
        # Use the native projection of the image to obtain the raster.
        # We pass scale instead of crsTransform since some images have positive y
        # resolution which means they are upside down and rasterio cannot merge
        # them.
        image = self.item_to_image(item)
        projection = image.select(self.bands[0]).projection().getInfo()
        logger.info("Starting task to retrieve image %s", item.name)
        task = ee.batch.Export.image.toCloudStorage(
            image=image,
            description=item.name,
            bucket=self.gcs_bucket_name,
            fileNamePrefix=blob_prefix,
            crs=projection["crs"],
            scale=projection["transform"][0],
            maxPixels=10000000000,
            fileFormat="GeoTIFF",
            skipEmptyTiles=True,
        )
        task.start()
        while True:
            time.sleep(10)
            status_dict = task.status()
            if status_dict["state"] in ["UNSUBMITTED", "READY", "RUNNING"]:
                continue
            assert status_dict["state"] == "COMPLETED"
            break

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[Item],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        Args:
            tile_store: the tile store to ingest into
            items: the items to ingest
            geometries: a list of geometries needed for each item
        """
        for item in items:
            if tile_store.is_raster_ready(item.name, self.bands):
                continue

            # Export the item to GCS.
            blob_prefix = f"{self.collection_name}/{item.name}.{os.getpid()}/"
            self.export_item(item, blob_prefix)

            # See what files the export produced.
            # If there are multiple, then we merge them into one file since that's the
            # simplest way to handle it.
            blobs = list(self.bucket.list_blobs(prefix=blob_prefix))

            with tempfile.TemporaryDirectory() as tmp_dir_name:
                if len(blobs) == 1:
                    local_fname = os.path.join(
                        tmp_dir_name, blobs[0].name.split("/")[-1]
                    )
                    blobs[0].download_to_filename(local_fname)
                    tile_store.write_raster_file(
                        item.name, self.bands, UPath(local_fname)
                    )

                else:
                    rasterio_datasets = []
                    for blob in blobs:
                        local_fname = os.path.join(
                            tmp_dir_name, blob.name.split("/")[-1]
                        )
                        blob.download_to_filename(local_fname)
                        src = rasterio.open(local_fname)
                        rasterio_datasets.append(src)

                    merge_kwargs: dict[str, Any] = {"sources": rasterio_datasets}
                    if self.dtype:
                        merge_kwargs["dtype"] = self.dtype.value
                    array, transform = rasterio.merge.merge(**merge_kwargs)
                    projection, bounds = (
                        get_raster_projection_and_bounds_from_transform(
                            rasterio_datasets[0].crs,
                            transform,
                            array.shape[2],
                            array.shape[1],
                        )
                    )

                    for ds in rasterio_datasets:
                        ds.close()

                    tile_store.write_raster(
                        item.name, self.bands, projection, bounds, array
                    )

            for blob in blobs:
                blob.delete()


class GoogleSatelliteEmbeddings(GEE):
    """GEE data source for the Google Satellite Embeddings.

    See here for details:
    https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
    """

    COLLECTION_NAME = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"

    def __init__(
        self,
        gcs_bucket_name: str,
        index_cache_dir: UPath,
        service_account_name: str,
        service_account_credentials: str,
    ):
        """Create a new GoogleSatelliteEmbeddings. See GEE for the arguments."""
        super().__init__(
            bands=[f"A{idx:02d}" for idx in range(64)],
            collection_name=self.COLLECTION_NAME,
            gcs_bucket_name=gcs_bucket_name,
            index_cache_dir=index_cache_dir,
            service_account_name=service_account_name,
            service_account_credentials=service_account_credentials,
        )

    @staticmethod
    def from_config(config: RasterLayerConfig, ds_path: UPath) -> "GEE":
        """Creates a new GEE instance from a configuration dictionary."""
        if config.data_source is None:
            raise ValueError("data_source is required in config")
        d = config.data_source.config_dict
        kwargs = {
            "gcs_bucket_name": d["gcs_bucket_name"],
            "index_cache_dir": join_upath(ds_path, d["index_cache_dir"]),
            "service_account_name": d["service_account_name"],
            "service_account_credentials": d["service_account_credentials"],
        }
        return GoogleSatelliteEmbeddings(**kwargs)

    # Override to add conversion to uint16.
    def item_to_image(self, item: Item) -> ee.image.Image:
        """Get the Image corresponding to the Item."""
        filtered = self.get_collection().filter(ee.Filter.eq("system:index", item.name))
        image = filtered.first()
        image = image.select(self.bands)
        return image.multiply(8192).add(8192).toUint16()
