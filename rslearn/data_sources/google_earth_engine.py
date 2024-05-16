"""Data source for raster or vector data in local files."""

import csv
import io
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Optional

import ee
import rasterio
import rasterio.merge
import shapely
import tqdm
from google.cloud import storage

import rslearn.data_sources.utils
import rslearn.utils.mgrs
from rslearn.config import DType, LayerConfig, RasterLayerConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.tile_stores import PrefixedTileStore, TileStore
from rslearn.utils import STGeometry
from rslearn.utils.rtree_index import RtreeIndex

from .data_source import DataSource, Item, QueryConfig
from .raster_source import ArrayWithTransform, get_needed_projections, ingest_raster


class GEE(DataSource):
    """A data source for ingesting images from Google Earth Engine."""

    def __init__(
        self,
        config: LayerConfig,
        collection_name: str,
        gcs_bucket_name: str,
        index_fname: str,
        service_account_name: str,
        service_account_credentials: str,
        filters: Optional[list[tuple[str, Any]]] = None,
        dtype: Optional[DType] = None,
    ) -> None:
        """Initialize a new GEE instance.

        Args:
            config: configuration for this layer.
            collection_name: the Earth Engine collection to ingest images from
            gcs_bucket_name: the Cloud Storage bucket to export GEE images to
            index_fname: rtree index filename
            service_account_name: name of the service account to use for authentication
            service_account_credentials: service account credentials filename
            filters: optional list of tuples (property_name, property_value) to filter
                images (using ee.Filter.eq)
            dtype: optional desired array data type. If the data obtained from GEE does
                not match this type, then it is converted.
        """
        self.config = config
        self.collection_name = collection_name
        self.gcs_bucket_name = gcs_bucket_name
        self.filters = filters
        self.dtype = dtype

        self.bucket = storage.Client().bucket(self.gcs_bucket_name)

        credentials = ee.ServiceAccountCredentials(
            service_account_name, service_account_credentials
        )
        ee.Initialize(credentials)

        index_needs_building = not os.path.exists(index_fname + ".dat")
        self.rtree_index = RtreeIndex(index_fname)
        if index_needs_building:
            self._build_index()

    @staticmethod
    def from_config(config: LayerConfig, root_dir: str = ".") -> "GEE":
        """Creates a new GEE instance from a configuration dictionary."""
        d = config.data_source.config_dict
        kwargs = {
            "config": config,
            "collection_name": d["collection_name"],
            "gcs_bucket_name": d["gcs_bucket_name"],
            "index_fname": os.path.join(root_dir, d["index_fname"]),
            "service_account_name": d["service_account_name"],
            "service_account_credentials": d["service_account_credentials"],
            "filters": d.get("filters"),
        }
        if "dtype" in d:
            kwargs["dtype"] = DType(d["dtype"])
        return GEE(**kwargs)

    def get_collection(self):
        """Returns the Earth Engine image collection for this data source."""
        image_collection = ee.ImageCollection(self.collection_name)
        for k, v in self.filters:
            cur_filter = ee.Filter.eq(k, v)
            image_collection = image_collection.filter(cur_filter)
            return image_collection

    def _build_index(self):
        csv_blob = self.bucket.blob(f"{self.collection_name}/index.csv")

        if not csv_blob.exists():
            # Export feature collection of image metadata to GCS.
            def image_to_feature(image):
                geometry = image.geometry().transform(proj="EPSG:4326")
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
            print(
                "started task to export GEE index "
                + f"for image collection {self.collection_name}"
            )
            while True:
                time.sleep(10)
                status_dict = task.status()
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
                self.rtree_index.insert(shp.bounds, json.dumps(item.serialize()))

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

            cur_items.sort(key=lambda item: item.geometry.time_range[0])

            cur_groups = rslearn.data_sources.utils.match_candidate_items_to_window(
                geometry, cur_items, query_config
            )
            groups.append(cur_groups)

        return groups

    def deserialize_item(self, serialized_item: Any) -> Item:
        """Deserializes an item from JSON-decoded data."""
        assert isinstance(serialized_item, dict)
        return Item.deserialize(serialized_item)

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
        assert isinstance(self.config, RasterLayerConfig)
        bands = []
        for band_set in self.config.band_sets:
            for band in band_set.bands:
                if band in bands:
                    continue
                bands.append(band)

        for item, cur_geometries in zip(items, geometries):
            cur_tile_store = PrefixedTileStore(tile_store, (item.name, "_".join(bands)))
            needed_projections = get_needed_projections(
                cur_tile_store, bands, self.config.band_sets, cur_geometries
            )
            if not needed_projections:
                continue

            filtered = self.get_collection().filter(
                ee.Filter.eq("system:index", item.name)
            )
            image = filtered.first()
            image = image.select(bands)

            # Use the native projection of the image to obtain the raster.
            projection = image.select(bands[0]).projection().getInfo()
            print(f"starting task to retrieve image {item.name}")
            blob_path = f"{self.collection_name}/{item.name}.{os.getpid()}/"
            task = ee.batch.Export.image.toCloudStorage(
                image=image,
                description=item.name,
                bucket=self.gcs_bucket_name,
                fileNamePrefix=blob_path,
                fileFormat="GeoTIFF",
                crs=projection["crs"],
                crsTransform=projection["transform"],
                maxPixels=10000000000,
            )
            task.start()
            while True:
                time.sleep(10)
                status_dict = task.status()
                if status_dict["state"] in ["UNSUBMITTED", "READY", "RUNNING"]:
                    continue
                assert status_dict["state"] == "COMPLETED"
                break

            # See what files the export produced.
            # If there are multiple, then we merge them into one file since that's the
            # simplest way to handle it.
            blobs = self.bucket.list_blobs(prefix=blob_path)
            blobs = list(blobs)
            raster = None

            if len(blobs) == 1:
                buf = io.BytesIO()
                blobs[0].download_to_file(buf)
                buf.seek(0)
                raster = rasterio.open(buf)

            else:
                with tempfile.TemporaryDirectory() as tmp_dir_name:
                    rasterio_datasets = []
                    for blob in blobs:
                        local_fname = os.path.join(
                            tmp_dir_name, blob.name.split("/")[-1]
                        )
                        blob.download_to_filename(local_fname)
                        src = rasterio.open(local_fname)
                        rasterio_datasets.append(src)

                    merge_kwargs = {
                        "datasets": rasterio_datasets,
                    }
                    if self.dtype:
                        merge_kwargs["dtype"] = self.dtype.value
                    array, transform = rasterio.merge.merge(**merge_kwargs)
                    crs = rasterio_datasets[0].crs

                    for ds in rasterio_datasets:
                        ds.close()

                raster = ArrayWithTransform(array, crs, transform)

            for projection in needed_projections:
                ingest_raster(
                    tile_store=cur_tile_store,
                    raster=raster,
                    projection=projection,
                    time_range=item.geometry.time_range,
                    layer_config=self.config,
                )

            raster.close()

            for blob in blobs:
                blob.delete()
