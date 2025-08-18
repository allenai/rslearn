"""Data source for raster or vector data in local files."""

import csv
import io
import json
import os
import tempfile
import time
from datetime import UTC, datetime
from typing import Any

import ee
import numpy as np
import numpy.typing as npt
import rasterio
import rasterio.merge
import shapely
import tqdm
from google.cloud import storage
from upath import UPath

import rslearn.data_sources.utils
from rslearn.config import DType, LayerConfig, RasterLayerConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset.materialize import RasterMaterializer
from rslearn.dataset.window import Window
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStore, TileStoreWithLayer
from rslearn.utils.array import copy_spatial_array
from rslearn.utils.fsspec import join_upath
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry
from rslearn.utils.raster_format import (
    Resampling,
    get_raster_projection_and_bounds_from_transform,
    get_transform_from_projection_and_bounds,
)
from rslearn.utils.rtree_index import RtreeIndex, get_cached_rtree

from .data_source import DataSource, Item, QueryConfig

logger = get_logger(__name__)


class NoValidPixelsException(Exception):
    """Exception when GEE API reports that export failed due to no valid pixels."""

    # Expected GEE error_message when the task fails.
    GEE_MESSAGE = "No valid (un-masked) pixels in export region."


class ExportException(Exception):
    """GEE API export error."""

    pass


class GEE(DataSource, TileStore):
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
                elif status_dict["state"] != "COMPLETED":
                    raise ValueError(
                        f"got unexpected GEE task state {status_dict['state']}"
                    )
                break

        # Read the CSV and add rows into the rtree index.
        with csv_blob.open() as f:
            reader = csv.DictReader(f)
            for row in tqdm.tqdm(reader, desc="Building index"):
                shp = shapely.geometry.shape(json.loads(row[".geo"]))
                if "E" in row["time"]:
                    unix_time = float(row["time"]) / 1000
                    ts = datetime.fromtimestamp(unix_time, tz=UTC)
                else:
                    ts = datetime.fromisoformat(row["time"]).replace(tzinfo=UTC)
                geometry = STGeometry(WGS84_PROJECTION, shp, (ts, ts))
                item = Item(row["system:index"], geometry)
                rtree_index.insert(shp.bounds, json.dumps(item.serialize()))

    def get_item_by_name(self, name: str) -> Item:
        """Gets an item by name.

        Args:
            name: the name of the item to get

        Returns:
            the item object
        """
        filtered = self.get_collection().filter(ee.Filter.eq("system:index", name))
        image = filtered.first()
        shp = shapely.geometry.shape(
            image.geometry().transform(proj="EPSG:4326", maxError=0.001).getInfo()
        )
        ts = datetime.fromisoformat(image.date().format().getInfo()).replace(tzinfo=UTC)
        geometry = STGeometry(WGS84_PROJECTION, shp, (ts, ts))
        return Item(name, geometry)

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

    def export_item(
        self,
        item: Item,
        blob_prefix: str,
        projection_and_bounds: tuple[Projection, PixelBounds] | None = None,
    ) -> None:
        """Export the item to the specified folder.

        Args:
            item: the item to export.
            blob_prefix: the prefix (folder) to use.
            projection_and_bounds: optionally use this projection and bounds instead of
                the extent of the image.
        """
        image = self.item_to_image(item)
        projection = image.select(self.bands[0]).projection().getInfo()
        logger.info("Starting task to retrieve image %s", item.name)

        extent_kwargs: dict[str, Any]
        if projection_and_bounds is not None:
            projection, bounds = projection_and_bounds
            transform = get_transform_from_projection_and_bounds(projection, bounds)
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            extent_kwargs = dict(
                crs=str(projection.crs),
                crsTransform=[
                    transform.a,
                    transform.b,
                    transform.c,
                    transform.d,
                    transform.e,
                    transform.f,
                ],
                dimensions=f"{width}x{height}",
            )
        else:
            # Use the native projection of the image.
            # We pass scale instead of crsTransform since some images have positive y
            # resolution which means they are upside down and rasterio cannot merge
            # them.
            extent_kwargs = dict(
                crs=projection["crs"],
                scale=projection["transform"][0],
            )

        task = ee.batch.Export.image.toCloudStorage(
            image=image,
            description=item.name,
            bucket=self.gcs_bucket_name,
            fileNamePrefix=blob_prefix,
            maxPixels=10000000000,
            fileFormat="GeoTIFF",
            skipEmptyTiles=True,
            **extent_kwargs,
        )
        task.start()
        while True:
            time.sleep(10)
            status_dict = task.status()
            if status_dict["state"] in ["UNSUBMITTED", "READY", "RUNNING"]:
                continue
            if status_dict["state"] == "COMPLETED":
                break
            if status_dict["state"] != "FAILED":
                raise ValueError(
                    f"got unexpected GEE task state {status_dict['state']}"
                )
            # The task failed. We see if it is an okay failure case or if we need to
            # raise exception.
            if status_dict["error_message"] == NoValidPixelsException.GEE_MESSAGE:
                raise NoValidPixelsException()
            raise ExportException(f"GEE task failed: {status_dict['error_message']}")

    def _merge_rasters(
        self,
        blobs: list[storage.Blob],
        crs_bounds: tuple[float, float, float, float] | None = None,
        res: float | None = None,
    ) -> tuple[npt.NDArray, Projection, PixelBounds]:
        """Merge multiple rasters split up during export by GEE.

        GEE can produce multiple rasters if it determines the file size exceeds its
        internal limit. So in this case we stitch them back together.

        Args:
            blobs: the list of GCS blobs where the rasters were written.
            crs_bounds: generate merged output under this bounds, in CRS coordinates
                (not pixel units).
            res: generate merged output under this resolution.

        Returns:
            a tuple (array, projection, bounds) where the projection and bounds
                indicate the extent of the array.
        """
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            rasterio_datasets = []
            for blob in blobs:
                local_fname = os.path.join(tmp_dir_name, blob.name.split("/")[-1])
                blob.download_to_filename(local_fname)
                src = rasterio.open(local_fname)
                rasterio_datasets.append(src)

            merge_kwargs: dict[str, Any] = dict(
                sources=rasterio_datasets,
                bounds=crs_bounds,
                res=res,
            )
            if self.dtype:
                merge_kwargs["dtype"] = self.dtype.value
            array, transform = rasterio.merge.merge(**merge_kwargs)
            projection, bounds = get_raster_projection_and_bounds_from_transform(
                rasterio_datasets[0].crs,
                transform,
                array.shape[2],
                array.shape[1],
            )

            for ds in rasterio_datasets:
                ds.close()

        return array, projection, bounds

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
                    array, projection, bounds = self._merge_rasters(blobs)
                    tile_store.write_raster(
                        item.name, self.bands, projection, bounds, array
                    )

            for blob in blobs:
                blob.delete()

    def is_raster_ready(
        self, layer_name: str, item_name: str, bands: list[str]
    ) -> bool:
        """Checks if this raster has been written to the store.

        Args:
            layer_name: the layer name or alias.
            item_name: the item.
            bands: the list of bands identifying which specific raster to read.

        Returns:
            whether there is a raster in the store matching the source, item, and
                bands.
        """
        # Always ready since we wrap accesses to Planetary Computer.
        return True

    def get_raster_bands(self, layer_name: str, item_name: str) -> list[list[str]]:
        """Get the sets of bands that have been stored for the specified item.

        Args:
            layer_name: the layer name or alias.
            item_name: the item.

        Returns:
            a list of lists of bands that are in the tile store (with one raster
                stored corresponding to each inner list). If no rasters are ready for
                this item, returns empty list.
        """
        return [self.bands]

    def get_raster_bounds(
        self, layer_name: str, item_name: str, bands: list[str], projection: Projection
    ) -> PixelBounds:
        """Get the bounds of the raster in the specified projection.

        Args:
            layer_name: the layer name or alias.
            item_name: the item to check.
            bands: the list of bands identifying which specific raster to read. These
                bands must match the bands of a stored raster.
            projection: the projection to get the raster's bounds in.

        Returns:
            the bounds of the raster in the projection.
        """
        item = self.get_item_by_name(item_name)
        geom = item.geometry.to_projection(projection)
        return (
            int(geom.shp.bounds[0]),
            int(geom.shp.bounds[1]),
            int(geom.shp.bounds[2]),
            int(geom.shp.bounds[3]),
        )

    def read_raster(
        self,
        layer_name: str,
        item_name: str,
        bands: list[str],
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
    ) -> npt.NDArray[Any]:
        """Read raster data from the store.

        Args:
            layer_name: the layer name or alias.
            item_name: the item to read.
            bands: the list of bands identifying which specific raster to read. These
                bands must match the bands of a stored raster.
            projection: the projection to read in.
            bounds: the bounds to read.
            resampling: the resampling method to use in case reprojection is needed.

        Returns:
            the raster data
        """
        # Extract the requested extent and export to GCS.
        bounds_str = f"{bounds[0]}_{bounds[1]}_{bounds[2]}_{bounds[3]}"
        item = self.get_item_by_name(item_name)
        blob_prefix = f"{self.collection_name}/{item.name}.{bounds_str}.{os.getpid()}/"

        try:
            self.export_item(
                item, blob_prefix, projection_and_bounds=(projection, bounds)
            )
        except NoValidPixelsException:
            # No valid pixels means the result should be empty.
            logger.info(
                f"No valid pixels in item {item.name} with projection={projection}, bounds={bounds}, returning empty image"
            )
            return np.zeros(
                (len(bands), bounds[3] - bounds[1], bounds[2] - bounds[0]),
                dtype=np.float32,
            )

        wanted_transform = get_transform_from_projection_and_bounds(projection, bounds)
        crs_bounds = (
            bounds[0] * projection.x_resolution,
            bounds[3] * projection.y_resolution,
            bounds[2] * projection.x_resolution,
            bounds[1] * projection.y_resolution,
        )

        blobs = list(self.bucket.list_blobs(prefix=blob_prefix))

        if len(blobs) == 1:
            # With a single output, we can simply read it with vrt.
            buf = io.BytesIO()
            blobs[0].download_to_file(buf)
            buf.seek(0)
            with rasterio.open(buf) as src:
                with rasterio.vrt.WarpedVRT(
                    src,
                    crs=projection.crs,
                    transform=wanted_transform,
                    width=bounds[2] - bounds[0],
                    height=bounds[3] - bounds[1],
                    resampling=resampling,
                ) as vrt:
                    return vrt.read()

        else:
            # With multiple outputs, we need to merge them together.
            # We can set the bounds in CRS coordinates when we do the merging.
            if projection.x_resolution != -projection.y_resolution:
                raise NotImplementedError(
                    "Only projection with x_res=-y_res is supported for GEE direct materialization"
                )
            src_array, _, src_bounds = self._merge_rasters(
                blobs, crs_bounds=crs_bounds, res=projection.x_resolution
            )

            # We copy the array if its bounds don't match exactly.
            if src_bounds == bounds:
                return src_array
            dst_array = np.zeros(
                (src_array.shape[0], bounds[3] - bounds[1], bounds[2] - bounds[0]),
                dtype=src_array.dtype,
            )
            copy_spatial_array(src_array, dst_array, src_bounds[0:2], bounds[0:2])
            return dst_array

    def materialize(
        self,
        window: Window,
        item_groups: list[list[Item]],
        layer_name: str,
        layer_cfg: LayerConfig,
    ) -> None:
        """Materialize data for the window.

        Args:
            window: the window to materialize
            item_groups: the items from get_items
            layer_name: the name of this layer
            layer_cfg: the config of this layer
        """
        assert isinstance(layer_cfg, RasterLayerConfig)
        RasterMaterializer().materialize(
            TileStoreWithLayer(self, layer_name),
            window,
            layer_name,
            layer_cfg,
            item_groups,
        )


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
