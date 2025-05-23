"""Data source for raster data in Registry of Open Data on AWS."""

import io
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from collections.abc import Generator
from datetime import datetime
from typing import Any, BinaryIO

import affine
import boto3
import dateutil.parser
import fiona
import fiona.transform
import numpy.typing as npt
import rasterio
import shapely
import shapely.geometry
import tqdm
from rasterio.enums import Resampling
from upath import UPath

import rslearn.data_sources.utils
from rslearn.config import LayerConfig, RasterLayerConfig
from rslearn.const import SHAPEFILE_AUX_EXTENSIONS, WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.dataset.materialize import RasterMaterializer
from rslearn.tile_stores import TileStore, TileStoreWithLayer
from rslearn.utils.fsspec import get_upath_local, join_upath, open_atomic
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry
from rslearn.utils.grid_index import GridIndex

from .data_source import DataSource, Item, QueryConfig

WRS2_GRID_SIZE = 1.0


class LandsatOliTirsItem(Item):
    """An item in the LandsatOliTirs data source."""

    def __init__(
        self, name: str, geometry: STGeometry, blob_path: str, cloud_cover: float
    ) -> None:
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
    def deserialize(d: dict) -> "LandsatOliTirsItem":
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


class LandsatOliTirs(DataSource, TileStore):
    """A data source for Landsat 8/9 OLI-TIRS imagery on AWS.

    Specifically, uses the usgs-landsat S3 bucket maintained by USGS. The data includes
    Tier 1/2 scenes but does not seem to include Real-Time scenes.

    See https://aws.amazon.com/marketplace/pp/prodview-ivr4jeq6flk7u for details about
    the bucket.
    """

    bucket_name = "usgs-landsat"
    bucket_prefix = "collection02/level-1/standard/oli-tirs"
    bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"]

    wrs2_url = "https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/WRS2_descending_0.zip"  # noqa
    """URL to download shapefile specifying polygon of each (path, row)."""

    def __init__(
        self,
        config: RasterLayerConfig,
        metadata_cache_dir: UPath,
        sort_by: str | None = None,
    ) -> None:
        """Initialize a new LandsatOliTirs instance.

        Args:
            config: configuration of this layer
            metadata_cache_dir: directory to cache product metadata files.
            sort_by: can be "cloud_cover", default arbitrary order; only has effect for
                SpaceMode.WITHIN.
        """
        self.config = config
        self.metadata_cache_dir = metadata_cache_dir
        self.sort_by = sort_by

        self.client = boto3.client("s3")
        self.bucket = boto3.resource("s3").Bucket(self.bucket_name)
        self.metadata_cache_dir.mkdir(parents=True, exist_ok=True)

        self.wrs2_index: GridIndex | None = None

    @staticmethod
    def from_config(config: RasterLayerConfig, ds_path: UPath) -> "LandsatOliTirs":
        """Creates a new LandsatOliTirs instance from a configuration dictionary."""
        if config.data_source is None:
            raise ValueError(f"data_source is required for config dict {config}")
        d = config.data_source.config_dict
        kwargs = dict(
            config=config,
            metadata_cache_dir=join_upath(ds_path, d["metadata_cache_dir"]),
        )
        if "sort_by" in d:
            kwargs["sort_by"] = d["sort_by"]

        return LandsatOliTirs(**kwargs)

    def _read_products(
        self, needed_year_pathrows: set[tuple[int, str, str]]
    ) -> Generator[LandsatOliTirsItem, None, None]:
        """Read _stac.json files and yield relevant LandsatOliTirsItems.

        Args:
            needed_year_pathrows: set of (year, path, row) where we need to search for
                images.
        """
        for year, path, row in tqdm.tqdm(
            needed_year_pathrows, desc="Reading product infos"
        ):
            assert len(path) == 3
            assert len(row) == 3
            local_fname = self.metadata_cache_dir / f"{year}_{path}_{row}.json"

            if not local_fname.exists():
                prefix = f"{self.bucket_prefix}/{year}/{path}/{row}/"
                items = []
                for obj in self.bucket.objects.filter(
                    Prefix=prefix, RequestPayer="requester"
                ):
                    # Only read the _stac.json files.
                    # Previously we used _MTL.json but those files don't have the full
                    # geometry of the Landsat scene, only the bounding box.
                    if not obj.key.endswith("_stac.json"):
                        continue
                    # Load JSON data.
                    buf = io.BytesIO()
                    self.bucket.download_fileobj(
                        obj.key, buf, ExtraArgs={"RequestPayer": "requester"}
                    )
                    buf.seek(0)
                    stac_data = json.load(buf)

                    # Get polygon coordinates.
                    shp = shapely.geometry.shape(stac_data["geometry"])

                    # Get datetime.
                    ts = dateutil.parser.isoparse(stac_data["properties"]["datetime"])

                    blob_path = obj.key.split("stac.json")[0]
                    time_range: tuple[datetime, datetime] = (ts, ts)
                    geometry = STGeometry(WGS84_PROJECTION, shp, time_range)
                    cloud_cover: float
                    if "eo:cloud_cover" in stac_data["properties"]:
                        cloud_cover = stac_data["properties"]["eo:cloud_cover"]
                    elif "landsat:cloud_cover_land" in stac_data["properties"]:
                        cloud_cover = stac_data["properties"][
                            "landsat:cloud_cover_land"
                        ]
                    else:
                        cloud_cover = -1
                    items.append(
                        LandsatOliTirsItem(
                            name=stac_data["id"],
                            geometry=geometry,
                            blob_path=blob_path,
                            cloud_cover=cloud_cover,
                        )
                    )

                with open_atomic(local_fname, "w") as f:
                    json.dump([item.serialize() for item in items], f)

            else:
                with local_fname.open() as f:
                    items = [
                        LandsatOliTirsItem.deserialize(item_dict)
                        for item_dict in json.load(f)
                    ]

            yield from items

    def _get_wrs2_polygons(self) -> list[tuple[shapely.Geometry, str, str]]:
        """Get polygons for each (path, row) in the WRS2 grid.

        Returns:
            List of (polygon, path, row).
        """
        prefix = "WRS2_descending"
        shp_fname = self.metadata_cache_dir / f"{prefix}.shp"
        if not shp_fname.exists():
            # Download and extract zip to cache dir.
            zip_fname = self.metadata_cache_dir / f"{prefix}.zip"
            print(f"Downloading {self.wrs2_url} to {zip_fname}")
            with urllib.request.urlopen(self.wrs2_url) as response:
                with zip_fname.open("wb") as f:
                    shutil.copyfileobj(response, f)
            with zip_fname.open("rb") as f:
                with zipfile.ZipFile(f, "r") as zipf:
                    member_names = zipf.namelist()
                    for ext in SHAPEFILE_AUX_EXTENSIONS:
                        cur_fname = "WRS2_descending" + ext
                        if cur_fname not in member_names:
                            continue
                        with zipf.open(cur_fname) as memberf:
                            with (self.metadata_cache_dir / (prefix + ext)).open(
                                "wb"
                            ) as f:
                                shutil.copyfileobj(memberf, f)

                    with zipf.open(f"{prefix}.shp") as memberf:
                        with shp_fname.open("wb") as f:
                            shutil.copyfileobj(memberf, f)

        aux_files: list[UPath] = []
        for ext in SHAPEFILE_AUX_EXTENSIONS:
            aux_files.append(self.metadata_cache_dir / (prefix + ext))

        with get_upath_local(shp_fname, extra_paths=aux_files) as local_fname:
            with fiona.open(local_fname) as src:
                polygons = []
                for feat in src:
                    shp = shapely.geometry.shape(feat["geometry"])

                    # Need to buffer the shape because Landsat scenes include some
                    # buffer beyond the polygon and this has caused us to not identify
                    # scenes in the past.
                    # 0.2 degree buffer seems sufficient.
                    shp = shp.buffer(0.2)

                    path = str(feat["properties"]["PATH"]).zfill(3)
                    row = str(feat["properties"]["ROW"]).zfill(3)
                    polygons.append((shp, path, row))
                return polygons

    def _get_wrs2_index(self) -> GridIndex:
        """Get a grid index over the WRS2 polygons."""
        if self.wrs2_index is not None:
            return self.wrs2_index

        # Index doesn't exist so we need to build it.
        # We cache it with the object since it takes a bit of time to create it.
        polygons = self._get_wrs2_polygons()
        self.wrs2_index = GridIndex(WRS2_GRID_SIZE)
        for polygon, path, row in polygons:
            self.wrs2_index.insert(polygon.bounds, (polygon, path, row))
        return self.wrs2_index

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[LandsatOliTirsItem]]]:
        """Get a list of items in the data source intersecting the given geometries.

        Args:
            geometries: the spatiotemporal geometries
            query_config: the query configuration

        Returns:
            List of groups of items that should be retrieved for each geometry.
        """
        wrs2_index = self._get_wrs2_index()
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
            for polygon, path, row in wrs2_index.query(wgs84_geometry.shp.bounds):
                if wgs84_geometry.shp.intersects(polygon):
                    cur_pathrows.add((path, row))
            for path, row in cur_pathrows:
                for year in range(
                    wgs84_geometry.time_range[0].year,
                    wgs84_geometry.time_range[1].year + 1,
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
                cur_items.sort(
                    key=lambda item: item.cloud_cover if item.cloud_cover >= 0 else 100
                )
            elif self.sort_by is not None:
                raise ValueError(f"invalid sort_by setting ({self.sort_by})")

            cur_groups: list[list[LandsatOliTirsItem]] = (
                rslearn.data_sources.utils.match_candidate_items_to_window(
                    geometry, cur_items, query_config
                )
            )
            groups.append(cur_groups)

        return groups

    def get_item_by_name(self, name: str) -> LandsatOliTirsItem:
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

    def deserialize_item(self, serialized_item: Any) -> LandsatOliTirsItem:
        """Deserializes an item from JSON-decoded data."""
        assert isinstance(serialized_item, dict)
        return LandsatOliTirsItem.deserialize(serialized_item)

    def retrieve_item(
        self, item: LandsatOliTirsItem
    ) -> Generator[tuple[str, BinaryIO], None, None]:
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
        tile_store: TileStoreWithLayer,
        items: list[LandsatOliTirsItem],
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
                if tile_store.is_raster_ready(item.name, band_names):
                    continue

                with tempfile.TemporaryDirectory() as tmp_dir:
                    fname = os.path.join(tmp_dir, f"{band}.tif")
                    self.bucket.download_file(
                        item.blob_path + f"{band}.TIF",
                        fname,
                        ExtraArgs={"RequestPayer": "requester"},
                    )
                    tile_store.write_raster_file(item.name, band_names, UPath(fname))

    # The functions below are to emulate TileStore functionality so we can easily
    # support materialization directly from the COGs.
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
        # Always ready since we access it on AWS bucket.
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
        return [[band] for band in self.bands]

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
        # Landsat assets have single band per asset.
        assert len(bands) == 1
        band = bands[0]

        # Get the item since it has the blob path.
        item = self.get_item_by_name(item_name)

        # Create pre-signed URL for rasterio access.
        # We do this because accessing via URL is much faster since rasterio can use
        # the URL directly.
        blob_key = item.blob_path + f"{band}.TIF"
        url = self.client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": self.bucket_name,
                "Key": blob_key,
                "RequestPayer": "requester",
            },
        )

        # Construct the transform to use for the warped dataset.
        wanted_transform = affine.Affine(
            projection.x_resolution,
            0,
            bounds[0] * projection.x_resolution,
            0,
            projection.y_resolution,
            bounds[1] * projection.y_resolution,
        )

        with rasterio.open(url) as src:
            with rasterio.vrt.WarpedVRT(
                src,
                crs=projection.crs,
                transform=wanted_transform,
                width=bounds[2] - bounds[0],
                height=bounds[3] - bounds[1],
                resampling=resampling,
            ) as vrt:
                return vrt.read()

    def materialize(
        self,
        window: Window,
        item_groups: list[list[LandsatOliTirsItem]],
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
