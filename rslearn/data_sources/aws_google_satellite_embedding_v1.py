"""Data source for Google Satellite Embedding V1 dataset on AWS Open Data."""

import os
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import boto3
import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterio
import rasterio.vrt
import shapely
import shapely.wkt
from botocore import UNSIGNED
from botocore.config import Config
from rasterio.enums import Resampling
from upath import UPath

import rslearn.data_sources.utils
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.data_source import (
    DataSourceContext,
    Item,
    QueryConfig,
)
from rslearn.data_sources.direct_materialize_data_source import (
    DirectMaterializeDataSource,
)
from rslearn.utils.fsspec import join_upath
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry
from rslearn.utils.grid_index import GridIndex

# Band names for the 64 embedding channels
BANDS = [f"A{idx:02d}" for idx in range(64)]

# S3 bucket configuration
BUCKET_NAME = "us-west-2.opendata.source.coop"
BUCKET_PREFIX = "tge-labs/aef/v1/annual"
INDEX_KEY = f"{BUCKET_PREFIX}/aef_index.csv"
HTTP_URL_BASE = f"https://s3.us-west-2.amazonaws.com/{BUCKET_NAME}"

# Grid index cell size for spatial queries
GRID_SIZE = 1.0


class GoogleSatelliteEmbeddingV1Item(Item):
    """An item in the GoogleSatelliteEmbeddingV1 data source."""

    def __init__(
        self,
        name: str,
        geometry: STGeometry,
        s3_path: str,
    ) -> None:
        """Creates a new GoogleSatelliteEmbeddingV1Item.

        Args:
            name: unique name of the item (the filename without extension)
            geometry: the spatial and temporal extent of the item
            s3_path: full S3 path to the TIFF file
        """
        super().__init__(name, geometry)
        self.s3_path = s3_path

    def serialize(self) -> dict:
        """Serializes the item to a JSON-encodable dictionary."""
        d = super().serialize()
        d["s3_path"] = self.s3_path
        return d

    @staticmethod
    def deserialize(d: dict) -> "GoogleSatelliteEmbeddingV1Item":
        """Deserializes an item from a JSON-decoded dictionary."""
        item = super(
            GoogleSatelliteEmbeddingV1Item, GoogleSatelliteEmbeddingV1Item
        ).deserialize(d)
        return GoogleSatelliteEmbeddingV1Item(
            name=item.name,
            geometry=item.geometry,
            s3_path=d["s3_path"],
        )


class GoogleSatelliteEmbeddingV1(
    DirectMaterializeDataSource[GoogleSatelliteEmbeddingV1Item]
):
    """Data source for Google Satellite Embedding V1 on AWS Open Data.

    It consists of annual satellite embeddings at 10m resolution with 64 bands
    (A00-A63). The data is stored as Cloud-Optimized GeoTIFFs organized by year and UTM
    zone. Each file covers 8192x8192 pixels.

    Available years: 2018-2024.

    See https://registry.opendata.aws/aef-source/ for details.
    """

    def __init__(
        self,
        metadata_cache_dir: str,
        apply_dequantization: bool = True,
        context: DataSourceContext = DataSourceContext(),
    ) -> None:
        """Initialize a new GoogleSatelliteEmbeddingV1 instance.

        Args:
            metadata_cache_dir: directory to cache the index file.
            apply_dequantization: whether to apply de-quantization to convert
                int8 values to float32. The raw data is quantized int8; the
                de-quantization maps values to [-1, 1] using the formula:
                ((values / 127.5) ** 2) * sign(values). The raw data has nodata value
                -128 while with dequantization the nodata value is 0.0.
            context: the data source context.
        """
        # We have a single asset containing all 64 bands. Here "image" is an arbitrary
        # name, since DirectMaterializeDataSource requires an asset name.
        super().__init__(asset_bands={"image": BANDS})

        self.apply_dequantization = apply_dequantization

        # Set up cache directory
        if context.ds_path is not None:
            self.metadata_cache_dir = join_upath(context.ds_path, metadata_cache_dir)
        else:
            self.metadata_cache_dir = UPath(metadata_cache_dir)
        self.metadata_cache_dir.mkdir(parents=True, exist_ok=True)

        # S3 client with anonymous access (only used for downloading index)
        self.s3_client = boto3.client(
            "s3",
            config=Config(signature_version=UNSIGNED),
            region_name="us-west-2",
        )

        # Lazy-loaded grid index
        self._grid_index: GridIndex | None = None
        self._items_by_name: dict[str, GoogleSatelliteEmbeddingV1Item] | None = None

    def _read_index_csv(self) -> pd.DataFrame:
        """Read the index CSV, downloading from S3 if not cached.

        Returns:
            DataFrame with WKT, path, and year columns.
        """
        cache_file = self.metadata_cache_dir / "aef_index.csv"
        if not cache_file.exists():
            response = self.s3_client.get_object(Bucket=BUCKET_NAME, Key=INDEX_KEY)
            content = response["Body"].read()
            with cache_file.open("wb") as f:
                f.write(content)

        return pd.read_csv(
            cache_file,
            header=None,
            usecols=[0, 2, 3],
            names=["WKT", "path", "year"],
        )

    def _load_index(
        self,
    ) -> tuple[GridIndex, dict[str, GoogleSatelliteEmbeddingV1Item]]:
        """Load the index file and build spatial index.

        Returns:
            Tuple of (grid_index, items_by_name dict).
        """
        if self._grid_index is not None and self._items_by_name is not None:
            return self._grid_index, self._items_by_name

        df = self._read_index_csv()

        grid_index = GridIndex(GRID_SIZE)
        items_by_name: dict[str, GoogleSatelliteEmbeddingV1Item] = {}

        for _, row in df.iterrows():
            shp = shapely.wkt.loads(row["WKT"])

            year = int(row["year"])
            time_range = (
                datetime(year, 1, 1, tzinfo=UTC),
                datetime(year, 12, 31, 23, 59, 59, tzinfo=UTC),
            )

            s3_path = row["path"]
            name = s3_path.split("/")[-1].replace(".tiff", "")

            geometry = STGeometry(WGS84_PROJECTION, shp, time_range)
            item = GoogleSatelliteEmbeddingV1Item(
                name=name,
                geometry=geometry,
                s3_path=s3_path,
            )

            grid_index.insert(shp.bounds, item)
            items_by_name[name] = item

        self._grid_index = grid_index
        self._items_by_name = items_by_name
        return grid_index, items_by_name

    # --- DataSource implementation ---

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[GoogleSatelliteEmbeddingV1Item]]]:
        """Get a list of items in the data source intersecting the given geometries."""
        grid_index, _ = self._load_index()

        wgs84_geometries = [
            geometry.to_projection(WGS84_PROJECTION) for geometry in geometries
        ]

        groups = []
        for geometry, wgs84_geometry in zip(geometries, wgs84_geometries):
            cur_items = []
            for item in grid_index.query(wgs84_geometry.shp.bounds):
                if not wgs84_geometry.shp.intersects(item.geometry.shp):
                    continue
                # Check time range if specified
                if wgs84_geometry.time_range is not None:
                    item_start, item_end = item.geometry.time_range
                    query_start, query_end = wgs84_geometry.time_range
                    if item_end < query_start or item_start > query_end:
                        continue
                cur_items.append(item)

            cur_items.sort(key=lambda item: item.geometry.time_range[0])

            cur_groups: list[list[GoogleSatelliteEmbeddingV1Item]] = (
                rslearn.data_sources.utils.match_candidate_items_to_window(
                    geometry, cur_items, query_config
                )
            )
            groups.append(cur_groups)

        return groups

    def get_item_by_name(self, name: str) -> GoogleSatelliteEmbeddingV1Item:
        """Gets an item by name."""
        _, items_by_name = self._load_index()
        if name not in items_by_name:
            raise ValueError(f"item {name} not found")
        return items_by_name[name]

    def deserialize_item(self, serialized_item: dict) -> GoogleSatelliteEmbeddingV1Item:
        """Deserializes an item from JSON-decoded data."""
        return GoogleSatelliteEmbeddingV1Item.deserialize(serialized_item)

    def ingest(
        self,
        tile_store: Any,
        items: list[GoogleSatelliteEmbeddingV1Item],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        Note: Each file is 2-3GB so this can be slow. Direct materialization via
        read_raster or materialize is recommended for most use cases.

        Args:
            tile_store: the tile store to ingest into
            items: the items to ingest
            geometries: a list of geometries needed for each item
        """
        for item in items:
            if tile_store.is_raster_ready(item.name, BANDS):
                continue

            # Download the TIFF file directly to disk
            key = item.s3_path.replace(f"s3://{BUCKET_NAME}/", "")
            with tempfile.TemporaryDirectory() as tmp_dir:
                local_path = os.path.join(tmp_dir, f"{item.name}.tiff")
                self.s3_client.download_file(BUCKET_NAME, key, local_path)
                tile_store.write_raster_file(item.name, BANDS, UPath(local_path))

    # --- DirectMaterializeDataSource implementation ---

    def get_asset_url(self, item_name: str, asset_key: str) -> str:
        """Get the HTTP URL to read the asset.

        Returns a /vsicurl/ URL that rasterio can read directly over HTTP.
        """
        item = self.get_item_by_name(item_name)
        # Convert s3://bucket/path to HTTP URL
        key = item.s3_path.replace(f"s3://{BUCKET_NAME}/", "")
        return f"/vsicurl/{HTTP_URL_BASE}/{key}"

    def get_read_callback(
        self, item_name: str, asset_key: str
    ) -> Callable[[npt.NDArray[Any]], npt.NDArray[Any]] | None:
        """Return a callback to apply de-quantization if enabled."""
        if not self.apply_dequantization:
            return None

        def dequantize(data: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
            # Handle nodata (-128)
            nodata_mask = data == -128
            float_data = data.astype(np.float32)
            result = ((float_data / 127.5) ** 2) * np.sign(float_data)
            result[nodata_mask] = 0.0
            return result

        return dequantize

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

        Overrides base class to handle band selection (the base class reads all bands).
        """
        asset_url = self.get_asset_url(item_name, "image")

        # Determine which band indices to read (1-indexed for rasterio)
        if bands == BANDS:
            band_indices = list(range(1, 65))
        else:
            band_indices = [BANDS.index(b) + 1 for b in bands]

        # Construct the transform for the requested bounds
        wanted_transform = rasterio.transform.Affine(
            projection.x_resolution,
            0,
            bounds[0] * projection.x_resolution,
            0,
            projection.y_resolution,
            bounds[1] * projection.y_resolution,
        )

        with rasterio.open(asset_url) as src:
            with rasterio.vrt.WarpedVRT(
                src,
                crs=projection.crs,
                transform=wanted_transform,
                width=bounds[2] - bounds[0],
                height=bounds[3] - bounds[1],
                resampling=resampling,
            ) as vrt:
                data = vrt.read(indexes=band_indices)

        # Apply callback if dequantization is enabled
        callback = self.get_read_callback(item_name, "image")
        if callback is not None:
            data = callback(data)

        return data
