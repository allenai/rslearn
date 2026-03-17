"""Data source for AlphaEarth annual embeddings on Source Cooperative."""

import os
import shutil
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import rasterio
import rasterio.vrt
import shapely.wkb
from rasterio.enums import Resampling
from upath import UPath

import rslearn.data_sources.utils
from rslearn.config import QueryConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.data_source import (
    DataSourceContext,
    Item,
    ItemLookupDataSource,
)
from rslearn.data_sources.direct_materialize_data_source import (
    DirectMaterializeDataSource,
)
from rslearn.utils.fsspec import join_upath
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry
from rslearn.utils.grid_index import GridIndex
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import get_raster_projection_and_bounds

BANDS = [f"A{idx:02d}" for idx in range(64)]
RAW_NODATA_VALUE = -128
DEQUANTIZED_NODATA_VALUE = -2.0

DEFAULT_INDEX_URL = (
    "https://data.source.coop/tge-labs/aef/v1/annual/"
    "aef_index_stac_geoparquet.parquet"
)
SOURCE_COOP_S3_PREFIX = "s3://us-west-2.opendata.source.coop/"
SOURCE_COOP_HTTP_PREFIX = "https://data.source.coop/"
GRID_SIZE = 1.0


class AlphaEarthItem(Item):
    """An item in the AlphaEarth data source."""

    def __init__(
        self,
        name: str,
        geometry: STGeometry,
        data_href: str,
    ) -> None:
        """Creates a new AlphaEarthItem.

        Args:
            name: unique name of the STAC item.
            geometry: the spatial and temporal extent of the item.
            data_href: href for the item's data asset.
        """
        super().__init__(name, geometry)
        self.data_href = data_href

    def serialize(self) -> dict:
        """Serializes the item to a JSON-encodable dictionary."""
        d = super().serialize()
        d["data_href"] = self.data_href
        return d

    @staticmethod
    def deserialize(d: dict) -> "AlphaEarthItem":
        """Deserializes an item from a JSON-decoded dictionary."""
        item = super(AlphaEarthItem, AlphaEarthItem).deserialize(d)
        return AlphaEarthItem(
            name=item.name,
            geometry=item.geometry,
            data_href=d["data_href"],
        )


def s3_to_https(href: str) -> str:
    """Convert Source Cooperative S3 hrefs to HTTPS URLs."""
    if href.startswith(SOURCE_COOP_S3_PREFIX):
        return SOURCE_COOP_HTTP_PREFIX + href.removeprefix(SOURCE_COOP_S3_PREFIX)
    return href


def _download_url(url: str, local_path: str) -> None:
    """Download a remote file to a local path."""
    with urllib.request.urlopen(url) as src, open(local_path, "wb") as dst:
        shutil.copyfileobj(src, dst)


class AlphaEarth(
    DirectMaterializeDataSource[AlphaEarthItem],
    ItemLookupDataSource[AlphaEarthItem],
):
    """Data source for AlphaEarth annual embeddings on Source Cooperative.

    This data source indexes the public GeoParquet STAC catalog hosted on
    Source Cooperative and reads the 64-band Cloud-Optimized GeoTIFF assets over
    HTTPS.

    Available years: 2018-2024.
    """

    def __init__(
        self,
        metadata_cache_dir: str,
        index_url: str = DEFAULT_INDEX_URL,
        apply_dequantization: bool = True,
        context: DataSourceContext = DataSourceContext(),
    ) -> None:
        """Initialize a new AlphaEarth instance.

        Args:
            metadata_cache_dir: directory to cache the GeoParquet index.
            index_url: URL or local path to the GeoParquet STAC index.
            apply_dequantization: whether to convert raw int8 values to float32 in
                [-1, 1]. The raw data uses -128 as nodata, which becomes -2.0 after
                dequantization so the nodata sentinel remains outside the valid
                embedding range.
            context: the data source context.
        """
        super().__init__(asset_bands={"image": BANDS})

        self.index_url = index_url
        self.apply_dequantization = apply_dequantization

        if context.ds_path is not None:
            self.metadata_cache_dir = join_upath(context.ds_path, metadata_cache_dir)
        else:
            self.metadata_cache_dir = UPath(metadata_cache_dir)
        self.metadata_cache_dir.mkdir(parents=True, exist_ok=True)

        self._grid_index: GridIndex | None = None
        self._items_by_name: dict[str, AlphaEarthItem] | None = None

    def _get_duckdb(self) -> Any:
        try:
            import duckdb
        except ImportError as exc:
            raise ImportError(
                "AlphaEarth requires duckdb. Install rslearn with the [extra] "
                "dependencies to use this data source."
            ) from exc
        return duckdb

    def _connect_duckdb(self) -> Any:
        duckdb = self._get_duckdb()
        con = duckdb.connect()
        try:
            con.execute("LOAD spatial;")
        except duckdb.Error:
            con.execute("INSTALL spatial;")
            con.execute("LOAD spatial;")
        return con

    def _get_local_index_path(self) -> str:
        parsed = urllib.parse.urlparse(self.index_url)
        if parsed.scheme in ("", "file"):
            return str(Path(parsed.path or self.index_url).expanduser().resolve())

        cache_name = Path(parsed.path).name or "alphaearth_index.parquet"
        cache_file = self.metadata_cache_dir / cache_name
        if not cache_file.exists():
            _download_url(self.index_url, os.fspath(cache_file))
        return os.fspath(cache_file)

    def _load_index(self) -> tuple[GridIndex, dict[str, AlphaEarthItem]]:
        """Load the GeoParquet index and build a spatial lookup index."""
        if self._grid_index is not None and self._items_by_name is not None:
            return self._grid_index, self._items_by_name

        con = self._connect_duckdb()
        index_path = self._get_local_index_path()
        rows = con.execute(
            """
            SELECT
              id,
              datetime,
              assets.data.href AS data_href,
              ST_AsWKB(geometry) AS geometry_wkb
            FROM read_parquet(?)
            """,
            [index_path],
        ).fetchall()

        grid_index = GridIndex(GRID_SIZE)
        items_by_name: dict[str, AlphaEarthItem] = {}

        for item_id, item_datetime, data_href, geometry_wkb in rows:
            shp = shapely.wkb.loads(geometry_wkb)

            year = item_datetime.astimezone(UTC).year
            time_range = (
                datetime(year, 1, 1, tzinfo=UTC),
                datetime(year, 12, 31, 23, 59, 59, tzinfo=UTC),
            )

            geometry = STGeometry(WGS84_PROJECTION, shp, time_range)
            item = AlphaEarthItem(
                name=item_id,
                geometry=geometry,
                data_href=data_href,
            )
            grid_index.insert(shp.bounds, item)
            items_by_name[item_id] = item

        self._grid_index = grid_index
        self._items_by_name = items_by_name
        return grid_index, items_by_name

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[list[AlphaEarthItem]]]:
        """Get items in the data source intersecting the given geometries."""
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
                if wgs84_geometry.time_range is not None:
                    item_start, item_end = item.geometry.time_range
                    query_start, query_end = wgs84_geometry.time_range
                    if item_end < query_start or item_start > query_end:
                        continue
                cur_items.append(item)

            cur_items.sort(key=lambda item: item.geometry.time_range[0])
            cur_groups: list[list[AlphaEarthItem]] = (
                rslearn.data_sources.utils.match_candidate_items_to_window(
                    geometry, cur_items, query_config
                )
            )
            groups.append(cur_groups)

        return groups

    def get_item_by_name(self, name: str) -> AlphaEarthItem:
        """Gets an item by name."""
        _, items_by_name = self._load_index()
        if name not in items_by_name:
            raise ValueError(f"item {name} not found")
        return items_by_name[name]

    def deserialize_item(self, serialized_item: dict) -> AlphaEarthItem:
        """Deserializes an item from JSON-decoded data."""
        return AlphaEarthItem.deserialize(serialized_item)

    def ingest(
        self,
        tile_store: Any,
        items: list[AlphaEarthItem],
        geometries: list[list[STGeometry]],
    ) -> None:
        """Ingest items into the given tile store.

        This is supported, but direct materialization is recommended instead since
        annual AlphaEarth TIFFs are large, around 3 GB each.
        """
        for item in items:
            if tile_store.is_raster_ready(item, BANDS):
                continue

            with tempfile.TemporaryDirectory() as tmp_dir:
                local_path = os.path.join(tmp_dir, f"{item.name}.tiff")
                _download_url(self.get_asset_url(item, "image"), local_path)

                if self.apply_dequantization:
                    with rasterio.open(local_path) as src:
                        array = src.read()
                        projection, bounds = get_raster_projection_and_bounds(src)
                    array = self._dequantize(array)
                    tile_store.write_raster(
                        item,
                        BANDS,
                        projection,
                        bounds,
                        RasterArray(
                            chw_array=array,
                            time_range=item.geometry.time_range,
                        ),
                    )
                else:
                    tile_store.write_raster_file(
                        item,
                        BANDS,
                        UPath(local_path),
                        time_range=item.geometry.time_range,
                    )

    def get_asset_url(self, item: AlphaEarthItem, asset_key: str) -> str:
        """Get the HTTPS URL for an AlphaEarth asset."""
        return s3_to_https(item.data_href)

    def get_default_nodata_value(self) -> float:
        """Get the default nodata value for AlphaEarth rasters."""
        if self.apply_dequantization:
            return DEQUANTIZED_NODATA_VALUE
        return float(RAW_NODATA_VALUE)

    def get_default_nodata_vals(self, bands: list[str]) -> list[float] | None:
        """Get default nodata values for AlphaEarth bands."""
        if not set(bands).issubset(BANDS):
            return None
        return [self.get_default_nodata_value() for _ in bands]

    def _dequantize(self, data: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        """Apply de-quantization to convert int8 values to float32."""
        nodata_mask = data == RAW_NODATA_VALUE
        float_data = data.astype(np.float32)
        result = ((float_data / 127.5) ** 2) * np.sign(float_data)
        result[nodata_mask] = DEQUANTIZED_NODATA_VALUE
        return result

    def get_read_callback(
        self, item: AlphaEarthItem, asset_key: str
    ) -> Callable[[npt.NDArray[Any]], npt.NDArray[Any]] | None:
        """Return a callback to apply de-quantization if enabled."""
        if not self.apply_dequantization:
            return None
        return self._dequantize

    def read_raster(
        self,
        layer_name: str,
        item: Item,
        bands: list[str],
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
    ) -> RasterArray:
        """Read raster data from the remote AlphaEarth COG."""
        if not isinstance(item, AlphaEarthItem):
            raise TypeError(f"expected AlphaEarthItem, got {type(item)}")

        asset_url = self.get_asset_url(item, "image")
        if bands == BANDS:
            band_indices = list(range(1, 65))
        else:
            band_indices = [BANDS.index(band) + 1 for band in bands]

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

        callback = self.get_read_callback(item, "image")
        if callback is not None:
            data = callback(data)

        return RasterArray(chw_array=data, time_range=item.geometry.time_range)
