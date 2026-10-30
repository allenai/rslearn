"""European 10 m soil texture maps published by BOKU University.

The dataset contains 122 Cloud-Optimized GeoTIFF tiles covering the EU-27 and the
United Kingdom. This source currently exposes the clay, sand, and silt predictions
from the 0--20 cm soil layer as their original byte-encoded fractions.

See https://doi.org/10.1038/s41597-026-08273-1 and
https://doi.org/10.5281/zenodo.21281853 for details.
"""

import json
from typing import cast

import affine
import numpy as np
import rasterio
import rasterio.vrt
import shapely
from rasterio.enums import Resampling
from typing_extensions import override
from upath import UPath

import rslearn.data_sources.utils
from rslearn.config import DType, QueryConfig
from rslearn.config.dataset import ResamplingMethod
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.data_source import (
    DataSourceContext,
    Item,
    ItemLookupDataSource,
)
from rslearn.data_sources.direct_materialize_data_source import (
    DirectMaterializeDataSource,
)
from rslearn.data_sources.utils import MatchedItemGroup
from rslearn.utils.fsspec import join_upath, open_atomic
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry
from rslearn.utils.grid_index import GridIndex
from rslearn.utils.raster_array import RasterArray, RasterMetadata

BASE_URL = "https://igeocloud.boku.ac.at/public.php/dav/files/9KQqb8pxPrArTta/Textures"

TEXTURE_BANDS = ["clay", "sand", "silt"]

TILE_COUNT = 122
GRID_INDEX_CELL_SIZE = 3.0


class EuropeanSoilProperties(
    DirectMaterializeDataSource[Item], ItemLookupDataSource[Item]
):
    """Read the BOKU high-resolution European clay, sand, and silt maps."""

    def __init__(
        self,
        cache_dir: str,
        context: DataSourceContext | None = None,
    ) -> None:
        """Create a European soil texture data source.

        Args:
            cache_dir: directory for the tile metadata cache, resolved relative to
                the dataset path when available. Delete the cache to refresh it.
            context: data source context supplied by rslearn.
        """
        super().__init__(asset_bands={"textures": TEXTURE_BANDS})
        if context is None:
            context = DataSourceContext()

        if context.layer_config is not None:
            band_sets = context.layer_config.band_sets
            if len(band_sets) != 1 or band_sets[0].bands != TEXTURE_BANDS:
                raise ValueError(
                    "EuropeanSoilProperties requires one band set with bands "
                    f"{TEXTURE_BANDS}"
                )
            if band_sets[0].dtype != DType.FLOAT32:
                raise ValueError(
                    "EuropeanSoilProperties requires dtype=float32 because "
                    "empty pixels are NaN"
                )
            if context.layer_config.resampling_method != ResamplingMethod.NEAREST:
                raise ValueError(
                    "EuropeanSoilProperties requires resampling_method=nearest "
                    "to keep texture values aligned with their validity indicators"
                )

        self._grid_index = GridIndex(GRID_INDEX_CELL_SIZE)
        self._items_by_name: dict[str, Item] = {}
        if context.ds_path is not None:
            self.cache_dir = join_upath(context.ds_path, cache_dir)
        else:
            self.cache_dir = UPath(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for item in self._load_items():
            self._grid_index.insert(item.geometry.shp.bounds, item)
            self._items_by_name[item.name] = item

    def _load_items(self) -> list[Item]:
        """Read cached tile metadata, or discover bounds from the COG headers."""
        cache_path = self.cache_dir / "tile_index.json"
        if cache_path.exists():
            with cache_path.open() as f:
                return [Item.deserialize(item) for item in json.load(f)]

        items = []
        for tile_number in range(1, TILE_COUNT + 1):
            name = f"Tile_{tile_number}"
            with rasterio.open(f"{BASE_URL}/{name}.tif") as src:
                if src.crs != WGS84_PROJECTION.crs:
                    raise ValueError(f"expected EPSG:4326 for {name}, got {src.crs}")
                geometry = STGeometry(WGS84_PROJECTION, shapely.box(*src.bounds), None)
            items.append(Item(name=name, geometry=geometry))

        # Publish only a complete index so failed discovery is retried next time.
        with open_atomic(cache_path, "w") as f:
            json.dump([item.serialize() for item in items], f)
        return items

    @override
    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[MatchedItemGroup[Item]]]:
        """Return texture tiles intersecting each requested geometry."""
        groups = []
        for geometry in geometries:
            wgs84_geometry = geometry.to_projection(WGS84_PROJECTION)
            candidate_items = [
                item
                for item in self._grid_index.query(wgs84_geometry.shp.bounds)
                if wgs84_geometry.shp.intersects(item.geometry.shp)
            ]
            groups.append(
                rslearn.data_sources.utils.match_candidate_items_to_window(
                    geometry, candidate_items, query_config
                )
            )
        return groups

    @override
    def get_item_by_name(self, name: str) -> Item:
        """Return a tile by its ``Tile_N`` name."""
        try:
            return self._items_by_name[name]
        except KeyError as exc:
            raise ValueError(f"European soil texture tile {name!r} not found") from exc

    @override
    def deserialize_item(self, serialized_item: dict) -> Item:
        return Item.deserialize(serialized_item)

    @override
    def get_asset_url(self, item: Item, asset_key: str) -> str:
        """Return the public range-readable COG URL for a tile."""
        if asset_key != "textures":
            raise ValueError(f"unknown asset key {asset_key!r}")
        return f"{BASE_URL}/{item.name}.tif"

    @override
    def get_raster_metadata(
        self, layer_name: str, item: Item, bands: list[str]
    ) -> RasterMetadata:
        """Return metadata for the unscaled output bands."""
        if bands != TEXTURE_BANDS:
            raise ValueError(f"expected bands {TEXTURE_BANDS}, got {bands}")
        return RasterMetadata(nodata_value=float("nan"))

    @override
    def read_raster(
        self,
        layer_name: str,
        item: Item,
        bands: list[str],
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.nearest,
    ) -> RasterArray:
        """Read, reproject, and mask the unscaled texture bands."""
        if bands != TEXTURE_BANDS:
            raise ValueError(f"expected bands {TEXTURE_BANDS}, got {bands}")

        if resampling != Resampling.nearest:
            raise ValueError(
                "EuropeanSoilProperties requires nearest-neighbour resampling"
            )

        typed_item = cast(Item, item)
        wanted_transform = affine.Affine(
            projection.x_resolution,
            0,
            bounds[0] * projection.x_resolution,
            0,
            projection.y_resolution,
            bounds[1] * projection.y_resolution,
        )

        with (
            rasterio.open(self.get_asset_url(typed_item, "textures")) as src,
            rasterio.vrt.WarpedVRT(
                src,
                crs=projection.crs,
                transform=wanted_transform,
                width=bounds[2] - bounds[0],
                height=bounds[3] - bounds[1],
                dtype="float32",
                resampling=resampling,
            ) as vrt,
        ):
            data = vrt.read(indexes=[1, 2, 3, 4, 5, 6])

        # Float32 represents every source byte exactly and supports NaN nodata.
        values = data[:3]
        # The provider defines an empty pixel as a value/uncertainty pair that
        # is both zero. Nearest-neighbour keeps each pair from the same pixel.
        empty = (data[:3] == 0) & (data[3:] == 0)
        values[empty] = np.nan

        return RasterArray(
            chw_array=values,
            time_range=typed_item.geometry.time_range,
            metadata=RasterMetadata(nodata_value=float("nan")),
        )
