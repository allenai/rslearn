"""European 10 m soil texture maps published by BOKU University.

The dataset contains 122 Cloud-Optimized GeoTIFF tiles covering the EU-27 and the
United Kingdom. This source currently exposes the clay, sand, and silt predictions
from the 0--20 cm soil layer as percentages.

See https://doi.org/10.1038/s41597-026-08273-1 and
https://doi.org/10.5281/zenodo.21281853 for details.
"""

from typing import cast

import affine
import numpy as np
import rasterio
import rasterio.vrt
import shapely
from rasterio.enums import Resampling
from typing_extensions import override

import rslearn.data_sources.utils
from rslearn.config import DType, QueryConfig
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
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry
from rslearn.utils.grid_index import GridIndex
from rslearn.utils.raster_array import RasterArray, RasterMetadata

BASE_URL = "https://igeocloud.boku.ac.at/public.php/dav/files/9KQqb8pxPrArTta/Textures"

TEXTURE_BANDS = ["clay", "sand", "silt"]

# Source pixels are bytes whose three texture values sum to approximately 255.
# The provider describes the public values as percentages, so convert the byte
# fractions to the 0--100 range. The corresponding uncertainty bands are only read
# to identify empty pixels, as documented by the provider.
PERCENT_SCALE = 100.0 / 255.0

# The published tiles form a regular 3-degree grid, but only cells intersecting the
# dataset coverage were exported. These rows are ordered exactly like Tile_1 through
# Tile_122. Values refer to indices in X_ORIGINS.
X_ORIGINS = (
    -10.4782190,
    -7.4782052,
    -4.4781915,
    -1.4782676,
    1.5217461,
    4.5217598,
    7.5217735,
    10.5217873,
    13.5218010,
    16.5218147,
    19.5217386,
    22.5217523,
    25.5217660,
    28.5217798,
    31.5217935,
)
ROW_MIN_Y_AND_X_INDICES = (
    (68.1270633, (9, 10, 11, 12, 13)),
    (65.1270496, (8, 9, 10, 11, 12, 13)),
    (62.1270359, (7, 8, 9, 10, 11, 12, 13, 14)),
    (59.1270222, (2, 3, 7, 8, 9, 10, 11, 12, 13)),
    (56.1270983, (1, 2, 6, 7, 8, 9, 10, 11, 12)),
    (53.1270846, (0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12)),
    (50.1270708, tuple(range(0, 12))),
    (47.1270571, tuple(range(1, 13))),
    (44.1270434, tuple(range(2, 14))),
    (41.1270297, tuple(range(0, 14))),
    (38.1270159, (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12)),
    (35.1270920, (0, 1, 2, 3, 7, 8, 10, 11, 12)),
    (32.1270783, (11, 12)),
)

# Actual tile dimensions are just over three degrees because the rasters contain a
# whole number of ~10 m pixels. This margin matches their published bounding boxes.
TILE_SIZE_DEGREES = 3.0002
GRID_INDEX_CELL_SIZE = 3.0


class EuropeanSoilProperties(
    DirectMaterializeDataSource[Item], ItemLookupDataSource[Item]
):
    """Read the BOKU high-resolution European clay, sand, and silt maps."""

    def __init__(self, context: DataSourceContext = DataSourceContext()) -> None:
        """Create a European soil texture data source.

        Args:
            context: data source context supplied by rslearn.
        """
        super().__init__(asset_bands={"textures": TEXTURE_BANDS})

        if context.layer_config is not None:
            band_sets = context.layer_config.band_sets
            if len(band_sets) != 1 or band_sets[0].bands != TEXTURE_BANDS:
                raise ValueError(
                    "EuropeanSoilProperties requires one band set with bands "
                    f"{TEXTURE_BANDS}"
                )
            if band_sets[0].dtype != DType.FLOAT32:
                raise ValueError(
                    "EuropeanSoilProperties requires dtype=float32 because values "
                    "are scaled to percentages and empty pixels are NaN"
                )

        self._grid_index = GridIndex(GRID_INDEX_CELL_SIZE)
        self._items_by_name: dict[str, Item] = {}
        tile_number = 1
        for min_y, x_indices in ROW_MIN_Y_AND_X_INDICES:
            for x_index in x_indices:
                min_x = X_ORIGINS[x_index]
                shp = shapely.box(
                    min_x,
                    min_y,
                    min_x + TILE_SIZE_DEGREES,
                    min_y + TILE_SIZE_DEGREES,
                )
                item = Item(
                    name=f"Tile_{tile_number}",
                    geometry=STGeometry(WGS84_PROJECTION, shp, None),
                )
                self._grid_index.insert(shp.bounds, item)
                self._items_by_name[item.name] = item
                tile_number += 1

        assert tile_number == 123

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
        """Return metadata for the scaled output bands."""
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
        resampling: Resampling = Resampling.bilinear,
    ) -> RasterArray:
        """Read, reproject, mask, and scale the three texture bands."""
        if bands != TEXTURE_BANDS:
            raise ValueError(f"expected bands {TEXTURE_BANDS}, got {bands}")

        typed_item = cast(Item, item)
        wanted_transform = affine.Affine(
            projection.x_resolution,
            0,
            bounds[0] * projection.x_resolution,
            0,
            projection.y_resolution,
            bounds[1] * projection.y_resolution,
        )

        with rasterio.open(self.get_asset_url(typed_item, "textures")) as src:
            with rasterio.vrt.WarpedVRT(
                src,
                crs=projection.crs,
                transform=wanted_transform,
                width=bounds[2] - bounds[0],
                height=bounds[3] - bounds[1],
                resampling=resampling,
            ) as vrt:
                raw_values = vrt.read(indexes=[1, 2, 3])

            # The provider defines an empty pixel as a value/uncertainty pair that
            # is both zero. Always resample those pairs with nearest-neighbour so
            # interpolation does not bleed values into empty source pixels.
            with rasterio.vrt.WarpedVRT(
                src,
                crs=projection.crs,
                transform=wanted_transform,
                width=bounds[2] - bounds[0],
                height=bounds[3] - bounds[1],
                resampling=Resampling.nearest,
            ) as mask_vrt:
                mask_data = mask_vrt.read(indexes=[1, 2, 3, 4, 5, 6])

        values = raw_values.astype(np.float32) * PERCENT_SCALE
        empty = (mask_data[:3] == 0) & (mask_data[3:] == 0)
        values[empty] = np.nan

        # The model produces compositional fractions summing to one before they are
        # quantized to bytes. Restore that invariant after reprojection; this also
        # prevents interpolation with adjacent empty pixels from depressing the sum.
        valid = np.all(np.isfinite(values), axis=0)
        totals = values.sum(axis=0)
        normalizable = valid & (totals > 0)
        values[:, normalizable] *= 100.0 / totals[normalizable]

        return RasterArray(
            chw_array=values,
            time_range=typed_item.geometry.time_range,
            metadata=RasterMetadata(nodata_value=float("nan")),
        )
