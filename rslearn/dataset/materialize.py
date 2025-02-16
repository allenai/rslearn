"""Classes to implement dataset materialization."""

from typing import Any, Generic, TypeVar

import numpy as np
import numpy.typing as npt
from class_registry import ClassRegistry
from rasterio.enums import Resampling

from rslearn.config import (
    LayerConfig,
    RasterFormatConfig,
    RasterLayerConfig,
    VectorLayerConfig,
)
from rslearn.data_sources.data_source import ItemType
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import PixelBounds, Projection
from rslearn.utils.raster_format import load_raster_format
from rslearn.utils.vector_format import load_vector_format

from .remap import Remapper, load_remapper
from .window import Window

Materializers = ClassRegistry()

LayerConfigType = TypeVar("LayerConfigType", bound=LayerConfig)


class Materializer(Generic[LayerConfigType]):
    """An abstract class that materializes data from a tile store."""

    def materialize(
        self,
        tile_store: TileStoreWithLayer,
        window: Window,
        layer_name: str,
        layer_cfg: LayerConfigType,
        item_groups: list[list[ItemType]],
    ) -> None:
        """Materialize portions of items corresponding to this window into the dataset.

        Args:
            tile_store: the tile store where the items have been ingested (unprefixed)
            window: the window to materialize
            layer_name: the name of the layer to materialize
            layer_cfg: the configuration of the layer to materialize
            item_groups: the items associated with this window and layer
        """
        raise NotImplementedError


def read_raster_window_from_tiles(
    dst: npt.NDArray[Any],
    tile_store: TileStoreWithLayer,
    item_name: str,
    bands: list[str],
    projection: Projection,
    bounds: PixelBounds,
    src_indexes: list[int],
    dst_indexes: list[int],
    remapper: Remapper | None = None,
    resampling: Resampling = Resampling.bilinear,
) -> None:
    """Read a window of raster data from tiles in a tile store.

    Pixels in the destination array are only overwritten if not already non-zero.

    Args:
        dst: the destination numpy array
        tile_store: the TileStore to read from.
        item_name: the item name.
        bands: the bands that identify the raster we want to read.
        projection: the projection of the dst array.
        bounds: the bounds of the dst array.
        src_indexes: the source band indexes to use
        dst_indexes: corresponding destination band indexes for each source band index
        remapper: optional remapper to apply on the source pixel values
        resampling: how to resample the pixels in case re-projection is needed.
    """
    # Only read the portion of the raster that overlaps with dst.
    # This way we can avoid creating big arrays that are all empty which speeds things
    # up for large windows.
    src_bounds = tile_store.get_raster_bounds(item_name, bands, projection)
    intersection = (
        max(bounds[0], src_bounds[0]),
        max(bounds[1], src_bounds[1]),
        min(bounds[2], src_bounds[2]),
        min(bounds[3], src_bounds[3]),
    )
    if intersection[2] <= intersection[0] or intersection[3] <= intersection[1]:
        return

    dst_col_offset = intersection[0] - bounds[0]
    dst_row_offset = intersection[1] - bounds[1]

    src = tile_store.read_raster(
        item_name, bands, projection, intersection, resampling=resampling
    )
    src = src[src_indexes, :, :]
    if remapper:
        src = remapper(src, dst.dtype)

    dst_crop = dst[
        :,
        dst_row_offset : dst_row_offset + src.shape[1],
        dst_col_offset : dst_col_offset + src.shape[2],
    ]
    mask = dst_crop[dst_indexes, :, :].max(axis=0) == 0
    for src_index, dst_index in enumerate(dst_indexes):
        dst_crop[dst_index, mask] = src[src_index, mask]


@Materializers.register("raster")
class RasterMaterializer(Materializer[RasterLayerConfig]):
    """A Materializer for raster data."""

    def materialize(
        self,
        tile_store: TileStoreWithLayer,
        window: Window,
        layer_name: str,
        layer_cfg: RasterLayerConfig,
        item_groups: list[list[ItemType]],
    ) -> None:
        """Materialize portions of items corresponding to this window into the dataset.

        Args:
            tile_store: the tile store where the items have been ingested
            window: the window to materialize
            layer_name: name of the layer to materialize
            layer_cfg: the configuration of the layer to materialize
            item_groups: the items associated with this window and layer
        """
        assert isinstance(layer_cfg, RasterLayerConfig)

        for band_cfg in layer_cfg.band_sets:
            # band_cfg could specify zoom_offset and maybe other parameters that affect
            # projection/bounds, so use the corrected projection/bounds.
            projection, bounds = band_cfg.get_final_projection_and_bounds(
                window.projection, window.bounds
            )

            # Also load remapper if set.
            remapper = None
            if band_cfg.remap:
                remapper = load_remapper(band_cfg.remap)

            if band_cfg.format is None or band_cfg.bands is None or bounds is None:
                raise ValueError(
                    f"No raster format or bands specified for {layer_name} \
                          with  {band_cfg}"
                )

            raster_format = load_raster_format(
                RasterFormatConfig(band_cfg.format["name"], band_cfg.format)
            )

            for group_id, group in enumerate(item_groups):
                dst = np.zeros(
                    (len(band_cfg.bands), bounds[3] - bounds[1], bounds[2] - bounds[0]),
                    dtype=band_cfg.dtype.value,
                )
                for item in group:
                    # Identify which tile store layer(s) to read to get the configured
                    # bands.
                    wanted_band_indexes = {}
                    for i, band in enumerate(band_cfg.bands):
                        wanted_band_indexes[band] = i

                    available_bands = tile_store.get_raster_bands(item.name)
                    needed_band_sets_and_indexes = []
                    for band_set in available_bands:
                        needed_src_indexes = []
                        needed_dst_indexes = []
                        for i, band in enumerate(band_set):
                            if band not in wanted_band_indexes:
                                continue
                            needed_src_indexes.append(i)
                            needed_dst_indexes.append(wanted_band_indexes[band])
                            del wanted_band_indexes[band]
                        if len(needed_src_indexes) == 0:
                            continue
                        needed_band_sets_and_indexes.append(
                            (band_set, needed_src_indexes, needed_dst_indexes)
                        )
                    if len(wanted_band_indexes) > 0:
                        # This item doesn't have all the needed bands, so skip it.
                        continue

                    for (
                        band_set,
                        src_indexes,
                        dst_indexes,
                    ) in needed_band_sets_and_indexes:
                        read_raster_window_from_tiles(
                            dst,
                            tile_store,
                            item.name,
                            band_set,
                            projection,
                            bounds,
                            src_indexes,
                            dst_indexes,
                            remapper,
                            resampling=layer_cfg.resampling_method,
                        )

                raster_format.encode_raster(
                    window.get_raster_dir(layer_name, band_cfg.bands, group_id),
                    projection,
                    bounds,
                    dst,
                )

        for group_id in range(len(item_groups)):
            window.mark_layer_completed(layer_name, group_id)


@Materializers.register("vector")
class VectorMaterializer(Materializer):
    """A Materializer for vector data."""

    def materialize(
        self,
        tile_store: TileStoreWithLayer,
        window: Window,
        layer_name: str,
        layer_cfg: LayerConfig,
        item_groups: list[list[ItemType]],
    ) -> None:
        """Materialize portions of items corresponding to this window into the dataset.

        Args:
            tile_store: the tile store where the items have been ingested (unprefixed)
            window: the window to materialize
            layer_name: the layer to materialize
            layer_cfg: the configuration of the layer to materialize
            item_groups: the items associated with this window and layer
        """
        assert isinstance(layer_cfg, VectorLayerConfig)

        projection, bounds = layer_cfg.get_final_projection_and_bounds(
            window.projection, window.bounds
        )
        if bounds is None:
            raise ValueError(f"No bounds specified for {layer_name}")
        vector_format = load_vector_format(layer_cfg.format)

        for group_id, group in enumerate(item_groups):
            features: list[Feature] = []

            for item in group:
                cur_features = tile_store.read_vector(item.name, projection, bounds)
                features.extend(cur_features)

            vector_format.encode_vector(
                window.get_layer_dir(layer_name, group_id), projection, features
            )

        for group_id in range(len(item_groups)):
            window.mark_layer_completed(layer_name, group_id)
