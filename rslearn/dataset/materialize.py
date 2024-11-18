"""Classes to implement dataset materialization."""

from typing import Any, Generic, TypeVar

import numpy as np
import numpy.typing as npt
from class_registry import ClassRegistry
from upath import UPath

from rslearn.config import (
    LayerConfig,
    RasterFormatConfig,
    RasterLayerConfig,
    VectorLayerConfig,
)
from rslearn.data_sources import Item
from rslearn.tile_stores import TileStore, TileStoreLayer, get_tile_store_for_layer
from rslearn.utils import Feature, PixelBounds
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
        tile_store: TileStore,
        window: Window,
        layer_name: str,
        layer_cfg: LayerConfigType,
        item_groups: list[list[Item]],
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
    ts_layer: TileStoreLayer,
    bounds: PixelBounds,
    src_indexes: list[int],
    dst_indexes: list[int],
    remapper: Remapper | None = None,
) -> None:
    """Read a window of raster data from tiles in a tile store.

    Pixels in the destination array are only overwritten if not already non-zero.

    Args:
        dst: the destination numpy array
        ts_layer: the tile store layer to read
        bounds: the bounds in pixel coordinates matching projection of ts_layer
        src_indexes: the source band indexes to use
        dst_indexes: corresponding destination band indexes for each source band index
        remapper: optional remapper to apply on the source pixel values
    """
    src_bounds = ts_layer.get_raster_bounds()
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

    src = ts_layer.read_raster(intersection)
    if src is None:
        raise ValueError(f"No raster data found for bounds {intersection}")
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
        tile_store: TileStore,
        window: Window,
        layer_name: str,
        layer_cfg: RasterLayerConfig,
        item_groups: list[list[Item]],
    ) -> None:
        """Materialize portions of items corresponding to this window into the dataset.

        Args:
            tile_store: the tile store where the items have been ingested (unprefixed)
            window: the window to materialize
            layer_name: name of the layer to materialize
            layer_cfg: the configuration of the layer to materialize
            item_groups: the items associated with this window and layer
        """
        assert isinstance(layer_cfg, RasterLayerConfig)

        layer_tile_store = get_tile_store_for_layer(tile_store, layer_name, layer_cfg)

        out_layer_dirs: list[UPath] = []
        for group_id in range(len(item_groups)):
            if group_id == 0:
                out_layer_name = layer_name
            else:
                out_layer_name = f"{layer_name}.{group_id}"
            out_layer_dir = window.path / "layers" / out_layer_name
            out_layer_dirs.append(out_layer_dir)

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
                    needed_band_indexes = {}
                    for i, band in enumerate(band_cfg.bands):
                        needed_band_indexes[band] = i
                    suffixes = layer_tile_store.list_layers((item.name,))
                    needed_suffixes_and_indexes = []
                    for suffix in suffixes:
                        bands = suffix.split("_")
                        needed_src_indexes = []
                        needed_dst_indexes = []
                        for i, band in enumerate(bands):
                            if band not in needed_band_indexes:
                                continue
                            needed_src_indexes.append(i)
                            needed_dst_indexes.append(needed_band_indexes[band])
                            del needed_band_indexes[band]
                        if len(needed_src_indexes) == 0:
                            continue
                        needed_suffixes_and_indexes.append(
                            (suffix, needed_src_indexes, needed_dst_indexes)
                        )
                    if len(needed_band_indexes) > 0:
                        # This item doesn't have all the needed bands, so skip it.
                        continue

                    for suffix, src_indexes, dst_indexes in needed_suffixes_and_indexes:
                        ts_layer = layer_tile_store.get_layer(
                            (item.name, suffix, str(projection))
                        )
                        if ts_layer is None:
                            raise ValueError(
                                f"No tile store layer found for {item.name} {suffix} \
                                      {projection}"
                            )
                        read_raster_window_from_tiles(
                            dst, ts_layer, bounds, src_indexes, dst_indexes, remapper
                        )

                raster_format.encode_raster(
                    out_layer_dirs[group_id] / "_".join(band_cfg.bands),
                    projection,
                    bounds,
                    dst,
                )

        for out_layer_dir in out_layer_dirs:
            (out_layer_dir / "completed").touch()


@Materializers.register("vector")
class VectorMaterializer(Materializer):
    """A Materializer for vector data."""

    def materialize(
        self,
        tile_store: TileStore,
        window: Window,
        layer_name: str,
        layer_cfg: LayerConfig,
        item_groups: list[list[Item]],
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

        out_layer_dirs: list[UPath] = []
        for group_id in range(len(item_groups)):
            if group_id == 0:
                out_layer_name = layer_name
            else:
                out_layer_name = f"{layer_name}.{group_id}"
            out_layer_dir = window.path / "layers" / out_layer_name
            out_layer_dirs.append(out_layer_dir)

        for group_id, group in enumerate(item_groups):
            features: list[Feature] = []

            for item in group:
                ts_layer = get_tile_store_for_layer(
                    tile_store, layer_name, layer_cfg
                ).get_layer((item.name, str(projection)))
                if ts_layer is None:
                    raise ValueError(
                        f"No tile store layer found for {item.name} {projection}"
                    )
                cur_features = ts_layer.read_vector(bounds)
                features.extend(cur_features)

            vector_format.encode_vector(out_layer_dirs[group_id], projection, features)

        for out_layer_dir in out_layer_dirs:
            (out_layer_dir / "completed").touch()
