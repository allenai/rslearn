"""Classes to implement dataset materialization."""

import os
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from class_registry import ClassRegistry

from rslearn.config import LayerConfig, RasterFormatConfig, RasterLayerConfig
from rslearn.const import TILE_SIZE
from rslearn.data_sources import Item
from rslearn.tile_stores import TileStore, TileStoreLayer
from rslearn.utils.raster_format import load_raster_format

from .remap import Remapper, load_remapper
from .window import Window

Materializers = ClassRegistry()


class Materializer:
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
            layer_cfg: the configuration of the layer to materialize
            item_groups: the items associated with this window and layer
        """
        raise NotImplementedError


def read_raster_window_from_tiles(
    dst: npt.NDArray[Any],
    ts_layer: TileStoreLayer,
    bounds: tuple[int, int, int, int],
    src_indexes: list[int],
    dst_indexes: list[int],
    remapper: Optional[Remapper] = None,
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
    # Load tiles one at a time.
    start_tile = (bounds[0] // TILE_SIZE, bounds[1] // TILE_SIZE)
    end_tile = ((bounds[2] - 1) // TILE_SIZE, (bounds[3] - 1) // TILE_SIZE)
    for i in range(start_tile[0], end_tile[0] + 1):
        for j in range(start_tile[1], end_tile[1] + 1):
            src = ts_layer.get_raster(i, j)
            if src is None:
                continue

            cur_col_off = TILE_SIZE * i
            cur_row_off = TILE_SIZE * j

            src_col_offset = max(bounds[0] - cur_col_off, 0)
            src_row_offset = max(bounds[1] - cur_row_off, 0)
            dst_col_offset = max(cur_col_off - bounds[0], 0)
            dst_row_offset = max(cur_row_off - bounds[1], 0)
            col_overlap = min(
                src.shape[2] - src_col_offset, dst.shape[2] - dst_col_offset
            )
            row_overlap = min(
                src.shape[1] - src_row_offset, dst.shape[1] - dst_row_offset
            )
            dst_crop = dst[
                :,
                dst_row_offset : dst_row_offset + row_overlap,
                dst_col_offset : dst_col_offset + col_overlap,
            ]
            src_crop = src[
                src_indexes,
                src_row_offset : src_row_offset + row_overlap,
                src_col_offset : src_col_offset + col_overlap,
            ]
            src_crop = remapper(src_crop, dst_crop.dtype) if remapper else src_crop
            mask = dst_crop[dst_indexes, :, :].max(axis=0) == 0
            dst_crop[dst_indexes, mask] = src_crop[:, mask]


@Materializers.register("raster")
class RasterMaterializer(Materializer):
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
            if band_cfg.remap_config:
                remapper = load_remapper(band_cfg.remap_config)
            print("remapper", remapper)

            raster_format = load_raster_format(RasterFormatConfig(band_cfg.format, {}))

            for group_id, group in enumerate(item_groups):
                if group_id == 0:
                    out_layer_name = layer_name
                else:
                    out_layer_name = f"{layer_name}.{group_id}"

                # Create output directory and skip processing this group if it's
                # already materialized.
                out_dir = os.path.join(window.window_root, "layers", out_layer_name)
                os.makedirs(out_dir, exist_ok=True)
                out_fname = os.path.join(
                    out_dir,
                    "_".join(band_cfg.bands) + "." + raster_format.get_extension(),
                )
                if os.path.exists(out_fname):
                    continue

                dst = np.zeros(
                    (len(band_cfg.bands), bounds[3] - bounds[1], bounds[2] - bounds[0]),
                    dtype=band_cfg.dtype.value,
                )
                for item in group:
                    print(band_cfg.bands, item.name)
                    # Identify which tile store layer(s) to read to get the configured
                    # bands.
                    needed_band_indexes = {}
                    for i, band in enumerate(band_cfg.bands):
                        needed_band_indexes[band] = i
                    suffixes = tile_store.list_layers((layer_name, item.name))
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
                        ts_layer = tile_store.get_layer(
                            (
                                layer_name,
                                item.name,
                                suffix,
                                str(projection),
                            )
                        )
                        print(suffix, src_indexes, dst_indexes)
                        read_raster_window_from_tiles(
                            dst, ts_layer, bounds, src_indexes, dst_indexes, remapper
                        )

                with open(out_fname, "wb") as f:
                    raster_format.encode_raster(f, projection, bounds, dst)
