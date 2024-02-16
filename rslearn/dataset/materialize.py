"""Classes to implement dataset materialization."""

import os
from typing import Any

import numpy as np
import numpy.typing as npt
from class_registry import ClassRegistry

from rslearn.config import LayerConfig, RasterFormatConfig, RasterLayerConfig
from rslearn.const import TILE_SIZE
from rslearn.data_sources import Item
from rslearn.data_sources.raster_source import get_final_projection_and_bounds
from rslearn.tile_stores import TileStore, TileStoreLayer
from rslearn.utils.raster_format import load_raster_format

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
    dst: npt.NDArray[Any], ts_layer: TileStoreLayer, bounds: tuple[int, int, int, int]
) -> None:
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
                :,
                src_row_offset : src_row_offset + row_overlap,
                src_col_offset : src_col_offset + col_overlap,
            ]
            dst_crop[dst_crop == 0] = src_crop[dst_crop == 0]


def merge_raster_arrays(
    dst: npt.NDArray[Any], src: npt.NDArray[Any]
) -> npt.NDArray[Any]:
    """Merge src into dst.

    Non-zero pixels in dst are retained while others are overwritten by src.

    Args:
        dst: the array to merge into
        src: the array to merge from

    Returns:
        dst, or src if dst is None
    """
    if dst is None:
        return src
    dst[dst == 0] = src[dst == 0]
    return dst


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
            projection, bounds = get_final_projection_and_bounds(
                window.projection, window.bounds, band_cfg
            )
            raster_format = load_raster_format(RasterFormatConfig(band_cfg.format, {}))
            for group_id, group in enumerate(item_groups):
                if group_id == 0:
                    out_layer_name = layer_name
                else:
                    out_layer_name = f"{layer_name}.{group_id}"
                out_dir = os.path.join(window.window_root, "layers", out_layer_name)
                os.makedirs(out_dir, exist_ok=True)
                out_fname = os.path.join(
                    out_dir,
                    "".join(band_cfg.bands) + "." + raster_format.get_extension(),
                )
                if os.path.exists(out_fname):
                    continue

                dst = np.zeros(
                    (len(band_cfg.bands), bounds[3] - bounds[1], bounds[2] - bounds[0]),
                    dtype=band_cfg.dtype.name.lower(),
                )
                for item in group:
                    print(band_cfg.bands, item.name)
                    # TODO: have a mapping from band to the data source band it'd be
                    # stored in and then load things properly in case the band set
                    # doesn't match up.
                    # maybe tile store should support a list sub-layer option.
                    ts_layer = tile_store.get_layer(
                        (
                            layer_name,
                            item.name,
                            "".join(band_cfg.bands),
                            str(projection),
                        )
                    )

                    read_raster_window_from_tiles(dst, ts_layer, bounds)

                with open(out_fname, "wb") as f:
                    raster_format.encode_raster(f, projection, bounds, dst)
