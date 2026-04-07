"""Classes to implement dataset materialization."""

from __future__ import annotations

from datetime import datetime

from rslearn.config import (
    BandSetConfig,
    LayerConfig,
)
from rslearn.data_sources.data_source import ItemType
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import PixelBounds, Projection
from rslearn.utils.raster_array import RasterArray

from .compositing import Compositor
from .remap import Remapper, load_remapper
from .tile_utils import get_needed_band_sets_and_indexes
from .window import Window


class Materializer:
    """An abstract class that materializes data from a tile store."""

    def materialize(
        self,
        tile_store: TileStoreWithLayer,
        window: Window,
        layer_name: str,
        layer_cfg: LayerConfig,
        item_groups: list[list[ItemType]],
        group_time_ranges: list[tuple[datetime, datetime] | None] | None = None,
    ) -> None:
        """Materialize portions of items corresponding to this window into the dataset.

        Args:
            tile_store: the tile store where the items have been ingested (unprefixed)
            window: the window to materialize
            layer_name: the name of the layer to materialize
            layer_cfg: the configuration of the layer to materialize
            item_groups: the items associated with this window and layer
            group_time_ranges: optional request time range for each item group
        """
        raise NotImplementedError


def resolve_nodata_values(
    tile_store: TileStoreWithLayer,
    items: list[ItemType],
    bands: list[str],
) -> tuple[float, ...] | None:
    """Resolve per-band nodata values from the tile store metadata.

    Probes the first item that has matching bands and reads nodata from the
    raster file header (no pixel data is read).  Returns ``None`` when no
    item has matching bands or no source declares nodata metadata.

    Args:
        tile_store: the tile store to query.
        items: candidate items (from one or more item groups).
        bands: the requested band names.

    Returns:
        A tuple of nodata values (one per band), or ``None`` when the source
        has no nodata.
    """
    for item in items:
        needed = get_needed_band_sets_and_indexes(item, bands, tile_store)
        if not needed:
            continue
        resolved = [0.0] * len(bands)
        has_nodata = False
        for src_bands, src_indexes, dst_indexes in needed:
            metadata = tile_store.get_raster_metadata(item, src_bands)
            if metadata.nodata_values is not None:
                has_nodata = True
                for src_idx, dst_idx in zip(src_indexes, dst_indexes):
                    resolved[dst_idx] = float(metadata.nodata_values[src_idx])
        return tuple(resolved) if has_nodata else None
    return None


def build_composite(
    group: list[ItemType],
    compositor: Compositor,
    tile_store: TileStoreWithLayer,
    layer_cfg: LayerConfig,
    band_cfg: BandSetConfig,
    projection: Projection,
    bounds: PixelBounds,
    remapper: Remapper | None,
    request_time_range: tuple[datetime, datetime] | None = None,
) -> RasterArray:
    """Build a composite for specified bands from items in the group.

    Args:
        group: list of items to composite together
        compositor: Compositor instance that implements the compositing logic.
        tile_store: tile store containing the raster data
        layer_cfg: the configuration of the layer to materialize
        band_cfg: the configuration of the layer to materialize. Contains the bands to process.
        projection: spatial projection for the composite
        bounds: pixel bounds defining the spatial extent of the composite
        remapper: remapper to apply to pixel values, or None
        request_time_range: optional request time range, passed through to compositing method.

    Returns:
        A RasterArray produced by the chosen compositing method.
    """
    nodata_vals: tuple[int | float, ...] | None = band_cfg.nodata_vals
    if nodata_vals is None:
        nodata_vals = resolve_nodata_values(tile_store, group, band_cfg.bands)

    return compositor.build_composite(
        group=group,
        nodata_vals=nodata_vals,
        bands=band_cfg.bands,
        bounds=bounds,
        band_dtype=band_cfg.dtype.get_numpy_dtype(),
        tile_store=tile_store,
        projection=projection,
        resampling_method=layer_cfg.resampling_method.get_rasterio_resampling(),
        remapper=remapper,
        request_time_range=request_time_range,
    )


# ---------------------------------------------------------------------------
# Materializers
# ---------------------------------------------------------------------------


class RasterMaterializer(Materializer):
    """A Materializer for raster data."""

    def materialize(
        self,
        tile_store: TileStoreWithLayer,
        window: Window,
        layer_name: str,
        layer_cfg: LayerConfig,
        item_groups: list[list[ItemType]],
        group_time_ranges: list[tuple[datetime, datetime] | None] | None = None,
    ) -> None:
        """Materialize portions of items corresponding to this window into the dataset.

        Args:
            tile_store: the tile store where the items have been ingested
            window: the window to materialize
            layer_name: name of the layer to materialize
            layer_cfg: the configuration of the layer to materialize
            item_groups: the items associated with this window and layer
            group_time_ranges: optional request time range for each item group
        """
        if group_time_ranges is not None and len(group_time_ranges) != len(item_groups):
            raise ValueError(
                "group_time_ranges length must match item_groups length during materialization"
            )

        if layer_cfg.data_source is not None:
            default_request_time_range = layer_cfg.data_source.get_request_time_range(
                window.time_range
            )
        else:
            default_request_time_range = window.time_range

        compositor = layer_cfg.instantiate_compositor()

        for band_cfg in layer_cfg.band_sets:
            projection, bounds = band_cfg.get_final_projection_and_bounds(
                window.projection, window.bounds
            )

            remapper = None
            if band_cfg.remap:
                remapper = load_remapper(band_cfg.remap)

            raster_format = band_cfg.instantiate_raster_format()

            for group_id, group in enumerate(item_groups):
                request_time_range = (
                    group_time_ranges[group_id]
                    if group_time_ranges is not None
                    else default_request_time_range
                )
                raster = build_composite(
                    group=group,
                    compositor=compositor,
                    tile_store=tile_store,
                    layer_cfg=layer_cfg,
                    band_cfg=band_cfg,
                    projection=projection,
                    bounds=bounds,
                    remapper=remapper,
                    request_time_range=request_time_range,
                )

                raster_format.encode_raster(
                    window.get_raster_dir(layer_name, band_cfg.bands, group_id),
                    projection,
                    bounds,
                    raster,
                )

        for group_id in range(len(item_groups)):
            window.mark_layer_completed(layer_name, group_id)


class VectorMaterializer(Materializer):
    """A Materializer for vector data."""

    def materialize(
        self,
        tile_store: TileStoreWithLayer,
        window: Window,
        layer_name: str,
        layer_cfg: LayerConfig,
        item_groups: list[list[ItemType]],
        group_time_ranges: list[tuple[datetime, datetime] | None] | None = None,
    ) -> None:
        """Materialize portions of items corresponding to this window into the dataset.

        Args:
            tile_store: the tile store where the items have been ingested (unprefixed)
            window: the window to materialize
            layer_name: the layer to materialize
            layer_cfg: the configuration of the layer to materialize
            item_groups: the items associated with this window and layer
            group_time_ranges: unused for vector materialization
        """
        vector_format = layer_cfg.instantiate_vector_format()

        for group_id, group in enumerate(item_groups):
            features: list[Feature] = []

            for item in group:
                cur_features = tile_store.read_vector(
                    item, window.projection, window.bounds
                )
                features.extend(cur_features)

            vector_format.encode_vector(
                window.get_layer_dir(layer_name, group_id), features
            )

        for group_id in range(len(item_groups)):
            window.mark_layer_completed(layer_name, group_id)
