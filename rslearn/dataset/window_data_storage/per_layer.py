"""WindowDataStorage that packs all item groups into one raster file per layer.

Stores combined rasters at ``layers/{layer_name}/{bandset_dir}/...`` with a
``window_storage_meta.json`` sidecar recording each group's number of
timesteps. Item groups are concatenated along the T axis, so reading back any
individual group requires the sidecar to know which T-slice corresponds to
which group. Vector data is not supported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
from rasterio.enums import Resampling
from typing_extensions import override
from upath import UPath

from rslearn.utils.array import unique_nodata_value
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import PixelBounds, Projection
from rslearn.utils.raster_array import RasterArray, RasterMetadata
from rslearn.utils.raster_format import RasterFormat, get_bandset_dirname
from rslearn.utils.vector_format import VectorFormat

from .per_item_group import PerItemGroupStorage
from .storage import LayerWriter, WindowDataStorage, WindowDataStorageFactory

if TYPE_CHECKING:
    from rslearn.dataset.window import Window


PER_LAYER_STORAGE_META_FNAME = "window_storage_meta.json"


def _per_layer_raster_dir(
    window_root: UPath, layer_name: str, bands: list[str]
) -> UPath:
    """Per-layer raster directory: ``layers/{layer_name}/{bandset}/``.

    Note this matches the location used by ``PerItemGroupStorage`` for
    ``group_idx=0``. ``PerLayerStorage`` is a distinct on-disk layout because
    of the sidecar at :data:`PER_LAYER_STORAGE_META_FNAME` and the combined
    T axis.
    """
    dirname = get_bandset_dirname(bands)
    return window_root / "layers" / layer_name / dirname


class _PerLayerStorageMeta:
    """Sidecar describing how groups are packed in a per-layer raster file."""

    def __init__(self, group_timestep_counts: list[int]) -> None:
        """Initialize the sidecar with the timestep count for each group."""
        self.group_timestep_counts = group_timestep_counts

    @property
    def num_groups(self) -> int:
        """Number of item groups stored in this band set."""
        return len(self.group_timestep_counts)

    def serialize(self) -> dict[str, Any]:
        """Return a JSON-encodable representation."""
        return {"group_timestep_counts": self.group_timestep_counts}

    @staticmethod
    def deserialize(d: dict[str, Any]) -> _PerLayerStorageMeta:
        """Inverse of :meth:`serialize`."""
        return _PerLayerStorageMeta(
            group_timestep_counts=list(d["group_timestep_counts"])
        )

    @staticmethod
    def read(raster_dir: UPath) -> _PerLayerStorageMeta:
        """Read the sidecar at ``raster_dir / window_storage_meta.json``."""
        with (raster_dir / PER_LAYER_STORAGE_META_FNAME).open() as f:
            return _PerLayerStorageMeta.deserialize(json.load(f))

    def write(self, raster_dir: UPath) -> None:
        """Write the sidecar to ``raster_dir / window_storage_meta.json``."""
        raster_dir.mkdir(parents=True, exist_ok=True)
        with (raster_dir / PER_LAYER_STORAGE_META_FNAME).open("w") as f:
            json.dump(self.serialize(), f)

    def t_slice_for_group(self, group_idx: int) -> slice:
        """Return the slice along T that corresponds to ``group_idx``."""
        if group_idx < 0 or group_idx >= self.num_groups:
            raise ValueError(
                f"group_idx {group_idx} out of range [0, {self.num_groups})"
            )
        start = sum(self.group_timestep_counts[:group_idx])
        end = start + self.group_timestep_counts[group_idx]
        return slice(start, end)


def _split_per_layer_raster(
    raster: RasterArray, meta: _PerLayerStorageMeta
) -> list[RasterArray]:
    """Split a packed CTHW raster back into one RasterArray per item group."""
    out: list[RasterArray] = []
    for group_idx in range(meta.num_groups):
        t_slice = meta.t_slice_for_group(group_idx)
        sub_array = raster.array[:, t_slice, :, :]
        sub_timestamps: list[tuple[datetime, datetime]] | None
        if raster.timestamps is None:
            sub_timestamps = None
        else:
            sub_timestamps = list(raster.timestamps[t_slice])
        out.append(
            RasterArray(
                array=sub_array,
                timestamps=sub_timestamps,
                metadata=RasterMetadata(nodata_value=raster.metadata.nodata_value),
            )
        )
    return out


@dataclass
class _BandsetBuffer:
    """Per-bandset buffer accumulated by :class:`_PerLayerStorageLayerWriter`."""

    bands: list[str]
    projection: Projection
    bounds: PixelBounds
    raster_format: RasterFormat
    # group_idx -> RasterArray
    rasters: dict[int, RasterArray]


class _PerLayerStorageLayerWriter(LayerWriter):
    """Writer for :class:`PerLayerStorage`.

    Raster writes are buffered until ``__exit__``; vector writes fall through
    to the per-item-group on-disk layout immediately, since
    :class:`PerLayerStorage` does not pack vector data.
    """

    def __init__(self, window: Window, layer_name: str) -> None:
        """Initialize a writer for ``window``/``layer_name``."""
        self._window = window
        self._layer_name = layer_name
        # bandset_key (from get_bandset_dirname) -> _BandsetBuffer
        self._buffers: dict[str, _BandsetBuffer] = {}
        # Vector data is not packed by PerLayerStorage, so we delegate vector
        # writes to a per-item-group writer.
        self._vector_writer = PerItemGroupStorage().open_layer_writer(
            window, layer_name
        )

    @override
    def write_raster(
        self,
        bands: list[str],
        raster_format: RasterFormat,
        projection: Projection,
        bounds: PixelBounds,
        raster: RasterArray,
        group_idx: int = 0,
    ) -> None:
        """Buffer the raster for ``group_idx`` (flushed on context exit)."""
        key = get_bandset_dirname(bands)
        buf = self._buffers.get(key)
        if buf is None:
            self._buffers[key] = _BandsetBuffer(
                bands=bands,
                projection=projection,
                bounds=bounds,
                raster_format=raster_format,
                rasters={group_idx: raster},
            )
            return

        if projection != buf.projection:
            raise ValueError(
                f"PerLayerStorage requires consistent projection across groups; "
                f"group {group_idx} projection {projection} != {buf.projection}"
            )
        if bounds != buf.bounds:
            raise ValueError(
                f"PerLayerStorage requires consistent bounds across groups; "
                f"group {group_idx} bounds {bounds} != {buf.bounds}"
            )
        if group_idx in buf.rasters:
            raise ValueError(
                f"PerLayerStorage already received group_idx={group_idx} for bands={bands}"
            )
        buf.rasters[group_idx] = raster

    @override
    def write_vector(
        self,
        vector_format: VectorFormat,
        features: list[Feature],
        group_idx: int = 0,
    ) -> None:
        """Encode ``features`` per-item-group (PerLayerStorage doesn't pack vector)."""
        self._vector_writer.write_vector(vector_format, features, group_idx=group_idx)

    @override
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Flush buffered groups to a single combined raster, unless an exc occurred."""
        self._vector_writer.__exit__(exc_type, exc, tb)
        if exc_type is not None:
            return None
        for buf in self._buffers.values():
            self._flush_bandset(buf)
        self._buffers = {}
        return None

    def _flush_bandset(self, buf: _BandsetBuffer) -> None:
        """Concatenate buffered groups along T and write the combined raster."""
        # Sort groups by group_idx so the on-disk T axis order is deterministic.
        group_idxs = sorted(buf.rasters.keys())
        if group_idxs != list(range(len(group_idxs))):
            raise ValueError(
                f"PerLayerStorage requires contiguous group indices [0, ..., N-1], "
                f"but got {group_idxs} for bands={buf.bands}"
            )
        arrays = [buf.rasters[i] for i in group_idxs]

        nodata = unique_nodata_value(
            [
                r.metadata.nodata_value
                for r in arrays
                if r.metadata.nodata_value is not None
            ]
        )

        # Either all groups have timestamps or none do; otherwise drop timestamps.
        all_have_timestamps = all(r.timestamps is not None for r in arrays)
        timestamps: list[tuple[datetime, datetime]] | None = None
        if all_have_timestamps:
            ts: list[tuple[datetime, datetime]] = []
            for r in arrays:
                assert r.timestamps is not None
                ts.extend(r.timestamps)
            timestamps = ts

        combined_array = np.concatenate([r.array for r in arrays], axis=1)
        meta = _PerLayerStorageMeta(
            group_timestep_counts=[r.array.shape[1] for r in arrays],
        )
        raster = RasterArray(
            array=combined_array,
            timestamps=timestamps,
            metadata=RasterMetadata(nodata_value=nodata),
        )

        raster_dir = _per_layer_raster_dir(
            self._window.window_root, self._layer_name, buf.bands
        )
        raster_dir.mkdir(parents=True, exist_ok=True)
        buf.raster_format.encode_raster(raster_dir, buf.projection, buf.bounds, raster)
        meta.write(raster_dir)


class PerLayerStorage(WindowDataStorage):
    """Storage that packs all item groups for a layer into one raster file.

    Raster on-disk layout: ``layers/{layer_name}/{bandset_dir}/`` with a
    ``window_storage_meta.json`` sidecar recording each group's number of
    timesteps. Item groups are concatenated along the T axis. This is more
    efficient when reading all groups at once but does not support concurrent
    or incremental writes.

    Vector data is not packed by this storage; vector reads and writes fall
    through to :class:`PerItemGroupStorage`.
    """

    def __init__(self) -> None:
        """Initialize the storage."""
        # PerLayerStorage delegates vector ops to PerItemGroupStorage so
        # mixed-modality datasets work out of the box.
        self._per_item_group_storage = PerItemGroupStorage()

    @override
    def open_layer_writer(
        self,
        window: Window,
        layer_name: str,
    ) -> LayerWriter:
        """Return a writer that buffers all groups before flushing."""
        return _PerLayerStorageLayerWriter(window, layer_name)

    @override
    def read_raster(
        self,
        window: Window,
        layer_name: str,
        bands: list[str],
        raster_format: RasterFormat,
        projection: Projection,
        bounds: PixelBounds,
        group_idx: int = 0,
        resampling: Resampling = Resampling.bilinear,
    ) -> RasterArray:
        """Decode the combined raster and slice out the requested item group."""
        raster_dir = _per_layer_raster_dir(window.window_root, layer_name, bands)
        meta = _PerLayerStorageMeta.read(raster_dir)
        combined = raster_format.decode_raster(
            raster_dir, projection, bounds, resampling
        )
        if combined.array.shape[1] != sum(meta.group_timestep_counts):
            raise ValueError(
                f"PerLayerStorage: combined raster T={combined.array.shape[1]} "
                f"does not match sidecar sum {sum(meta.group_timestep_counts)}"
            )
        t_slice = meta.t_slice_for_group(group_idx)
        sub_timestamps: list[tuple[datetime, datetime]] | None
        if combined.timestamps is None:
            sub_timestamps = None
        else:
            sub_timestamps = list(combined.timestamps[t_slice])
        return RasterArray(
            array=combined.array[:, t_slice, :, :],
            timestamps=sub_timestamps,
            metadata=RasterMetadata(nodata_value=combined.metadata.nodata_value),
        )

    @override
    def read_all_rasters(
        self,
        window: Window,
        layer_name: str,
        bands: list[str],
        num_groups: int,
        raster_format: RasterFormat,
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
    ) -> list[RasterArray]:
        """Decode the combined raster once and split into per-group RasterArrays."""
        raster_dir = _per_layer_raster_dir(window.window_root, layer_name, bands)
        meta = _PerLayerStorageMeta.read(raster_dir)
        if meta.num_groups != num_groups:
            raise ValueError(
                f"PerLayerStorage: expected {num_groups} groups but sidecar has "
                f"{meta.num_groups}"
            )
        combined = raster_format.decode_raster(
            raster_dir, projection, bounds, resampling
        )
        if combined.array.shape[1] != sum(meta.group_timestep_counts):
            raise ValueError(
                f"PerLayerStorage: combined raster T={combined.array.shape[1]} "
                f"does not match sidecar sum {sum(meta.group_timestep_counts)}"
            )
        return _split_per_layer_raster(combined, meta)

    @override
    def read_vector(
        self,
        window: Window,
        layer_name: str,
        vector_format: VectorFormat,
        projection: Projection,
        bounds: PixelBounds,
        group_idx: int = 0,
    ) -> list[Feature]:
        """Decode vector features per-item-group (we don't currently handle per-layer vector)."""
        return self._per_item_group_storage.read_vector(
            window,
            layer_name,
            vector_format,
            projection,
            bounds,
            group_idx=group_idx,
        )


class PerLayerStorageFactory(WindowDataStorageFactory):
    """Factory for :class:`PerLayerStorage`."""

    @override
    def get_storage(self, ds_path: UPath) -> PerLayerStorage:
        """Return a :class:`PerLayerStorage` (does not depend on ``ds_path``)."""
        return PerLayerStorage()
