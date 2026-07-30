"""Online integration tests for the Copernicus GLO-30 data source (hits real S3).

These tests access the public AWS ``copernicus-dem-30m`` bucket and report the
time consumption of the different stages (item discovery and ingest). Run with
``-s`` to see the timing output, e.g.::

    uv run pytest tests/online/data_sources/test_aws_glo30.py -s
"""

import pathlib
import time

import pytest
from upath import UPath

from rslearn.config import QueryConfig, SpaceMode
from rslearn.data_sources.aws_glo30 import CopernicusGLO30
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils.geometry import STGeometry


def _make_tile_store(tmp_path: pathlib.Path, name: str) -> DefaultTileStore:
    tile_store_dir = UPath(tmp_path / name)
    tile_store = DefaultTileStore(str(tile_store_dir))
    tile_store.set_dataset_path(tile_store_dir)
    return tile_store


def test_glo30_aws_access(tmp_path: pathlib.Path, seattle2020: STGeometry) -> None:
    """Access the real AWS GLO-30 bucket and report timing (elevation only)."""
    data_source = CopernicusGLO30()
    # Restrict to elevation only for a clean single-band timing measurement.
    data_source.band_names = ["elevation"]
    data_source._needs_slope = False
    data_source._needs_aspect = False

    t0 = time.perf_counter()
    query_config = QueryConfig(space_mode=SpaceMode.MOSAIC, max_matches=1)
    item_groups = data_source.get_items([seattle2020], query_config)[0]
    t_get_items = time.perf_counter() - t0

    assert len(item_groups) == 1
    assert len(item_groups[0].items) >= 1
    item = item_groups[0].items[0]
    # Seattle (lon=-122.33, lat=47.61) falls in the N47/W123 tile.
    assert "N47" in item.name and "W123" in item.name

    tile_store = _make_tile_store(tmp_path, "tiles")
    layer_name = "layer"

    t0 = time.perf_counter()
    data_source.ingest(
        TileStoreWithLayer(tile_store, layer_name),
        item_groups[0].items,
        [[seattle2020]],
    )
    t_ingest = time.perf_counter() - t0

    assert tile_store.is_raster_ready(layer_name, item, ["elevation"])

    print("\n[GLO-30 AWS] elevation-only")
    print(f"  tiles ingested : {len(item_groups[0].items)}")
    print(f"  get_items      : {t_get_items:.3f} s")
    print(f"  ingest         : {t_ingest:.3f} s")


def test_glo30_aws_with_slope_aspect(
    tmp_path: pathlib.Path, seattle2020: STGeometry
) -> None:
    """Access the real AWS GLO-30 bucket and report timing (elevation+slope+aspect)."""
    data_source = CopernicusGLO30()
    assert data_source.band_names == ["elevation", "slope", "aspect"]

    query_config = QueryConfig(space_mode=SpaceMode.MOSAIC, max_matches=1)
    item_groups = data_source.get_items([seattle2020], query_config)[0]
    item = item_groups[0].items[0]

    tile_store = _make_tile_store(tmp_path, "tiles")
    layer_name = "layer"

    t0 = time.perf_counter()
    data_source.ingest(
        TileStoreWithLayer(tile_store, layer_name),
        item_groups[0].items,
        [[seattle2020]],
    )
    t_ingest = time.perf_counter() - t0

    assert tile_store.is_raster_ready(
        layer_name, item, ["elevation", "slope", "aspect"]
    )

    print("\n[GLO-30 AWS] elevation+slope+aspect")
    print(f"  tiles ingested : {len(item_groups[0].items)}")
    print(f"  ingest (incl. slope/aspect compute): {t_ingest:.3f} s")


def test_glo30_timing_comparison(
    tmp_path: pathlib.Path, seattle2020: STGeometry
) -> None:
    """Compare AWS vs Planetary Computer ingest time for the same window.

    Skipped if the optional ``planetary_computer`` dependency is not installed.
    """
    planetary_computer = pytest.importorskip("planetary_computer")
    assert planetary_computer  # silence unused-import linters
    from rslearn.data_sources.planetary_computer import CopDemGlo30

    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)

    # --- AWS ---
    aws = CopernicusGLO30()
    aws.band_names = ["elevation"]
    aws._needs_slope = False
    aws._needs_aspect = False

    t0 = time.perf_counter()
    aws_groups = aws.get_items(
        [seattle2020], QueryConfig(space_mode=SpaceMode.MOSAIC, max_matches=1)
    )[0]
    aws_get_items = time.perf_counter() - t0

    aws_item = aws_groups[0].items[0]
    aws_store = _make_tile_store(tmp_path, "aws")
    t0 = time.perf_counter()
    aws.ingest(
        TileStoreWithLayer(aws_store, "layer"),
        aws_groups[0].items,
        [[seattle2020]],
    )
    aws_ingest = time.perf_counter() - t0
    assert aws_store.is_raster_ready("layer", aws_item, ["elevation"])

    # --- Planetary Computer ---
    pc = CopDemGlo30(band_name="DEM")

    t0 = time.perf_counter()
    pc_groups = pc.get_items([seattle2020], query_config)[0]
    pc_get_items = time.perf_counter() - t0

    pc_item = pc_groups[0].items[0]
    pc_store = _make_tile_store(tmp_path, "pc")
    t0 = time.perf_counter()
    pc.ingest(
        TileStoreWithLayer(pc_store, "layer"),
        pc_groups[0].items,
        [[seattle2020]],
    )
    pc_ingest = time.perf_counter() - t0
    assert pc_store.is_raster_ready("layer", pc_item, ["DEM"])

    print("\n[GLO-30 timing comparison] (Seattle, elevation only)")
    print(f"  {'stage':<12} {'AWS (s)':>10} {'PC (s)':>10}")
    print(f"  {'get_items':<12} {aws_get_items:>10.3f} {pc_get_items:>10.3f}")
    print(f"  {'ingest':<12} {aws_ingest:>10.3f} {pc_ingest:>10.3f}")
    print(
        f"  {'total':<12} {aws_get_items + aws_ingest:>10.3f} "
        f"{pc_get_items + pc_ingest:>10.3f}"
    )
