"""Online integration tests for the Copernicus GLO-30 data source (hits real S3).

These tests access the public AWS ``copernicus-dem-30m`` bucket and report the time
consumption of the different stages (item discovery and direct raster reads). Run
with ``-s`` to see the timing output, e.g.::

    uv run pytest tests/online/data_sources/test_aws_glo30.py -s
"""

import pathlib
import time

import numpy as np
import pytest
import shapely

from rslearn.config import QueryConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.aws_glo30 import CopernicusGLO30
from rslearn.utils.geometry import PixelBounds, STGeometry

MOSAIC_QUERY = QueryConfig(space_mode=SpaceMode.MOSAIC, max_matches=1)


def _pixel_bounds(geometry: STGeometry) -> PixelBounds:
    return (
        int(geometry.shp.bounds[0]),
        int(geometry.shp.bounds[1]),
        int(geometry.shp.bounds[2]),
        int(geometry.shp.bounds[3]),
    )


def test_glo30_aws_access(tmp_path: pathlib.Path, seattle2020: STGeometry) -> None:
    """Read elevation directly from the real AWS GLO-30 bucket and report timing."""
    data_source = CopernicusGLO30(metadata_cache_dir=str(tmp_path / "cache"))

    t0 = time.perf_counter()
    item_groups = data_source.get_items([seattle2020], MOSAIC_QUERY)[0]
    t_get_items = time.perf_counter() - t0

    assert len(item_groups) == 1
    items = item_groups[0].items
    assert len(items) >= 1
    item = items[0]
    # Seattle (lon=-122.33, lat=47.61) falls in the N47/W123 tile.
    assert "N47" in item.name and "W123" in item.name

    t0 = time.perf_counter()
    array = data_source.read_raster(
        layer_name="layer",
        item=item,
        bands=["elevation"],
        projection=seattle2020.projection,
        bounds=_pixel_bounds(seattle2020),
    )
    t_read = time.perf_counter() - t0

    chw = array.get_chw_array()
    assert chw.shape[0] == 1
    # Seattle is near sea level but above it, and well below Mount Rainier.
    valid = chw[np.isfinite(chw)]
    assert valid.size > 0
    assert valid.min() > -100
    assert valid.max() < 5000

    print("\n[GLO-30 AWS] direct materialization")
    print(f"  tiles matched : {len(items)}")
    print(f"  get_items     : {t_get_items:.3f} s")
    print(f"  read_raster   : {t_read:.3f} s")


def test_glo30_skips_ocean_tiles(tmp_path: pathlib.Path) -> None:
    """A window over open ocean should match no tiles rather than failing."""
    data_source = CopernicusGLO30(metadata_cache_dir=str(tmp_path / "cache"))
    # A patch of the middle of the Pacific Ocean.
    ocean = STGeometry(WGS84_PROJECTION, shapely.box(-140.5, 20.2, -140.2, 20.4), None)
    item_groups = data_source.get_items([ocean], MOSAIC_QUERY)[0]
    assert item_groups[0].items == []


def test_glo30_timing_comparison(
    tmp_path: pathlib.Path, seattle2020: STGeometry
) -> None:
    """Compare AWS vs Planetary Computer direct read time for the same window.

    Skipped if the optional ``planetary_computer`` dependency is not installed.
    """
    planetary_computer = pytest.importorskip("planetary_computer")
    assert planetary_computer  # silence unused-import linters
    from rslearn.data_sources.planetary_computer import CopDemGlo30

    bounds = _pixel_bounds(seattle2020)

    # --- AWS ---
    aws = CopernicusGLO30(metadata_cache_dir=str(tmp_path / "cache"))
    t0 = time.perf_counter()
    aws_groups = aws.get_items([seattle2020], MOSAIC_QUERY)[0]
    aws_get_items = time.perf_counter() - t0

    aws_item = aws_groups[0].items[0]
    t0 = time.perf_counter()
    aws.read_raster(
        layer_name="layer",
        item=aws_item,
        bands=["elevation"],
        projection=seattle2020.projection,
        bounds=bounds,
    )
    aws_read = time.perf_counter() - t0

    # --- Planetary Computer ---
    pc = CopDemGlo30(band_name="DEM")
    t0 = time.perf_counter()
    pc_groups = pc.get_items(
        [seattle2020], QueryConfig(space_mode=SpaceMode.INTERSECTS)
    )[0]
    pc_get_items = time.perf_counter() - t0

    pc_item = pc_groups[0].items[0]
    t0 = time.perf_counter()
    pc.read_raster(
        layer_name="layer",
        item=pc_item,
        bands=["DEM"],
        projection=seattle2020.projection,
        bounds=bounds,
    )
    pc_read = time.perf_counter() - t0

    print("\n[GLO-30 timing comparison] (Seattle, direct materialization)")
    print(f"  {'stage':<12} {'AWS (s)':>10} {'PC (s)':>10}")
    print(f"  {'get_items':<12} {aws_get_items:>10.3f} {pc_get_items:>10.3f}")
    print(f"  {'read_raster':<12} {aws_read:>10.3f} {pc_read:>10.3f}")
    print(
        f"  {'total':<12} {aws_get_items + aws_read:>10.3f} "
        f"{pc_get_items + pc_read:>10.3f}"
    )
