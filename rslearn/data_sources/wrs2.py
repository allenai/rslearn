"""Utilities for working with Landsat WRS-2 path/row polygons."""

import shutil
import urllib.request
import zipfile

import fiona
import shapely
import shapely.geometry
from upath import UPath

from rslearn.const import SHAPEFILE_AUX_EXTENSIONS
from rslearn.utils.fsspec import get_upath_local
from rslearn.utils.geometry import STGeometry, flatten_shape
from rslearn.utils.grid_index import GridIndex

WRS2_GRID_SIZE = 1.0

WRS2_URL = "https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/WRS2_descending_0.zip"


def get_wrs2_polygons(
    cache_dir: UPath,
    buffer_degrees: float = 0.2,
) -> list[tuple[shapely.Geometry, str, str]]:
    """Get polygons for each (path, row) in the WRS2 grid.

    Args:
        cache_dir: local directory to cache downloaded shapefile artifacts.
        buffer_degrees: amount to buffer each WRS polygon. Landsat scenes often
            extend slightly beyond nominal WRS polygons.

    Returns:
        List of (polygon, zero-padded path, zero-padded row).
    """
    prefix = "WRS2_descending"
    shp_fname = cache_dir / f"{prefix}.shp"
    if not shp_fname.exists():
        zip_fname = cache_dir / f"{prefix}.zip"
        with urllib.request.urlopen(WRS2_URL) as response:
            with zip_fname.open("wb") as f:
                shutil.copyfileobj(response, f)

        with zip_fname.open("rb") as f:
            with zipfile.ZipFile(f, "r") as zipf:
                member_names = zipf.namelist()
                for ext in SHAPEFILE_AUX_EXTENSIONS:
                    cur_fname = prefix + ext
                    if cur_fname not in member_names:
                        continue
                    with zipf.open(cur_fname) as memberf:
                        with (cache_dir / (prefix + ext)).open("wb") as out_f:
                            shutil.copyfileobj(memberf, out_f)

                with zipf.open(f"{prefix}.shp") as memberf:
                    with shp_fname.open("wb") as out_f:
                        shutil.copyfileobj(memberf, out_f)

    aux_files = [cache_dir / (prefix + ext) for ext in SHAPEFILE_AUX_EXTENSIONS]
    with get_upath_local(shp_fname, extra_paths=aux_files) as local_fname:
        with fiona.open(local_fname) as src:
            polygons = []
            for feat in src:
                shp = shapely.geometry.shape(feat["geometry"])
                shp = shp.buffer(buffer_degrees)
                path = str(feat["properties"]["PATH"]).zfill(3)
                row = str(feat["properties"]["ROW"]).zfill(3)
                polygons.append((shp, path, row))
            return polygons


def build_wrs2_grid_index(cache_dir: UPath) -> GridIndex:
    """Build a GridIndex over buffered WRS2 polygons."""
    grid_index = GridIndex(WRS2_GRID_SIZE)
    for polygon, path, row in get_wrs2_polygons(cache_dir):
        grid_index.insert(polygon.bounds, (polygon, path, row))
    return grid_index


def get_pathrows_for_geometry(
    wrs2_index: GridIndex,
    wgs84_geometry: STGeometry,
) -> set[tuple[str, str]]:
    """Get WRS2 path/row pairs intersecting a geometry in WGS84."""
    pathrows = set()
    for shp in flatten_shape(wgs84_geometry.shp):
        for polygon, path, row in wrs2_index.query(shp.bounds):
            if wgs84_geometry.shp.intersects(polygon):
                pathrows.add((path, row))
    return pathrows
