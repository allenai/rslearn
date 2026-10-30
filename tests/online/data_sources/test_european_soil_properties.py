"""Online test for the BOKU European soil texture data source."""

import pathlib

import numpy as np
import shapely
from rasterio import CRS

from rslearn.config import QueryConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.european_soil_properties import (
    TEXTURE_BANDS,
    EuropeanSoilProperties,
)
from rslearn.utils.geometry import Projection, STGeometry


def test_read_raster(tmp_path: pathlib.Path) -> None:
    """Read a small agricultural area east of Vienna from the public COG."""
    projection = Projection(CRS.from_epsg(32633), 10, -10)
    point = STGeometry(WGS84_PROJECTION, shapely.Point(16.7, 48.2), None)
    projected_point = point.to_projection(projection).shp
    size = 32
    geometry = STGeometry(
        projection,
        shapely.box(
            projected_point.x - size // 2,
            projected_point.y - size // 2,
            projected_point.x + size // 2,
            projected_point.y + size // 2,
        ),
        None,
    )

    data_source = EuropeanSoilProperties(cache_dir=str(tmp_path / "cache"))
    groups = data_source.get_items(
        [geometry], QueryConfig(space_mode=SpaceMode.INTERSECTS)
    )[0]
    assert len(groups) == 1
    assert len(groups[0].items) == 1

    min_x, min_y, max_x, max_y = geometry.shp.bounds
    bounds = (int(min_x), int(min_y), int(max_x), int(max_y))
    raster = data_source.read_raster(
        layer_name="soil_texture",
        item=groups[0].items[0],
        bands=TEXTURE_BANDS,
        projection=projection,
        bounds=bounds,
    )
    array = raster.get_chw_array()

    assert array.shape == (3, size, size)
    valid = np.all(np.isfinite(array), axis=0)
    assert valid.any()
    assert np.all((array[:, valid] >= 0) & (array[:, valid] <= 100))
    np.testing.assert_allclose(array[:, valid].sum(axis=0), 100, atol=1.5)
