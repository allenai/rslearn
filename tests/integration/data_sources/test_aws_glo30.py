"""Mocked integration tests for the Copernicus GLO-30 data source."""

import io
import pathlib
from typing import Any
from unittest.mock import MagicMock

import boto3
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from upath import UPath

from rslearn.config import (
    BandSetConfig,
    DType,
    LayerConfig,
    LayerType,
    QueryConfig,
    SpaceMode,
)
from rslearn.data_sources import Item
from rslearn.data_sources.aws_glo30 import GLO30_BUCKET, CopernicusGLO30, _tile_name
from rslearn.dataset import Window
from rslearn.dataset.storage.file import FileWindowStorage
from rslearn.dataset.window_data_storage.per_item_group import (
    PerItemGroupStorageFactory,
)
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat

# Seattle (lon=-122.33, lat=47.61) falls in the N47/W123 tile.
TILE_LAT = 47
TILE_LON = -123
MIN_ELEVATION = 500.0
MAX_ELEVATION = 1400.0


def _make_glo30_geotiff(path: pathlib.Path) -> UPath:
    """Create a small GLO-30-like GeoTIFF covering one 1x1 degree cell.

    The DEM has a simple gradient (elevation increases northward) so that derived
    terrain values would be nonzero.
    """
    west, south, east, north = TILE_LON, TILE_LAT, TILE_LON + 1, TILE_LAT + 1
    width, height = 10, 10
    x_res, y_res = 0.1, -0.1
    projection = Projection(CRS.from_epsg(4326), x_res, y_res)
    bounds = (
        int(west / x_res),
        int(north / y_res),
        int(east / x_res),
        int(south / y_res),
    )

    # Row 0 is north and highest.
    data = np.zeros((1, height, width), dtype=np.float32)
    for row in range(height):
        data[0, row, :] = (height - 1 - row) * 100.0
    data[0] += MIN_ELEVATION

    raster_dir = UPath(path / "glo30_raster")
    tile_name = _tile_name(TILE_LAT, TILE_LON)
    GeotiffRasterFormat().encode_raster(
        raster_dir,
        projection,
        bounds,
        RasterArray(chw_array=data),
        fname=f"{tile_name}.tif",
    )
    return raster_dir / f"{tile_name}.tif"


def _mock_boto3(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch boto3 so the tileList.txt download returns just our test tile."""
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {
        "Body": io.BytesIO(f"{_tile_name(TILE_LAT, TILE_LON)}\n".encode())
    }
    monkeypatch.setattr(boto3, "client", lambda *a, **kw: mock_s3)


def _redirect_rasterio(monkeypatch: pytest.MonkeyPatch, local_tif: UPath) -> None:
    """Redirect reads of the GLO-30 bucket to a local GeoTIFF."""
    original_open = rasterio.open

    def mock_open(url: Any, *args: Any, **kwargs: Any) -> Any:
        if GLO30_BUCKET in str(url):
            return original_open(str(local_tif), *args, **kwargs)
        return original_open(url, *args, **kwargs)

    monkeypatch.setattr(rasterio, "open", mock_open)


@pytest.fixture
def data_source(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> CopernicusGLO30:
    _mock_boto3(monkeypatch)
    _redirect_rasterio(monkeypatch, _make_glo30_geotiff(tmp_path))
    return CopernicusGLO30(metadata_cache_dir=str(tmp_path / "cache"))


def _get_item(data_source: CopernicusGLO30, geometry: STGeometry) -> Item:
    query_config = QueryConfig(space_mode=SpaceMode.MOSAIC, max_matches=1)
    item_groups = data_source.get_items([geometry], query_config)[0]
    assert len(item_groups) == 1
    items = item_groups[0].items
    assert len(items) == 1
    return items[0]


def test_direct_materialize(
    data_source: CopernicusGLO30, seattle2020: STGeometry
) -> None:
    """read_raster should return elevation without any ingest step."""
    item = _get_item(data_source, seattle2020)
    assert item.name == _tile_name(TILE_LAT, TILE_LON)

    bounds = (
        int(seattle2020.shp.bounds[0]),
        int(seattle2020.shp.bounds[1]),
        int(seattle2020.shp.bounds[2]),
        int(seattle2020.shp.bounds[3]),
    )
    array = data_source.read_raster(
        layer_name="layer",
        item=item,
        bands=["elevation"],
        projection=seattle2020.projection,
        bounds=bounds,
    )
    chw = array.get_chw_array()
    assert chw.shape[0] == 1
    assert (chw >= MIN_ELEVATION).all()
    assert (chw <= MAX_ELEVATION).all()


def test_materialize_window(
    tmp_path: pathlib.Path, data_source: CopernicusGLO30, seattle2020: STGeometry
) -> None:
    """The layer should materialize end to end with ingest disabled."""
    query_config = QueryConfig(space_mode=SpaceMode.MOSAIC, max_matches=1)
    item_groups = data_source.get_items([seattle2020], query_config)[0]

    layer_config = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[BandSetConfig(dtype=DType.FLOAT32, bands=["elevation"])],
    )
    bounds = (
        int(seattle2020.shp.bounds[0]),
        int(seattle2020.shp.bounds[1]),
        int(seattle2020.shp.bounds[2]),
        int(seattle2020.shp.bounds[3]),
    )
    window = Window(
        storage=FileWindowStorage(UPath(tmp_path / "rslearn_dataset")),
        group="default",
        name="default",
        projection=seattle2020.projection,
        bounds=bounds,
        time_range=seattle2020.time_range,
        data_factory=PerItemGroupStorageFactory(),
    )
    window.save()

    data_source.materialize(
        window,
        [group.items for group in item_groups],
        "layer",
        layer_config,
    )
    assert window.is_layer_completed("layer")
