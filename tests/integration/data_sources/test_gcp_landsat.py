"""Integration tests for the GCP Landsat data source (mocked GCS)."""

import json
import pathlib
import shutil
from unittest.mock import MagicMock

import numpy as np
import pytest
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.gcp_landsat import Landsat, LandsatItem
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils.geometry import STGeometry, flatten_shape
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat
from rslearn.utils.rtree_index import RtreeIndex

PIXEL_VALUE = 5000
MOCK_PRODUCT_ID = "LC08_L1TP_046027_20200715_20200724_02_T1"
MOCK_BLOB_PATH = "LC08/L1/02/046/027/LC08_L1TP_046027_20200715_20200724_02_T1/"


@pytest.fixture
def test_geotiff(tmp_path: pathlib.Path, seattle2020: STGeometry) -> UPath:
    """Create a small single-band uint16 GeoTIFF in the seattle2020 projection."""
    bounds = (
        int(seattle2020.shp.bounds[0]),
        int(seattle2020.shp.bounds[1]),
        int(seattle2020.shp.bounds[2]),
        int(seattle2020.shp.bounds[3]),
    )
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    data = np.full((1, height, width), PIXEL_VALUE, dtype=np.uint16)
    raster_dir = UPath(tmp_path / "raster")
    fmt = GeotiffRasterFormat()
    fmt.encode_raster(
        raster_dir, seattle2020.projection, bounds, RasterArray(chw_array=data)
    )
    return raster_dir / fmt.fname


def _make_item(seattle2020: STGeometry) -> LandsatItem:
    """Create a mock LandsatItem covering the seattle2020 fixture area."""
    wgs84_geom = seattle2020.to_projection(WGS84_PROJECTION)
    geometry = STGeometry(
        WGS84_PROJECTION,
        wgs84_geom.shp,
        seattle2020.time_range,
    )
    return LandsatItem(
        name=MOCK_PRODUCT_ID,
        geometry=geometry,
        blob_path=MOCK_BLOB_PATH,
        cloud_cover=5.0,
        spacecraft_id="LANDSAT_8",
        data_type="L1TP",
    )


def _prepare_cache(cache_dir: pathlib.Path, seattle2020: STGeometry) -> None:
    """Pre-populate the rtree index cache with a single mock item."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    item = _make_item(seattle2020)
    rtree_path = str(cache_dir / "rtree_index")
    rtree = RtreeIndex(rtree_path)
    for shp in flatten_shape(item.geometry.shp):
        rtree.insert(shp.bounds, json.dumps(item.serialize()))
    del rtree
    (cache_dir / "rtree_index.done").touch()


def _make_mock_storage(tif_path: UPath) -> MagicMock:
    """Create a mock storage.Client whose bucket downloads the test GeoTIFF."""
    mock_blob = MagicMock()

    def mock_download(local_fname: str) -> None:
        shutil.copy(str(tif_path), local_fname)

    mock_blob.download_to_filename.side_effect = mock_download

    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob

    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    return mock_client


def test_ingest(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    test_geotiff: UPath,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test ingesting a Landsat band into a tile store with mocked GCS."""
    cache_dir = tmp_path / "cache"
    _prepare_cache(cache_dir, seattle2020)

    mock_client = _make_mock_storage(test_geotiff)
    monkeypatch.setattr(
        "rslearn.data_sources.gcp_landsat.storage.Client",
        lambda: mock_client,
    )

    data_source = Landsat(
        index_cache_dir=str(cache_dir),
        bands=["B4"],
    )
    mock_item = _make_item(seattle2020)

    tile_store_dir = UPath(tmp_path / "tiles")
    tile_store = DefaultTileStore(str(tile_store_dir))
    tile_store.set_dataset_path(tile_store_dir)
    layer_name = "layer"

    data_source.ingest(
        TileStoreWithLayer(tile_store, layer_name),
        [mock_item],
        [[seattle2020]],
    )
    assert tile_store.is_raster_ready(layer_name, mock_item, ["B4"])

    bounds = (
        int(seattle2020.shp.bounds[0]),
        int(seattle2020.shp.bounds[1]),
        int(seattle2020.shp.bounds[2]),
        int(seattle2020.shp.bounds[3]),
    )
    raster_data = tile_store.read_raster(
        layer_name, mock_item, ["B4"], seattle2020.projection, bounds
    )
    assert (raster_data.get_chw_array() == PIXEL_VALUE).all()


def test_direct_materialize(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    test_geotiff: UPath,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test direct materialization (read_raster) with mocked asset URL."""
    cache_dir = tmp_path / "cache"
    _prepare_cache(cache_dir, seattle2020)

    mock_client = _make_mock_storage(test_geotiff)
    monkeypatch.setattr(
        "rslearn.data_sources.gcp_landsat.storage.Client",
        lambda: mock_client,
    )

    data_source = Landsat(
        index_cache_dir=str(cache_dir),
        bands=["B4"],
    )
    mock_item = _make_item(seattle2020)

    monkeypatch.setattr(
        data_source, "get_asset_url", lambda item, asset_key: str(test_geotiff)
    )

    bounds = (
        int(seattle2020.shp.bounds[0]),
        int(seattle2020.shp.bounds[1]),
        int(seattle2020.shp.bounds[2]),
        int(seattle2020.shp.bounds[3]),
    )
    array = data_source.read_raster(
        layer_name="fake",
        item=mock_item,
        bands=["B4"],
        projection=seattle2020.projection,
        bounds=bounds,
    )
    assert (array.get_chw_array() == PIXEL_VALUE).all()
