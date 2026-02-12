"""Mocked integration tests for the GCP Public Data Sentinel-2 data source."""

import pathlib
import shutil
from unittest.mock import MagicMock

import numpy as np
import pytest
from google.cloud import storage
from rasterio.crs import CRS
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.gcp_public_data import Sentinel2, Sentinel2Item
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.raster_format import GeotiffRasterFormat

SEATTLE_WGS84_BOUNDS = (-122.34, 47.60, -122.32, 47.62)
MOCK_ITEM_NAME = "S2A_MSIL1C_20200715T000000_N0209_R001_T10TEM_20200715T000000"
MOCK_BLOB_PREFIX = "tiles/10/T/EM/S2A_MSIL1C_20200715T000000_N0209_R001_T10TEM_20200715T000000.SAFE/GRANULE/fake/IMG_DATA/T10TEM_20200715T000000_"
DEGREES_PER_PIXEL = 0.001


def _make_test_geotiff(path: pathlib.Path) -> pathlib.Path:
    """Create a small test GeoTIFF (the real files are JP2 but it's okay)."""
    # Use UTM projection like real Sentinel-2 band files.
    projection = Projection(CRS.from_epsg(32610), 10, -10)
    # Small area in UTM pixel coords.
    bounds = (5500, -52740, 5516, -52724)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    data = np.ones((1, height, width), dtype=np.uint16) * 1000
    raster_dir = UPath(path / "raster")
    fmt = GeotiffRasterFormat()
    fmt.encode_raster(raster_dir, projection, bounds, data)
    return raster_dir / fmt.fname


def _make_item(seattle2020: STGeometry) -> Sentinel2Item:
    """Create a mock Sentinel2Item corresponding to the seattle2020 fixture."""
    wgs84_geom = seattle2020.to_projection(WGS84_PROJECTION)
    geometry = STGeometry(
        WGS84_PROJECTION,
        wgs84_geom.shp,
        seattle2020.time_range,
    )
    return Sentinel2Item(
        name=MOCK_ITEM_NAME,
        geometry=geometry,
        blob_prefix=MOCK_BLOB_PREFIX,
        cloud_cover=5.0,
    )


def test_ingest(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test GCP Public Data Sentinel-2 ingest with mocked GCS bucket."""
    tif_path = _make_test_geotiff(tmp_path)
    mock_item = _make_item(seattle2020)

    # Mock the GCS storage client and bucket to just return the test GeoTIFF on
    # download_to_filename call. We don't mock any of the listing blobs stuff.
    mock_bucket = MagicMock()

    def mock_download(local_fname: str) -> None:
        shutil.copy(str(tif_path), local_fname)

    mock_blob = MagicMock()
    mock_blob.download_to_filename.side_effect = mock_download
    mock_bucket.blob.return_value = mock_blob

    mock_storage_client = MagicMock()
    mock_storage_client.bucket.return_value = mock_bucket

    monkeypatch.setattr(
        storage.Client, "create_anonymous_client", lambda: mock_storage_client
    )
    data_source = Sentinel2(
        index_cache_dir=str(tmp_path / "cache"),
        use_rtree_index=False,
        use_bigquery=False,
        bands=["B04"],
    )

    # Directly ingest with the mock item. We skip get_items here since we don't want to
    # deal with mocking the listing blobs operations.
    tile_store_dir = UPath(tmp_path / "tiles")
    tile_store = DefaultTileStore(str(tile_store_dir))
    tile_store.set_dataset_path(tile_store_dir)
    layer_name = "layer"

    data_source.ingest(
        TileStoreWithLayer(tile_store, layer_name),
        [mock_item],
        [[seattle2020]],
    )
    assert tile_store.is_raster_ready(layer_name, mock_item.name, ["B04"])
