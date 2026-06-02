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
from rslearn.data_sources.gcp_public_data import (
    Sentinel2,
    Sentinel2Item,
    Sentinel2ProductType,
)
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat

SEATTLE_WGS84_BOUNDS = (-122.34, 47.60, -122.32, 47.62)
DEGREES_PER_PIXEL = 0.001

MOCK_L1C_ITEM_NAME = "S2A_MSIL1C_20200715T000000_N0209_R001_T10TEM_20200715T000000"
MOCK_L1C_BLOB_PREFIX = "tiles/10/T/EM/S2A_MSIL1C_20200715T000000_N0209_R001_T10TEM_20200715T000000.SAFE/GRANULE/fake/IMG_DATA/T10TEM_20200715T000000_"
MOCK_L2A_ITEM_NAME = "S2A_MSIL2A_20200715T000000_N0209_R001_T10TEM_20200715T000000"
MOCK_L2A_BLOB_PREFIX = "L2/tiles/10/T/EM/S2A_MSIL2A_20200715T000000_N0209_R001_T10TEM_20200715T000000.SAFE/GRANULE/fake/IMG_DATA/"


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
    fmt.encode_raster(raster_dir, projection, bounds, RasterArray(chw_array=data))
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
        name=MOCK_L1C_ITEM_NAME,
        geometry=geometry,
        blob_prefix=MOCK_L1C_BLOB_PREFIX,
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
    assert tile_store.is_raster_ready(layer_name, mock_item, ["B04"])


def _make_l2a_item(seattle2020: STGeometry) -> Sentinel2Item:
    """Create a mock L2A Sentinel2Item corresponding to the seattle2020 fixture."""
    wgs84_geom = seattle2020.to_projection(WGS84_PROJECTION)
    geometry = STGeometry(
        WGS84_PROJECTION,
        wgs84_geom.shp,
        seattle2020.time_range,
    )
    return Sentinel2Item(
        name=MOCK_L2A_ITEM_NAME,
        geometry=geometry,
        blob_prefix=MOCK_L2A_BLOB_PREFIX,
        cloud_cover=5.0,
    )


def test_ingest_l2a(
    tmp_path: pathlib.Path,
    seattle2020: STGeometry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test GCP Public Data Sentinel-2 L2A ingest with mocked GCS bucket."""
    tif_path = _make_test_geotiff(tmp_path)
    mock_item = _make_l2a_item(seattle2020)

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
        product_type=Sentinel2ProductType.L2A,
        use_rtree_index=False,
        use_bigquery=False,
        bands=["B04"],
    )

    # The L2A B04 band file is stored under the R10m resolution folder with a per-scene
    # stem, so the blob path passed to bucket.blob should reflect that layout.
    expected_blob_path = (
        MOCK_L2A_BLOB_PREFIX + "R10m/L2A_T10TEM_20200715T000000_B04_10m.jp2"
    )

    tile_store_dir = UPath(tmp_path / "tiles")
    tile_store = DefaultTileStore(str(tile_store_dir))
    tile_store.set_dataset_path(tile_store_dir)
    layer_name = "layer"

    data_source.ingest(
        TileStoreWithLayer(tile_store, layer_name),
        [mock_item],
        [[seattle2020]],
    )
    assert tile_store.is_raster_ready(layer_name, mock_item, ["B04"])
    mock_bucket.blob.assert_called_once_with(expected_blob_path)


def test_l2a_name_translation(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L2A index lookups translate an L1C scene name to the matching L2A scene.

    The L2A scene shares mission, sensing time, orbit, and tile with the L1C scene but
    may have a different processing baseline and discriminator timestamp, so the name
    is resolved by listing the bucket and matching on those shared fields.
    """
    l1c_name = "S2A_MSIL1C_20200105T185751_N0208_R113_T10TEM_20200105T204120"
    l2a_name = "S2A_MSIL2A_20200105T185751_N0213_R113_T10TEM_20200105T211205"

    mock_bucket = MagicMock()

    captured = {}

    def mock_list_blobs(prefix: str, delimiter: str | None = None) -> MagicMock:
        captured["prefix"] = prefix
        result = MagicMock()
        result.prefixes = [
            # A non-matching orbit that should be filtered out.
            "L2/tiles/10/T/EM/S2A_MSIL2A_20200105T185751_N0213_R999_T10TEM_x.SAFE/",
            f"L2/tiles/10/T/EM/{l2a_name}.SAFE/",
        ]
        return result

    mock_bucket.list_blobs.side_effect = mock_list_blobs

    mock_storage_client = MagicMock()
    mock_storage_client.bucket.return_value = mock_bucket

    monkeypatch.setattr(
        storage.Client, "create_anonymous_client", lambda: mock_storage_client
    )
    data_source = Sentinel2(
        index_cache_dir=str(tmp_path / "cache"),
        product_type=Sentinel2ProductType.L2A,
        use_rtree_index=False,
        use_bigquery=False,
    )

    resolved = data_source._l1c_name_to_l2a_name(l1c_name)
    assert resolved == l2a_name
    assert captured["prefix"] == "L2/tiles/10/T/EM/S2A_MSIL2A_20200105T185751_"
