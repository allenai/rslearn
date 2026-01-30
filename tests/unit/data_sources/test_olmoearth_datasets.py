"""Unit tests for OlmoEarthDatasets (mocked HTTP server)."""

import pathlib
from datetime import UTC, datetime

import numpy as np
import rasterio
import shapely
from pytest_httpserver import HTTPServer
from rasterio.crs import CRS
from upath import UPath

from rslearn.data_sources.olmoearth_datasets import (
    OlmoEarthAsset,
    OlmoEarthDataProvider,
    OlmoEarthDatasets,
    OlmoEarthItem,
)
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils.geometry import Projection, STGeometry


def _make_minimal_geotiff_bytes() -> bytes:
    """Return bytes of a minimal 1-band GeoTIFF that rasterio can read."""
    arr = np.zeros((1, 64, 64), dtype=np.uint16)
    arr[0, 10:20, 10:20] = 500  # non-zero patch
    profile = {
        "driver": "GTiff",
        "height": 64,
        "width": 64,
        "count": 1,
        "dtype": arr.dtype,
        "crs": CRS.from_epsg(32610),
        "transform": rasterio.Affine(10, 0, 0, 0, -10, 0),
    }
    with rasterio.MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(arr)
        return memfile.read()


def _make_mock_item(
    bad_url: str,
    good_url: str,
) -> OlmoEarthItem:
    """Item with two providers: one bad URL (fails), one good URL (succeeds)."""
    ts = datetime(2025, 1, 1, tzinfo=UTC)
    time_range = (ts, ts)
    proj = Projection(CRS.from_epsg(32610), 10, -10)
    box = shapely.box(0, 0, 64, 64)
    geom = STGeometry(proj, box, time_range)

    bad_asset = OlmoEarthAsset(name="B04", url=bad_url, bands=["B04"])
    good_asset = OlmoEarthAsset(name="B04", url=good_url, bands=["B04"])

    bad_provider = OlmoEarthDataProvider(
        provider_name="bad_provider",
        provider_id="bad-1",
        collection="sentinel-2-l2a",
        properties={},
        assets=[bad_asset],
    )
    good_provider = OlmoEarthDataProvider(
        provider_name="good_provider",
        provider_id="good-1",
        collection="sentinel-2-l2a",
        properties={},
        assets=[good_asset],
    )

    return OlmoEarthItem(
        name="S2A_FAKE_20250101_FAKE",
        geometry=geom,
        properties={},
        data_providers=[bad_provider, good_provider],
    )


def test_ingest_falls_back_when_first_provider_fails(
    tmp_path: pathlib.Path,
    httpserver: HTTPServer,
) -> None:
    """Verify that ingestion succeeds with one provider when the other provider returns 500.

    We mock an HTTP server that returns 500 with one path returns 500 while the other
    serves a minimal GeoTIFF without errors. The item has two providers pointing at
    these URLs. We run ingest multiple times and verify that on one attempt the bad
    provider is tried first, fails, but then we succeed with the good provider.
    """
    bad_path = "/bad/fail.tif"
    good_path = "/good/b04.tif"
    good_tif_bytes = _make_minimal_geotiff_bytes()

    httpserver.expect_request(bad_path, method="GET").respond_with_data(b"", status=500)
    httpserver.expect_request(good_path, method="GET").respond_with_data(good_tif_bytes)

    bad_url = httpserver.url_for(bad_path)
    good_url = httpserver.url_for(good_path)
    mock_item = _make_mock_item(bad_url=bad_url, good_url=good_url)

    data_source = OlmoEarthDatasets(
        collection="sentinel-2-l2a",
        asset_bands={"B04": ["B04"]},
    )
    layer_name = "layer"
    bad_matcher = httpserver.create_matcher(bad_path)

    # Try a few times until we get a run where we try bad provider first, fail, and
    # then succeed with the good provider.
    found_desired_run = False
    for i in range(10):
        store_dir = UPath(tmp_path / str(i))
        store_dir.mkdir(parents=True, exist_ok=True)
        tile_store = DefaultTileStore(str(store_dir))
        tile_store.set_dataset_path(store_dir)
        data_source.ingest(
            TileStoreWithLayer(tile_store, layer_name),
            [mock_item],
            [[mock_item.geometry]],
        )
        assert tile_store.is_raster_ready(layer_name, mock_item.name, ["B04"]), (
            f"Run {i}: raster not ready"
        )
        bad_count = httpserver.get_matching_requests_count(bad_matcher)
        if bad_count > 0:
            # This run tried the bad provider first, then fell back to good.
            found_desired_run = True
            break

    assert found_desired_run, "In 10 runs, bad provider was never tried first."
