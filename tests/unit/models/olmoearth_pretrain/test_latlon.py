"""Tests for lat/lon encoding support in OlmoEarth wrapper."""

from datetime import datetime

import torch
from rasterio.crs import CRS

from rslearn.models.olmoearth_pretrain.model import OlmoEarth
from rslearn.train.model_context import ModelContext, RasterImage, SampleMetadata
from rslearn.utils.geometry import Projection


def _make_metadata(
    col_start: int = 0,
    row_start: int = 0,
    col_end: int = 64,
    row_end: int = 64,
    crs: CRS | None = None,
    x_resolution: float = 1.0,
    y_resolution: float = 1.0,
) -> SampleMetadata:
    """Create a SampleMetadata for testing."""
    if crs is None:
        crs = CRS.from_epsg(4326)
    return SampleMetadata(
        window_group="default",
        window_name="test",
        window_bounds=(col_start, row_start, col_end, row_end),
        crop_bounds=(col_start, row_start, col_end, row_end),
        crop_idx=0,
        num_crops_in_window=1,
        time_range=None,
        projection=Projection(crs, x_resolution, y_resolution),
        dataset_source=None,
    )


def test_compute_latlon_from_metadata_wgs84() -> None:
    """Test lat/lon computation with WGS84 CRS (identity transform)."""
    # WGS84 with 1 degree/pixel: pixel center at (32, 32) -> CRS (32, 32) -> lon=32, lat=32
    meta = _make_metadata(
        col_start=0,
        row_start=0,
        col_end=64,
        row_end=64,
        crs=CRS.from_epsg(4326),
        x_resolution=1.0,
        y_resolution=1.0,
    )
    result = OlmoEarth._compute_latlon_from_metadata([meta], torch.device("cpu"))
    assert result.shape == (1, 2)
    # Pixel center: (32, 32), CRS coords: (32*1, 32*1) = (32, 32)
    # WGS84 to WGS84: lon=32, lat=32 -> output is (lat, lon) = (32, 32)
    assert abs(result[0, 0].item() - 32.0) < 0.1  # lat
    assert abs(result[0, 1].item() - 32.0) < 0.1  # lon


def test_compute_latlon_from_metadata_batch() -> None:
    """Test with multiple samples producing different lat/lon values."""
    meta1 = _make_metadata(col_start=0, row_start=0, col_end=20, row_end=20)
    meta2 = _make_metadata(col_start=100, row_start=50, col_end=120, row_end=70)
    result = OlmoEarth._compute_latlon_from_metadata(
        [meta1, meta2], torch.device("cpu")
    )
    assert result.shape == (2, 2)
    # Different crops should produce different lat/lon
    assert not torch.allclose(result[0], result[1])


def test_compute_latlon_from_metadata_utm() -> None:
    """Test lat/lon computation with UTM CRS (non-trivial transform)."""
    # UTM zone 32N (EPSG:32632), centered roughly on 9E 0N (equator)
    # Pixel at center (500000/10, 0/10) with 10m resolution
    utm_crs = CRS.from_epsg(32632)
    meta = _make_metadata(
        col_start=49990,
        row_start=0,
        col_end=50010,
        row_end=20,
        crs=utm_crs,
        x_resolution=10.0,
        y_resolution=10.0,
    )
    result = OlmoEarth._compute_latlon_from_metadata([meta], torch.device("cpu"))
    assert result.shape == (1, 2)
    # Center pixel: (50000, 10), CRS: (500000, 100) -> roughly (9E, ~0N)
    lat = result[0, 0].item()
    lon = result[0, 1].item()
    assert -1 < lat < 1, f"Expected lat near equator, got {lat}"
    assert 8 < lon < 10, f"Expected lon near 9E, got {lon}"


def test_forward_with_use_latlon() -> None:
    """Test that forward pass works with use_latlon=True."""
    model = OlmoEarth(
        checkpoint_path="tests/unit/models/olmoearth_pretrain/",
        random_initialization=True,
        patch_size=4,
        embedding_size=128,
        use_latlon=True,
    )

    T = 2
    H = 4
    W = 4
    inputs = [
        {
            "sentinel2_l2a": RasterImage(
                image=torch.zeros(
                    (12, T, H, W), dtype=torch.float32, device=torch.device("cpu")
                ),
                timestamps=[
                    (datetime(2025, x, 1), datetime(2025, x, 1))
                    for x in range(1, T + 1)
                ],
            )
        }
    ]
    meta = _make_metadata(
        col_start=0,
        row_start=0,
        col_end=64,
        row_end=64,
        crs=CRS.from_epsg(4326),
        x_resolution=1.0,
        y_resolution=1.0,
    )
    feature_map = model(ModelContext(inputs=inputs, metadatas=[meta]))

    assert len(feature_map.feature_maps) == 1
    features = feature_map.feature_maps[0]
    assert features.shape == (1, 128, 1, 1)


def test_forward_without_use_latlon_unchanged() -> None:
    """Test that use_latlon=False (default) doesn't change behavior."""
    model = OlmoEarth(
        checkpoint_path="tests/unit/models/olmoearth_pretrain/",
        random_initialization=True,
        patch_size=4,
        embedding_size=128,
        use_latlon=False,
    )

    T = 2
    H = 4
    W = 4
    inputs = [
        {
            "sentinel2_l2a": RasterImage(
                image=torch.zeros(
                    (12, T, H, W), dtype=torch.float32, device=torch.device("cpu")
                ),
                timestamps=[
                    (datetime(2025, x, 1), datetime(2025, x, 1))
                    for x in range(1, T + 1)
                ],
            )
        }
    ]
    # Should work with empty metadatas (existing behavior)
    feature_map = model(ModelContext(inputs=inputs, metadatas=[]))

    assert len(feature_map.feature_maps) == 1
    features = feature_map.feature_maps[0]
    assert features.shape == (1, 128, 1, 1)
