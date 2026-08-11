"""Unit tests for rslearn.train.transforms.terrain."""

import pytest
import torch

from rslearn.train.model_context import RasterImage
from rslearn.train.transforms.terrain import ElevationToSlopeAspect

PIXEL_SIZE_M = 30.0


def _elevation_image(dem: torch.Tensor) -> RasterImage:
    """Wrap an HW elevation tensor as a single-channel, single-timestep CTHW image."""
    return RasterImage(dem[None, None, :, :])


def _apply(dem: torch.Tensor, **kwargs: object) -> torch.Tensor:
    transform = ElevationToSlopeAspect(pixel_size_m=PIXEL_SIZE_M, **kwargs)  # type: ignore[arg-type]
    input_dict = {"elevation": _elevation_image(dem)}
    input_dict, _ = transform(input_dict, {})
    return input_dict["elevation"].image


def test_flat_surface() -> None:
    """A constant-elevation surface should have zero slope and flat aspect."""
    dem = torch.full((10, 10), 100.0)
    out = _apply(dem, bands=["slope", "aspect"])
    torch.testing.assert_close(out[0], torch.zeros_like(out[0]), atol=1e-5, rtol=0)
    torch.testing.assert_close(out[1], torch.full_like(out[1], -1.0))


def test_north_facing_slope() -> None:
    """Elevation increasing southward means steepest descent points north."""
    dem = torch.arange(20, dtype=torch.float32)[:, None].repeat(1, 20) * 10.0
    out = _apply(dem, bands=["slope", "aspect"])
    slope, aspect = out[0, 0], out[1, 0]

    assert slope[10, 10] > 0.0
    interior = aspect[5:15, 5:15]
    mean_aspect = interior.mean().item()
    assert mean_aspect < 30.0 or mean_aspect > 330.0


def test_east_facing_slope() -> None:
    """Elevation decreasing eastward means steepest descent points east."""
    dem = torch.arange(19, -1, -1, dtype=torch.float32)[None, :].repeat(20, 1) * 10.0
    out = _apply(dem, bands=["slope", "aspect"])
    slope, aspect = out[0, 0], out[1, 0]

    assert slope[10, 10] > 0.0
    interior = aspect[5:15, 5:15]
    assert 60.0 < interior.mean().item() < 120.0


def test_known_slope_angle() -> None:
    """A 1:1 rise over the pixel size should give a 45 degree slope."""
    dem = torch.arange(20, dtype=torch.float32)[:, None].repeat(1, 20) * PIXEL_SIZE_M
    out = _apply(dem, bands=["slope"])
    assert out[0, 0, 10, 10].item() == pytest.approx(45.0, abs=1e-4)


def test_slope_and_aspect_ranges() -> None:
    """Slope stays in [0, 90) and aspect in [0, 360) or the flat value."""
    torch.manual_seed(42)
    dem = torch.randn(50, 50) * 500 + 1000
    out = _apply(dem, bands=["slope", "aspect"])
    slope, aspect = out[0], out[1]

    assert bool((slope >= 0).all())
    assert bool((slope < 90).all())
    positive = aspect[aspect >= 0]
    assert bool((positive < 360).all())
    negative = aspect[aspect < 0]
    torch.testing.assert_close(negative, torch.full_like(negative, -1.0))


def test_nan_propagation() -> None:
    """NaN elevations should propagate into the derived bands."""
    dem = torch.full((10, 10), 100.0)
    dem[5, 5] = float("nan")
    out = _apply(dem, bands=["slope", "aspect"])
    assert bool(torch.isnan(out[0, 0, 5, 5]))
    assert bool(torch.isnan(out[1, 0, 5, 5]))


def test_band_selection_and_order() -> None:
    """Bands default to all three, and a custom list controls channel order."""
    dem = torch.full((8, 8), 42.0)

    default_out = _apply(dem)
    assert default_out.shape[0] == 3
    torch.testing.assert_close(default_out[0, 0], torch.full((8, 8), 42.0))

    out = _apply(dem, bands=["aspect", "elevation"])
    assert out.shape[0] == 2
    torch.testing.assert_close(out[0, 0], torch.full((8, 8), -1.0))
    torch.testing.assert_close(out[1, 0], torch.full((8, 8), 42.0))


def test_preserves_time_dimension() -> None:
    dem = torch.zeros((3, 8, 8))
    transform = ElevationToSlopeAspect(pixel_size_m=PIXEL_SIZE_M)
    input_dict = {"elevation": RasterImage(dem[None, :, :, :])}
    input_dict, _ = transform(input_dict, {})
    assert input_dict["elevation"].image.shape == (3, 3, 8, 8)


def test_writes_to_output_selector() -> None:
    dem = torch.full((8, 8), 5.0)
    transform = ElevationToSlopeAspect(
        pixel_size_m=PIXEL_SIZE_M, output_selector="terrain"
    )
    input_dict = {"elevation": _elevation_image(dem)}
    input_dict, _ = transform(input_dict, {})
    assert input_dict["terrain"].image.shape[0] == 3
    assert input_dict["elevation"].image.shape[0] == 1


def test_skip_missing() -> None:
    transform = ElevationToSlopeAspect(pixel_size_m=PIXEL_SIZE_M, skip_missing=True)
    input_dict: dict = {}
    input_dict, _ = transform(input_dict, {})
    assert input_dict == {}


def test_rejects_multichannel_input() -> None:
    transform = ElevationToSlopeAspect(pixel_size_m=PIXEL_SIZE_M)
    input_dict = {"elevation": RasterImage(torch.zeros((2, 1, 8, 8)))}
    with pytest.raises(ValueError, match="single-channel"):
        transform(input_dict, {})


def test_rejects_bad_config() -> None:
    with pytest.raises(ValueError, match="pixel_size_m must be positive"):
        ElevationToSlopeAspect(pixel_size_m=0)
    with pytest.raises(ValueError, match="unsupported band"):
        ElevationToSlopeAspect(pixel_size_m=30.0, bands=["nope"])
    with pytest.raises(ValueError, match="at least one band"):
        ElevationToSlopeAspect(pixel_size_m=30.0, bands=[])
