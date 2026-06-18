"""Test Tessera normalization transform."""

import torch

from rslearn.models.tessera import NORM_STATS, TesseraNormalize
from rslearn.train.model_context import RasterImage


def test_tessera_normalize_s1_standard_db_to_scaled_values() -> None:
    """Sentinel-1 dB inputs are converted to Tessera's upstream scale."""
    image = torch.tensor(
        [
            [[[-10.0, -60.0]]],
            [[[-10.0, -60.0]]],
        ],
        dtype=torch.float32,
    )
    transform = TesseraNormalize(skip_missing=True)
    inputs = {"s1_ascending": RasterImage(image)}

    transform(inputs, {})

    stats = NORM_STATS["mpc"]
    expected_scaled = (2 * -10.0 + 50.0) * 200.0
    expected = (
        torch.tensor([expected_scaled, expected_scaled])
        - torch.tensor(stats["s1a_mean"])
    ) / torch.tensor(stats["s1a_std"])
    assert torch.allclose(inputs["s1_ascending"].image[:, 0, 0, 0], expected)
    # -60 dB clamps to a scaled value of 0, which is still standardized (no
    # nodata handling).
    expected_clamped = (
        torch.tensor([0.0, 0.0]) - torch.tensor(stats["s1a_mean"])
    ) / torch.tensor(stats["s1a_std"])
    assert torch.allclose(inputs["s1_ascending"].image[:, 0, 0, 1], expected_clamped)


def test_tessera_normalize_s2_standardizes_all_pixels() -> None:
    """Sentinel-2 pixels are standardized, including all-zero pixels."""
    image = torch.zeros((10, 1, 1, 2), dtype=torch.float32)
    image[:, 0, 0, 0] = torch.arange(1000, 1010, dtype=torch.float32)
    transform = TesseraNormalize(skip_missing=True)
    inputs = {"s2": RasterImage(image)}

    transform(inputs, {})

    stats = NORM_STATS["mpc"]
    expected = (
        torch.arange(1000, 1010, dtype=torch.float32) - torch.tensor(stats["s2_mean"])
    ) / torch.tensor(stats["s2_std"])
    assert torch.allclose(inputs["s2"].image[:, 0, 0, 0], expected)
    expected_zero = (torch.zeros(10) - torch.tensor(stats["s2_mean"])) / torch.tensor(
        stats["s2_std"]
    )
    assert torch.allclose(inputs["s2"].image[:, 0, 0, 1], expected_zero)


def test_tessera_normalize_aws_uses_aws_stats() -> None:
    """The data_source argument selects the corresponding stats family."""
    image = torch.full((2, 1, 1, 1), -10.0, dtype=torch.float32)
    transform = TesseraNormalize(data_source="aws", skip_missing=True)
    inputs = {"s1_descending": RasterImage(image)}

    transform(inputs, {})

    stats = NORM_STATS["aws"]
    expected_scaled = (2 * -10.0 + 50.0) * 200.0
    expected = (
        torch.tensor([expected_scaled, expected_scaled])
        - torch.tensor(stats["s1d_mean"])
    ) / torch.tensor(stats["s1d_std"])
    assert torch.allclose(inputs["s1_descending"].image[:, 0, 0, 0], expected)
