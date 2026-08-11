"""Test Tessera model wrapper."""

from datetime import datetime

import torch

from rslearn.models.tessera import Tessera
from rslearn.train.model_context import ModelContext, RasterImage

# Configure a smaller Tessera model to use for testing. This won't be loaded from a
# checkpoint, just randomly initialized.
TINY_CONFIG = {
    "fusion_method": "concat",
    "latent_dim": 2,
    "representation_dim": 5,
    "s2_num_heads": 2,
    "s2_num_layers": 1,
    "s2_dim_feedforward": 16,
    "s1_num_heads": 2,
    "s1_num_layers": 1,
    "s1_dim_feedforward": 16,
    "split_s1_modalities": False,
}


def _timestamps(count: int) -> list[tuple[datetime, datetime]]:
    return [
        (
            datetime(2024, 1, idx + 1, 0, 0, 0),
            datetime(2024, 1, idx + 1, 0, 0, 0),
        )
        for idx in range(count)
    ]


def _inputs() -> list[dict[str, RasterImage]]:
    s2 = torch.ones((10, 3, 2, 2), dtype=torch.float32)
    s1_ascending = torch.ones((2, 2, 2, 2), dtype=torch.float32)
    s1_descending = torch.ones((2, 1, 2, 2), dtype=torch.float32)
    return [
        {
            "s2": RasterImage(s2, timestamps=_timestamps(3)),
            "s1_ascending": RasterImage(s1_ascending, timestamps=_timestamps(2)),
            "s1_descending": RasterImage(s1_descending, timestamps=_timestamps(1)),
        }
    ]


def test_time_ranges_to_doy() -> None:
    doys = Tessera._time_ranges_to_doy(
        [(datetime(2024, 2, 29), datetime(2024, 2, 29))],
        torch.device("cpu"),
    )
    # 29 February 2024 is the 60th day of the year.
    assert doys.tolist() == [60.0]


def test_build_sequence_appends_doy_per_pixel() -> None:
    image = torch.zeros((10, 2, 1, 1), dtype=torch.float32)
    image[:, 0, 0, 0] = torch.linspace(-2.0, 2.0, steps=10)
    image[:, 1, 0, 0] = torch.linspace(1.0, 3.0, steps=10)
    doys = torch.tensor([42.0, 100.0], dtype=torch.float32)

    sequence = Tessera._build_sequence(image, doys, num_pixels=1)

    assert sequence.shape == (1, 2, 11)
    # First part is the image content.
    assert torch.allclose(sequence[0, 0, :10], image[:, 0, 0, 0])
    assert torch.allclose(sequence[0, 1, :10], image[:, 1, 0, 0])
    # Index 10 contains the day-of-year.
    assert sequence[0, 0, 10].item() == 42.0
    assert sequence[0, 1, 10].item() == 100.0


def test_tessera_forward_shape() -> None:
    tessera = Tessera(
        random_initialization=True,
        model_config=TINY_CONFIG,
        pixel_batch_size=2,
        apply_amp=False,
    )
    tessera.eval()
    features = tessera(ModelContext(inputs=_inputs(), metadatas=[]))
    assert len(features.feature_maps) == 1
    assert features.feature_maps[0].shape == (1, 5, 2, 2)
    assert tessera.get_backbone_channels() == [(1, 5)]
