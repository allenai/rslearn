"""Test Tessera model wrapper."""

from datetime import datetime
from pathlib import Path

import pytest
import torch

from rslearn.models.singletask import SingleTaskModel
from rslearn.models.tessera import Tessera
from rslearn.models.tessera.tessera import build_resample_indices
from rslearn.train.model_context import ModelContext, RasterImage
from rslearn.train.tasks.embedding import EmbeddingHead

TINY_CONFIG = {
    "fusion_method": "concat",
    "latent_dim": 2,
    "representation_dim": 5,
    "save_embedding_dim": 3,
    "s2_num_heads": 2,
    "s2_num_layers": 1,
    "s2_dim_feedforward": 16,
    "s1_num_heads": 2,
    "s1_num_layers": 1,
    "s1_dim_feedforward": 16,
    "split_s1_modalities": False,
    "num_obs_checkpoints": [2, 4],
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


def test_build_resample_indices() -> None:
    assert build_resample_indices(3, 3) == [0, 1, 2]
    assert build_resample_indices(3, 5) == [0, 1, 2, 1, 1]
    assert build_resample_indices(6, 3) == [1, 3, 5]
    assert build_resample_indices(0, 3) == []


def test_time_ranges_to_doy() -> None:
    doys = Tessera._time_ranges_to_doy(
        [(datetime(2024, 2, 29), datetime(2024, 2, 29))],
        torch.device("cpu"),
    )
    assert doys.tolist() == [60.0]


def test_prepare_s2_sequence_preserves_normalized_values() -> None:
    tessera = Tessera(
        random_initialization=True,
        model_config=TINY_CONFIG,
        save_embedding_dim=3,
        apply_amp=False,
    )
    image = torch.zeros((10, 2, 1, 1), dtype=torch.float32)
    image[:, 0, 0, 0] = torch.linspace(-2.0, 2.0, steps=10)
    doys = torch.tensor([42.0, 100.0], dtype=torch.float32)

    sequence = tessera._prepare_s2_sequence(image, doys, row=0, col=0, target=1)

    assert sequence.shape == (1, 11)
    assert torch.allclose(sequence[0, :10], image[:, 0, 0, 0])
    assert sequence[0, 10].item() == 42.0


def test_prepare_s1_sequence_preserves_normalized_values() -> None:
    tessera = Tessera(
        random_initialization=True,
        model_config=TINY_CONFIG,
        save_embedding_dim=3,
        apply_amp=False,
    )
    ascending = torch.zeros((2, 2, 1, 1), dtype=torch.float32)
    ascending[:, 0, 0, 0] = torch.tensor([0.5, -1.25], dtype=torch.float32)
    descending = torch.zeros((2, 1, 1, 1), dtype=torch.float32)
    asc_doys = torch.tensor([12.0, 20.0], dtype=torch.float32)
    desc_doys = torch.tensor([30.0], dtype=torch.float32)

    sequence = tessera._prepare_s1_sequence(
        ascending,
        asc_doys,
        descending,
        desc_doys,
        row=0,
        col=0,
        target=1,
    )

    assert sequence.shape == (1, 3)
    assert torch.allclose(sequence[0, :2], ascending[:, 0, 0, 0])
    assert sequence[0, 2].item() == 12.0


def test_tessera_forward_shape() -> None:
    tessera = Tessera(
        random_initialization=True,
        model_config=TINY_CONFIG,
        save_embedding_dim=3,
        pixel_batch_size=2,
        apply_amp=False,
    )
    tessera.eval()
    features = tessera(ModelContext(inputs=_inputs(), metadatas=[]))
    assert len(features.feature_maps) == 1
    assert features.feature_maps[0].shape == (1, 3, 2, 2)
    assert tessera.get_backbone_channels() == [(1, 3)]


def test_tessera_requires_timestamps() -> None:
    tessera = Tessera(
        random_initialization=True,
        model_config=TINY_CONFIG,
        save_embedding_dim=3,
        apply_amp=False,
    )
    inputs = _inputs()
    inputs[0]["s2"].timestamps = None
    with pytest.raises(ValueError, match="requires timestamps"):
        tessera(ModelContext(inputs=inputs, metadatas=[]))


def test_tessera_checkpoint_loads_prefixed_state(tmp_path: Path) -> None:
    source = Tessera(
        random_initialization=True,
        model_config=TINY_CONFIG,
        save_embedding_dim=3,
        apply_amp=False,
    )
    state = {
        f"_orig_mod.{key}": value.detach().clone()
        for key, value in source.model.state_dict().items()
    }
    state["projector.weight"] = torch.ones(1)
    checkpoint_path = tmp_path / "tessera.pt"
    torch.save({"model_state": state}, checkpoint_path)

    loaded = Tessera(
        checkpoint_path=checkpoint_path,
        model_config=TINY_CONFIG,
        save_embedding_dim=3,
        apply_amp=False,
    )
    key = "s2_backbone.embedding.0.weight"
    assert torch.equal(source.model.state_dict()[key], loaded.model.state_dict()[key])


def test_tessera_works_with_embedding_head() -> None:
    model = SingleTaskModel(
        encoder=[
            Tessera(
                random_initialization=True,
                model_config=TINY_CONFIG,
                save_embedding_dim=3,
                apply_amp=False,
            )
        ],
        decoder=[EmbeddingHead()],
    )
    model.eval()
    output = model(ModelContext(inputs=_inputs(), metadatas=[]))
    assert tuple(output.outputs.shape) == (1, 3, 2, 2)
