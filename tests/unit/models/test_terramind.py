"""Test the Terramind model."""

import pathlib
from typing import Any

import huggingface_hub.constants
import torch

from rslearn.models.terramind import Terramind, TerramindNormalize, TerramindSize
from rslearn.train.model_context import ModelContext, RasterImage


def test_terramind_without_resizing(tmp_path: pathlib.Path, monkeypatch: Any) -> None:
    # Use monkeypatch to set HF_HUB_CACHE so we can store the weights in a temp dir.
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    terramind = Terramind(
        model_size=TerramindSize.BASE, modalities=["RGB"], do_resizing=False
    )

    inputs = [
        {
            "RGB": RasterImage(torch.zeros((3, 1, 32, 32), dtype=torch.float32)),
        }
    ]

    # Apply normalization before input into model
    normalize = TerramindNormalize()
    input_dict, _ = normalize.forward(inputs[0], {})
    normalized_inputs = [input_dict]

    feature_list = terramind.forward(
        ModelContext(inputs=normalized_inputs, metadatas=[])
    )
    # Should yield one feature map since there's only one output scale.
    assert len(feature_list.feature_maps) == 1
    features = feature_list.feature_maps[0]
    # Features should be BxCxHxW
    assert features.shape[0] == 1
    assert features.shape[2] == 2
    assert features.shape[3] == 2


def test_terramind_with_resizing(tmp_path: pathlib.Path, monkeypatch: Any) -> None:
    # Use monkeypatch to set HF_HUB_CACHE so we can store the weights in a temp dir.
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    terramind = Terramind(
        model_size=TerramindSize.BASE, modalities=["RGB"], do_resizing=True
    )

    inputs = [
        {
            "RGB": RasterImage(torch.zeros((3, 1, 32, 32), dtype=torch.float32)),
        }
    ]

    feature_list = terramind.forward(ModelContext(inputs=inputs, metadatas=[]))
    # Should yield one feature map since there's only one output scale.
    assert len(feature_list.feature_maps) == 1
    features = feature_list.feature_maps[0]
    # Features should be BxCxHxW
    assert features.shape[0] == 1
    assert features.shape[2] == 14
    assert features.shape[3] == 14
