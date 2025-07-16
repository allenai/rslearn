"""Test the Terramind model."""

import pathlib
from typing import Any

import huggingface_hub.constants
import torch

from rslearn.models.terramind import Terramind, TerramindNormalize, TerramindSize


def test_terramind(tmp_path: pathlib.Path, monkeypatch: Any) -> None:
    # Use monkeypatch to set HF_HUB_CACHE so we can store the weights in a temp dir.
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    terramind = Terramind(model_size=TerramindSize.BASE, modalities=["RGB"])

    inputs = [
        {
            "RGB": torch.zeros((3, 256, 256), dtype=torch.float32),
        }
    ]

    # Apply normalization before input into model
    normalize = TerramindNormalize()
    input_dict, _ = normalize.forward(inputs[0], {})
    normalized_inputs = [input_dict]

    feature_list = terramind.forward(normalized_inputs)
    # Should yield one feature map since there's only one output scale.
    assert len(feature_list) == 1
    features = feature_list[0]
    # Features should be BxCxHxW
    assert features.shape[0] == 1
    assert features.shape[2] == 16
    assert features.shape[3] == 16
