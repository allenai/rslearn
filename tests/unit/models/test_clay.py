"""Test the Clay model."""

import pathlib
from typing import Any

import huggingface_hub.constants
import torch

from rslearn.models.clay import Clay, ClayNormalize, ClaySize


def test_clay(tmp_path: pathlib.Path, monkeypatch: Any) -> None:
    # Redirect HF Hub cache to tmpdir so model/metadata are downloaded there.
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))

    # Build Clay model
    clay = Clay(model_size=ClaySize.LARGE, modality="sentinel-2-l2a")

    # One input sample, Sentinel-2 L2A modality, 10 bands × 256 × 256
    inputs = [
        {
            "sentinel-2-l2a": torch.zeros((10, 256, 256), dtype=torch.float32),
        }
    ]

    # Apply Clay normalization before forward
    normalize = ClayNormalize()
    input_dict, _ = normalize.forward(inputs[0], {})
    normalized_inputs = [input_dict]

    # Forward pass
    feature_list = clay.forward(normalized_inputs)

    # Should yield one feature map
    assert len(feature_list) == 1
    features = feature_list[0]

    # Check feature shape: (B, D, H', W') with B=1, D=1024, H'=W'=32 (256/8)
    assert features.shape[0] == 1
    assert features.shape[1] == 1024
    assert features.shape[2] == 32
    assert features.shape[3] == 32

    # Backbone channels should match patch size and depth
    assert clay.get_backbone_channels() == [(8, 1024)]
