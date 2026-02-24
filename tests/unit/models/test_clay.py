"""Test the Clay model."""

import pathlib
from typing import Any

import huggingface_hub.constants
import torch

from rslearn.models.clay.clay import Clay, ClayNormalize, ClaySize
from rslearn.train.model_context import ModelContext, RasterImage


def test_clay(tmp_path: pathlib.Path, monkeypatch: Any) -> None:
    # Redirect HF Hub cache to tmpdir so model/metadata are downloaded there.
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))

    # Build Clay model.
    # Use do_resizing=False to avoid memory issue when running this test on small
    # machines.
    clay = Clay(model_size=ClaySize.LARGE, modality="sentinel-2-l2a", do_resizing=False)

    # One input sample, Sentinel-2 L2A modality, 10 bands x 32 x 32
    inputs = [
        {
            "sentinel-2-l2a": RasterImage(
                torch.zeros((10, 1, 32, 32), dtype=torch.float32)
            ),
        }
    ]

    # Apply Clay normalization before forward
    normalize = ClayNormalize()
    input_dict, _ = normalize.forward(inputs[0], {})
    normalized_inputs = [input_dict]

    # Forward pass
    feature_list = clay.forward(ModelContext(inputs=normalized_inputs, metadatas=[]))

    # Should yield one feature map
    assert len(feature_list.feature_maps) == 1
    features = feature_list.feature_maps[0]

    # Check feature shape: (B, D, H', W') with B=1, D=1024, H'=W'=4 (32/8)
    assert features.shape == (1, 1024, 4, 4)

    # Backbone channels should match patch size and depth
    assert clay.get_backbone_channels() == [(8, 1024)]
