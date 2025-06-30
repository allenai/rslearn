"""Test the Terramind model."""

import pytest
import torch

from rslearn.models.terramind import Terramind, TerramindSize


@pytest.mark.parametrize("model_size", [TerramindSize.BASE, TerramindSize.LARGE])
def test_terramind_forward(model_size: TerramindSize) -> None:
    # Initialize the Terramind model with the given size
    terramind = Terramind(model_size=model_size, image_size=256, modalities=["RGB"])
    inputs = [
        {
            "RGB": torch.zeros((3, 256, 256), dtype=torch.float32),
        }
    ]
    feature_list = terramind.forward(inputs)
    # Should yield one feature map since there's only one output scale.
    assert len(feature_list) == 1
    features = feature_list[0]
    # features should be BxHxWxC.
    assert features.shape[0] == 1 and len(features.shape) == 4
