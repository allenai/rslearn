"""Test the Terramind model."""

import pytest
import torch

from rslearn.models.terramind import Terramind, TerramindNormalize, TerramindSize


@pytest.mark.parametrize("model_size", [TerramindSize.BASE, TerramindSize.LARGE])
def test_terramind_forward(model_size: TerramindSize) -> None:
    terramind = Terramind(model_size=model_size, modalities=["RGB"])
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
