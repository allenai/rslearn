"""Test the Terramind model."""

import pytest
import torch

from rslearn.models.terramind import Terramind, TerramindSize


@pytest.mark.parametrize("model_size", [TerramindSize.BASE, TerramindSize.LARGE])
def test_terramind_forward(model_size: TerramindSize) -> None:
    # Initialize the Terramind model with the given size
    terramind = Terramind(model_size=model_size, image_size=256, modalities=["RGB"])

    # Create a dummy input with the expected shape
    inputs = [
        {
            "RGB": torch.zeros((3, 256, 256), dtype=torch.float32),
        }
    ]

    # Perform a forward pass
    feature_list = terramind.forward(inputs)

    # Check that the output is a list with one tensor
    assert len(feature_list) == 1

    # Extract the features
    features = feature_list[0]

    # Check the shape of the features
    assert features.shape[0] == 1 and len(features.shape) == 4
