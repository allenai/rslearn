import pytest
import torch

from rslearn.models.molmo import Molmo


@pytest.mark.parametrize(
    "model_name", ["allenai/MolmoE-1B-0924", "allenai/Molmo-7B-D-0924"]
)
def test_molmo(model_name: str) -> None:
    # Make sure the forward pass works.
    molmo = Molmo(model_name=model_name)
    inputs = [
        {
            "image": torch.zeros((3, 1, 32, 32), dtype=torch.float32),
        }
    ]
    feature_list = molmo(inputs)
    # Should yield one feature map since there's only one output scale.
    assert len(feature_list) == 1
    features = feature_list[0]
    # features should be BxCxHxW.
    assert features.shape[0] == 1 and len(features.shape) == 4
