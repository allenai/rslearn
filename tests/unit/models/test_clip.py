import pytest
import torch

from rslearn.models.clip import CLIP


@pytest.mark.parametrize(
    "model_name", ["openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14-336"]
)
def test_molmo(model_name: str) -> None:
    # Make sure the forward pass works.
    clip = CLIP(model_name=model_name)
    inputs = [
        {
            "image": torch.zeros((3, 32, 32), dtype=torch.float32),
        }
    ]
    feature_list = clip(inputs)
    # Should yield one feature map since there's only one output scale.
    assert len(feature_list) == 1
    features = feature_list[0]
    # features should be BxCxHxW.
    assert features.shape[0] == 1 and len(features.shape) == 4
