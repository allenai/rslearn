import pytest
import torch

from rslearn.models.clip import CLIP
from rslearn.train.model_context import ModelContext
from rslearn.utils.raster_format import RasterImage


@pytest.mark.parametrize(
    "model_name", ["openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14-336"]
)
def test_clip(model_name: str) -> None:
    # Make sure the forward pass works.
    clip = CLIP(model_name=model_name)
    inputs = [
        {
            "image": RasterImage(torch.zeros((3, 1, 32, 32), dtype=torch.float32)),
        }
    ]
    feature_list = clip(ModelContext(inputs=inputs, metadatas=[])).feature_maps
    # Should yield one feature map since there's only one output scale.
    assert len(feature_list) == 1
    features = feature_list[0]
    # features should be BxCxHxW.
    assert features.shape[0] == 1 and len(features.shape) == 4
