import torch

from rslearn.models.clip import CLIP
from rslearn.train.model_context import ModelContext, RasterImage

# Test only with a small model.
MODEL_NAME = "openai/clip-vit-base-patch32"


def test_clip() -> None:
    # Make sure the forward pass works.
    clip = CLIP(model_name=MODEL_NAME)
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
