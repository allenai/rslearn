import torch

from rslearn.models.ssl4eo_s12 import Ssl4eoS12
from rslearn.train.model_context import ModelContext, RasterImage


def test_ssl4eo() -> None:
    """Verify that the forward pass for Prithvi works."""
    input_hw = 512
    model = Ssl4eoS12(backbone_ckpt_path=None)

    inputs = [
        {
            "image": RasterImage(
                torch.zeros((13, 1, input_hw, input_hw), dtype=torch.float32)
            ),
        }
    ]
    feature_list = model(ModelContext(inputs=inputs, metadatas=[])).feature_maps
    assert len(feature_list) == 4
    for idx, features in enumerate(feature_list):
        # features should be BxCxHxW
        assert features.shape[0] == 1 and len(features.shape) == 4
        downsample_factor, channels = model.get_backbone_channels()[idx]
        assert features.shape[1] == channels
        assert features.shape[2] == features.shape[3] == input_hw // downsample_factor
