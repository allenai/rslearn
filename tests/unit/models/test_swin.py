import pathlib

import torch

from rslearn.models.swin import Swin
from rslearn.train.model_context import ModelContext
from rslearn.utils.raster_format import RasterImage


def test_swin(tmp_path: pathlib.Path) -> None:
    """Verify that the forward pass for Prithvi works."""
    input_bands = 3
    input_hw = 32
    model = Swin(input_channels=input_bands, output_layers=[1, 3, 5, 7])

    inputs = [
        {
            "image": RasterImage(
                torch.zeros((input_bands, 1, input_hw, input_hw), dtype=torch.float32)
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
