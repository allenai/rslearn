import pathlib

import torch

from rslearn.models.prithvi import PRITHVI_MEAN, PrithviV2


def test_prithvi(tmp_path: pathlib.Path) -> None:
    """Verify that the forward pass for Galileo works."""
    input_hw = 32
    prithvi = PrithviV2(pretrained_path=tmp_path)

    inputs = [
        {
            "sentinel2": torch.zeros(
                (len(PRITHVI_MEAN), input_hw, input_hw), dtype=torch.float32
            ),
        }
    ]
    feature_list = prithvi(inputs)
    assert len(feature_list) == len(prithvi.model.encoder.blocks)
    for features in feature_list:
        # features should be BxCxHxW.
        assert features.shape[0] == 1 and len(features.shape) == 4
        feat_hw = input_hw // prithvi.patch_size
        assert features.shape[2] == feat_hw and features.shape[3] == feat_hw
