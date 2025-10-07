import pathlib

import torch

from rslearn.models.prithvi import PrithviV2


def test_prithvi(tmp_path: pathlib.Path) -> None:
    """Verify that the forward pass for Galileo works."""
    input_hw = 32
    prithvi = PrithviV2(cache_dir=tmp_path)

    inputs = [
        {
            "image": torch.zeros(
                (len(prithvi.bands), input_hw, input_hw), dtype=torch.float32
            ),
        }
    ]
    feature_list = prithvi(inputs)
    assert len(feature_list) == len(prithvi.model.encoder.blocks)
    for features in feature_list:
        # features should be BxCxHxW.
        assert features.shape[0] == 1 and len(features.shape) == 4
        feat_hw = prithvi.image_resolution // prithvi.patch_size
        assert features.shape[2] == feat_hw and features.shape[3] == feat_hw


def test_prithvi_mt(tmp_path: pathlib.Path) -> None:
    """Verify that the forward pass for Galileo works."""
    input_hw = 32
    num_timesteps = 10
    prithvi = PrithviV2(cache_dir=tmp_path)

    inputs = [
        {
            "image": torch.zeros(
                (len(prithvi.bands) * num_timesteps, input_hw, input_hw),
                dtype=torch.float32,
            ),
        }
    ]
    feature_list = prithvi(inputs)
    assert len(feature_list) == len(prithvi.model.encoder.blocks)
    for features in feature_list:
        # features should be BxCxHxW.
        assert features.shape[0] == 1 and len(features.shape) == 4
        feat_hw = prithvi.image_resolution // prithvi.patch_size
        assert features.shape[2] == feat_hw and features.shape[3] == feat_hw
