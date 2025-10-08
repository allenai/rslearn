import pathlib

import torch

from rslearn.models.prithvi import PrithviV2, PrithviV2Models


def test_prithvi(tmp_path: pathlib.Path) -> None:
    """Verify that the forward pass for Prithvi works."""
    for model_size in [PrithviV2Models.VIT_300, PrithviV2Models.VIT_600]:
        # make a unique cache dir per model
        model_tmp_path = tmp_path / model_size.value
        model_tmp_path.mkdir()
        input_hw = 32
        prithvi = PrithviV2(cache_dir=model_tmp_path, size=model_size)

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
    """Verify that the forward pass for Prithvi works."""

    # the implementation is shared with the single timestep model,
    # and the VIT_600 crashes the github runner. Running
    # for the VIT_600 passes locally.
    for model_size in [PrithviV2Models.VIT_300]:
        # make a unique cache dir per model
        model_tmp_path = tmp_path / model_size.value
        model_tmp_path.mkdir()
        input_hw = 32
        num_timesteps = 10
        prithvi = PrithviV2(cache_dir=model_tmp_path, size=model_size)

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
