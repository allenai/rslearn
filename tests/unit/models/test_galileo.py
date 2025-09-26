import pathlib
import tempfile

import torch
from pytest import MonkeyPatch

from rslearn.models.galileo import GalileoModel, GalileoSize


def test_galileo(tmp_path: pathlib.Path, monkeypatch: MonkeyPatch) -> None:
    """Verify that the forward pass for Galileo works."""
    input_hw = 32
    patch_size = 4
    # We override the temporary directory so we don't retain the model weights outside
    # of this test.
    monkeypatch.setattr(tempfile, "gettempdir", lambda: tmp_path)
    galileo = GalileoModel(size=GalileoSize.NANO, patch_size=patch_size)

    inputs = [
        {
            "s2": torch.zeros((10, input_hw, input_hw), dtype=torch.float32),
            "s1": torch.zeros((2, input_hw, input_hw), dtype=torch.float32),
            "era5": torch.zeros((2, input_hw, input_hw), dtype=torch.float32),
        }
    ]
    feature_list = galileo(inputs)
    # Should yield one feature map since there's only one output scale.
    assert len(feature_list) == 1
    features = feature_list[0]
    # features should be BxCxHxW.
    assert features.shape[0] == 1 and len(features.shape) == 4
    feat_hw = input_hw // patch_size
    assert features.shape[2] == feat_hw and features.shape[3] == feat_hw


def test_galileo_mt(tmp_path: pathlib.Path, monkeypatch: MonkeyPatch) -> None:
    """Verify that the forward pass for Galileo works."""
    input_hw = 32
    patch_size = 4
    num_timesteps = 10
    # We override the temporary directory so we don't retain the model weights outside
    # of this test.
    monkeypatch.setattr(tempfile, "gettempdir", lambda: tmp_path)
    galileo = GalileoModel(size=GalileoSize.NANO, patch_size=patch_size)

    inputs = [
        {
            "s2": torch.zeros(
                (10 * num_timesteps, input_hw, input_hw), dtype=torch.float32
            ),
            "s1": torch.zeros(
                (2 * num_timesteps, input_hw, input_hw), dtype=torch.float32
            ),
            "era5": torch.zeros(
                (2 * num_timesteps, input_hw, input_hw), dtype=torch.float32
            ),
        }
    ]
    feature_list = galileo(inputs)
    # Should yield one feature map since there's only one output scale.
    assert len(feature_list) == 1
    features = feature_list[0]
    # features should be BxCxHxW.
    assert features.shape[0] == 1 and len(features.shape) == 4
    feat_hw = input_hw // patch_size
    assert features.shape[2] == feat_hw and features.shape[3] == feat_hw


def test_galileo_hw_less_than_ps(
    tmp_path: pathlib.Path, monkeypatch: MonkeyPatch
) -> None:
    """Verify that the forward pass for Galileo works."""
    input_hw = 1
    patch_size = 4
    # We override the temporary directory so we don't retain the model weights outside
    # of this test.
    monkeypatch.setattr(tempfile, "gettempdir", lambda: tmp_path)
    galileo = GalileoModel(size=GalileoSize.NANO, patch_size=patch_size)

    inputs = [
        {
            "s2": torch.zeros((10, input_hw, input_hw), dtype=torch.float32),
            "s1": torch.zeros((2, input_hw, input_hw), dtype=torch.float32),
        }
    ]
    feature_list = galileo(inputs)
    # Should yield one feature map since there's only one output scale.
    assert len(feature_list) == 1
    features = feature_list[0]
    # features should be BxCxHxW.
    assert features.shape[0] == 1 and len(features.shape) == 4
    assert features.shape[2] == 1 and features.shape[3] == 1
