import pathlib
import tempfile

import torch
from pytest import MonkeyPatch

from rslearn.models.presto import Presto


def test_presto(tmp_path: pathlib.Path, monkeypatch: MonkeyPatch) -> None:
    """Verify that the forward pass for Presto works."""
    input_hw = 16
    # We override the temporary directory so we don't retain the model weights outside
    # of this test.
    monkeypatch.setattr(tempfile, "gettempdir", lambda: tmp_path)
    presto = Presto()

    inputs = [
        {
            "s2": torch.zeros((10, input_hw, input_hw), dtype=torch.float32),
            "s1": torch.zeros((2, input_hw, input_hw), dtype=torch.float32),
            "era5": torch.zeros((2, input_hw, input_hw), dtype=torch.float32),
        }
    ]
    feature_list = presto(inputs)
    # Should yield one feature map since there's only one output scale.
    assert len(feature_list) == 1
    features = feature_list[0]
    # features should be BxCxHxW.
    assert features.shape[0] == 1 and len(features.shape) == 4
    assert features.shape[2] == input_hw and features.shape[3] == input_hw


def test_presto_mt(tmp_path: pathlib.Path, monkeypatch: MonkeyPatch) -> None:
    """Verify that the forward pass for Presto works with multiple timesteps."""
    input_hw = 32
    num_timesteps = 10
    # We override the temporary directory so we don't retain the model weights outside
    # of this test.
    monkeypatch.setattr(tempfile, "gettempdir", lambda: tmp_path)
    presto = Presto()
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
    feature_list = presto(inputs)
    # Should yield one feature map since there's only one output scale.
    assert len(feature_list) == 1
    features = feature_list[0]
    # features should be BxCxHxW.
    assert features.shape[0] == 1 and len(features.shape) == 4
    assert features.shape[2] == input_hw and features.shape[3] == input_hw
