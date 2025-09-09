import pathlib
import tempfile
from typing import Any

import torch

from rslearn.models.galileo import GalileoModel, GalileoSize


def test_galileo(tmp_path: pathlib.Path, monkeypatch: Any) -> None:
    """Verify that the forward pass for Galileo works."""
    input_hw = 32
    # We override the temporary directory so we don't retain the model weights outside
    # of this test.
    monkeypatch.setattr(tempfile, "gettempdir", lambda: tmp_path)
    galileo = GalileoModel(size=GalileoSize.NANO)

    inputs = [
        {
            "s2": torch.zeros((10, input_hw, input_hw), dtype=torch.float32),
        }
    ]
    feature_list = galileo(inputs)
    # Should yield one feature map since there's only one output scale.
    assert len(feature_list) == 1
    features = feature_list[0]
    # features should be BxCxHxW.
    assert features.shape[0] == 1 and len(features.shape) == 4
    # feat_hw = input_hw // PATCH_SIZE
    # assert features.shape[2] == feat_hw and features.shape[3] == feat_hw
