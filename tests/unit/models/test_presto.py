import pathlib
import tempfile
from datetime import datetime

import torch
from einops import rearrange
from pytest import MonkeyPatch

from rslearn.models.presto import Presto
from rslearn.train.model_context import ModelContext
from rslearn.utils.raster_format import RasterImage


def test_presto(tmp_path: pathlib.Path, monkeypatch: MonkeyPatch) -> None:
    """Verify that the forward pass for Presto works."""
    input_hw = 16
    # We override the temporary directory so we don't retain the model weights outside
    # of this test.
    monkeypatch.setattr(tempfile, "gettempdir", lambda: tmp_path)
    # we use a small pixel batch size here that doesn't divide cleanly
    # into (b * h * w) so that we
    # test the indexing functionality
    presto = Presto(pixel_batch_size=5)

    inputs = [
        {
            "s2": RasterImage(
                torch.zeros((10, 1, input_hw, input_hw), dtype=torch.float32),
                timestamps=[
                    (datetime(2025, x, 1), datetime(2025, x, 1)) for x in range(1, 2)
                ],
            ),
            "s1": RasterImage(
                torch.zeros((2, 1, input_hw, input_hw), dtype=torch.float32),
                timestamps=[
                    (datetime(2025, x, 1), datetime(2025, x, 1)) for x in range(1, 2)
                ],
            ),
            "era5": RasterImage(
                torch.zeros((2, 1, input_hw, input_hw), dtype=torch.float32),
                timestamps=[
                    (datetime(2025, x, 1), datetime(2025, x, 1)) for x in range(1, 2)
                ],
            ),
        }
    ]
    feature_list = presto(ModelContext(inputs=inputs, metadatas=[])).feature_maps
    # Should yield one feature map since there's only one output scale.
    assert len(feature_list) == 1
    features = feature_list[0]
    # features should be BxCxHxW.
    assert features.shape[0] == 1 and len(features.shape) == 4
    assert features.shape[2] == input_hw and features.shape[3] == input_hw
    # we initialize the output features to 0. This makes sure no
    # d is all 0s since this indicates something went wrong with out indexing
    features = torch.sum(rearrange(features, "b d h w -> (b h w) d"), dim=-1)
    assert not (features == 0).any()


def test_presto_mt(tmp_path: pathlib.Path, monkeypatch: MonkeyPatch) -> None:
    """Verify that the forward pass for Presto works with multiple timesteps."""
    input_hw = 32
    num_timesteps = 10
    # We override the temporary directory so we don't retain the model weights outside
    # of this test.
    monkeypatch.setattr(tempfile, "gettempdir", lambda: tmp_path)
    # we use a small pixel batch size here that doesn't divide cleanly
    # into (b * h * w) so that we
    # test the indexing functionality
    presto = Presto(pixel_batch_size=7)
    inputs = [
        {
            "s2": RasterImage(
                torch.zeros(
                    (10, num_timesteps, input_hw, input_hw), dtype=torch.float32
                ),
                timestamps=[
                    (datetime(2025, x, 1), datetime(2025, x, 1))
                    for x in range(1, num_timesteps + 1)
                ],
            ),
            "s1": RasterImage(
                torch.zeros(
                    (2, num_timesteps, input_hw, input_hw), dtype=torch.float32
                ),
                timestamps=[
                    (datetime(2025, x, 1), datetime(2025, x, 1))
                    for x in range(1, num_timesteps + 1)
                ],
            ),
            "era5": RasterImage(
                torch.zeros(
                    (2, num_timesteps, input_hw, input_hw), dtype=torch.float32
                ),
                timestamps=[
                    (datetime(2025, x, 1), datetime(2025, x, 1))
                    for x in range(1, num_timesteps + 1)
                ],
            ),
        }
    ]
    feature_list = presto(ModelContext(inputs=inputs, metadatas=[])).feature_maps
    # Should yield one feature map since there's only one output scale.
    assert len(feature_list) == 1
    features = feature_list[0]
    # features should be BxCxHxW.
    assert features.shape[0] == 1 and len(features.shape) == 4
    assert features.shape[2] == input_hw and features.shape[3] == input_hw
    # we initialize the output features to 0. This makes sure no
    # d is all 0s since this indicates something went wrong with out indexing
    features = torch.sum(rearrange(features, "b d h w -> (b h w) d"), dim=-1)
    assert not (features == 0).any()
