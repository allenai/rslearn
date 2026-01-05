import pathlib
import tempfile
from datetime import datetime

import torch
from pytest import MonkeyPatch

from rslearn.models.galileo import GalileoModel, GalileoSize
from rslearn.train.model_context import ModelContext
from rslearn.utils.raster_format import RasterImage


def test_galileo(tmp_path: pathlib.Path, monkeypatch: MonkeyPatch) -> None:
    """Verify that the forward pass for Galileo works."""
    input_hw = 8
    patch_size = 4
    # We override the temporary directory so we don't retain the model weights outside
    # of this test.
    monkeypatch.setattr(tempfile, "gettempdir", lambda: tmp_path)
    galileo = GalileoModel(size=GalileoSize.NANO, patch_size=patch_size)

    inputs = [
        {
            "s2": RasterImage(
                torch.zeros((10, 1, input_hw, input_hw), dtype=torch.float32)
            ),
            "s1": RasterImage(
                torch.zeros((2, 1, input_hw, input_hw), dtype=torch.float32)
            ),
            "era5": RasterImage(
                torch.zeros((2, 1, input_hw, input_hw), dtype=torch.float32)
            ),
            "srtm": RasterImage(
                torch.zeros((2, 1, input_hw, input_hw), dtype=torch.float32)
            ),
            "latlon": RasterImage(
                torch.zeros((2, 1, input_hw, input_hw), dtype=torch.float32)
            ),
        }
    ]
    feature_list = galileo(ModelContext(inputs=inputs, metadatas=[])).feature_maps
    # Should yield one feature map since there's only one output scale.
    assert len(feature_list) == 1
    features = feature_list[0]
    # features should be BxCxHxW.
    assert features.shape[0] == 1 and len(features.shape) == 4
    feat_hw = input_hw // patch_size
    assert features.shape[2] == feat_hw and features.shape[3] == feat_hw


def test_galileo_mt(tmp_path: pathlib.Path, monkeypatch: MonkeyPatch) -> None:
    """Verify that the forward pass for Galileo works."""
    input_hw = 8
    patch_size = 4
    num_timesteps = 2
    # We override the temporary directory so we don't retain the model weights outside
    # of this test.
    monkeypatch.setattr(tempfile, "gettempdir", lambda: tmp_path)
    galileo = GalileoModel(size=GalileoSize.NANO, patch_size=patch_size)

    inputs = [
        {
            "s2": RasterImage(
                torch.zeros(
                    (10, num_timesteps, input_hw, input_hw), dtype=torch.float32
                )
            ),
            # srtm is static in time so only has one time dimension
            "srtm": RasterImage(
                torch.zeros((2, 1, input_hw, input_hw), dtype=torch.float32)
            ),
        }
    ]
    feature_list = galileo(ModelContext(inputs=inputs, metadatas=[])).feature_maps
    # Should yield one feature map since there's only one output scale.
    assert len(feature_list) == 1
    features = feature_list[0]
    # features should be BxCxHxW.
    assert features.shape[0] == 1 and len(features.shape) == 4
    feat_hw = input_hw // patch_size
    assert features.shape[2] == feat_hw and features.shape[3] == feat_hw


def test_galileo_mt_with_timestamps(
    tmp_path: pathlib.Path, monkeypatch: MonkeyPatch
) -> None:
    """Verify that the forward pass for Galileo works."""
    input_hw = 8
    patch_size = 4
    num_timesteps = 2
    # We override the temporary directory so we don't retain the model weights outside
    # of this test.
    monkeypatch.setattr(tempfile, "gettempdir", lambda: tmp_path)
    galileo = GalileoModel(size=GalileoSize.NANO, patch_size=patch_size)

    inputs = [
        {
            "s2": RasterImage(
                image=torch.zeros(
                    (10, num_timesteps, input_hw, input_hw), dtype=torch.float32
                ),
                timestamps=[
                    (datetime(2025, x, 1), datetime(2025, x, 1))
                    for x in range(1, num_timesteps + 1)
                ],
            ),
            # srtm is static in time so only has one time dimension
            "srtm": RasterImage(
                image=torch.zeros((2, 1, input_hw, input_hw), dtype=torch.float32),
                timestamps=[((datetime(2017, 6, 1), datetime(2017, 6, 1)))],
            ),
        }
    ]
    feature_list = galileo(ModelContext(inputs=inputs, metadatas=[])).feature_maps
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
            "s2": RasterImage(
                torch.zeros((10, 1, input_hw, input_hw), dtype=torch.float32)
            ),
            "s1": RasterImage(
                torch.zeros((2, 1, input_hw, input_hw), dtype=torch.float32)
            ),
            "era5": RasterImage(
                torch.zeros((2, 1, input_hw, input_hw), dtype=torch.float32)
            ),
            "srtm": RasterImage(
                torch.zeros((2, 1, input_hw, input_hw), dtype=torch.float32)
            ),
            "latlon": RasterImage(
                torch.zeros((2, 1, input_hw, input_hw), dtype=torch.float32)
            ),
        }
    ]
    feature_list = galileo(ModelContext(inputs=inputs, metadatas=[])).feature_maps
    # Should yield one feature map since there's only one output scale.
    assert len(feature_list) == 1
    features = feature_list[0]
    # features should be BxCxHxW.
    assert features.shape[0] == 1 and len(features.shape) == 4
    assert features.shape[2] == 1 and features.shape[3] == 1
