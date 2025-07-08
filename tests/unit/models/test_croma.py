import os
import shutil
import tempfile

import torch

from rslearn.models.croma import PATCH_SIZE, Croma, CromaModality, CromaSize


def test_croma() -> None:
    # Make sure the forward pass works.
    input_hw = 32
    croma = Croma(
        size=CromaSize.BASE, modality=CromaModality.SENTINEL2, image_resolution=input_hw
    )
    inputs = [
        {
            "sentinel2": torch.zeros((12, input_hw, input_hw), dtype=torch.float32),
        }
    ]
    feature_list = croma(inputs)
    # Should yield one feature map since there's only one output scale.
    assert len(feature_list) == 1
    features = feature_list[0]
    # features should be BxCxHxW.
    assert features.shape[0] == 1 and len(features.shape) == 4
    feat_hw = input_hw // PATCH_SIZE
    assert features.shape[2] == feat_hw and features.shape[3] == feat_hw

    # Delete any cached models.
    cache_dir = os.path.join(tempfile.gettempdir(), "rslearn_cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
