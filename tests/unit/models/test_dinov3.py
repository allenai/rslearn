"""Test DinoV3 model."""

import pytest
import torch

from rslearn.models.dinov3 import DinoV3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
def test_dinov3() -> None:
    """Verify that the forward pass for DinoV3 works."""
    input_hw = 32
    # We override the temporary directory so we don't retain the model weights outside
    # of this test.

    inputs = [
        {
            "s2": torch.zeros(
                (
                    3,
                    input_hw,
                    input_hw,
                ),
                dtype=torch.float32,
                device=DEVICE,
            ),
            "landsat": torch.zeros(
                (
                    3,
                    input_hw,
                    input_hw,
                ),
                dtype=torch.float32,
                device=DEVICE,
            ),
        }
    ]
    dinov3 = DinoV3(checkpoint_dir=None).to(DEVICE)
    with torch.no_grad():
        feature_list = dinov3(inputs)
    assert (
        feature_list[0].shape == torch.Size([1, 1024, 2, 2]) and len(feature_list) == 1
    )
