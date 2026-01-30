"""Tests for the rslearn.models.upsample module."""

import torch

from rslearn.models.component import FeatureMaps
from rslearn.models.upsample import Upsample
from rslearn.train.model_context import ModelContext


def test_upsample_nearest() -> None:
    """Test Upsample with nearest mode."""
    batch_size = 1
    num_channels = 1

    # Create a 2x2 input with known values
    # Input: [[1, 2],
    #         [3, 4]]
    input_tensor = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32
    )  # Shape: (1, 1, 2, 2)

    feature_maps = FeatureMaps([input_tensor])

    upsample = Upsample(scale_factor=2, mode="nearest")
    result = upsample(feature_maps, ModelContext(inputs=[], metadatas=[]))

    # Verify it is 2x upsampling so get 4x4 output.
    assert isinstance(result, FeatureMaps)
    assert len(result.feature_maps) == 1
    assert result.feature_maps[0].shape == (batch_size, num_channels, 4, 4)

    output = result.feature_maps[0][0, 0, :, :]  # Get the single channel output
    # Expected 4x4 output with nearest neighbor:
    # [[1, 1, 2, 2],
    #  [1, 1, 2, 2],
    #  [3, 3, 4, 4],
    #  [3, 3, 4, 4]]
    expected = torch.tensor(
        [
            [1.0, 1.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
            [3.0, 3.0, 4.0, 4.0],
            [3.0, 3.0, 4.0, 4.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(output, expected)
