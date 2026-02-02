"""Tests for rslearn.models.simple_time_series."""

from datetime import datetime
from typing import Any

import pytest
import torch
from typing_extensions import override

from rslearn.models.component import FeatureExtractor, FeatureMaps
from rslearn.models.simple_time_series import SimpleTimeSeries
from rslearn.train.model_context import ModelContext, RasterImage


class IdentityFeatureExtractor(FeatureExtractor):
    """A feature extractor for testing that just returns its input.

    Args:
        expected_timesteps: if set, raises an error if the input doesn't have
            exactly this many timesteps.
    """

    def __init__(self, expected_timesteps: int) -> None:
        super().__init__()
        self.expected_timesteps = expected_timesteps

    @override
    def forward(self, context: ModelContext) -> Any:
        # Here we stack the inputs to get a BCHW feature map.
        # We do temporal mean pooling across all timesteps.
        images = [input_dict["image"].image for input_dict in context.inputs]
        for image in images:
            if image.shape[1] != self.expected_timesteps:
                raise ValueError(
                    f"expected {self.expected_timesteps} timesteps, "
                    f"got {image.shape[1]}"
                )
        feat = torch.stack([image.mean(dim=1) for image in images], dim=0)
        return FeatureMaps([feat])


def test_simple_time_series() -> None:
    """Just make sure SimpleTimeSeries performs mean temporal pooling properly."""
    extractor = IdentityFeatureExtractor(expected_timesteps=1)
    model = SimpleTimeSeries(
        encoder=extractor,
        num_timesteps_per_forward_pass=1,
        op="mean",
        backbone_channels=[(1, 1)],
    )
    # Try with input with two timesteps, one is 5s and one is 6s.
    input_dict = {
        "image": RasterImage(
            torch.stack(
                [
                    5 * torch.ones((1, 4, 4), dtype=torch.float32),
                    6 * torch.ones((1, 4, 4), dtype=torch.float32),
                ],
                dim=1,
            ),
            timestamps=[
                (datetime(2026, 1, 7), datetime(2026, 1, 7)),
                (datetime(2026, 1, 8), datetime(2026, 1, 8)),
            ],
        ),
    }
    context = ModelContext(
        inputs=[input_dict],
        metadatas=[],
    )
    output = model(context)
    # The result should have one feature map. Since we use IdentityFeatureExtractor,
    # the values should be mean of the two inputs.
    assert isinstance(output, FeatureMaps)
    assert len(output.feature_maps) == 1
    feat = output.feature_maps[0]
    assert feat.shape == (1, 1, 4, 4)
    assert feat[0, 0, 0, 0] == pytest.approx(5.5)


def test_simple_time_series_multi_timestep_batching() -> None:
    """Test batching multiple timesteps together.

    This tests the case where we want to pass multiple timesteps to the model at once.
    For example, if we have a CTHW tensor with C=3, T=4 and we set
    num_timesteps_per_forward_pass=2, then the model receives 2 timesteps at a time.
    """
    # Input: C=3, T=4, H=8, W=8
    # With num_timesteps_per_forward_pass=2, this is split into 2 batches of 2 timesteps.
    # Create 4 timesteps with values 1, 2, 3, 4 so we can verify the mean.
    timesteps = []
    for i in range(4):
        timesteps.append((i + 1) * torch.ones((3, 8, 8), dtype=torch.float32))
    input_dict = {
        "image": RasterImage(
            torch.stack(timesteps, dim=1),  # Shape: (3, 4, 8, 8)
            timestamps=[
                (datetime(2026, 1, i + 1), datetime(2026, 1, i + 1)) for i in range(4)
            ],
        ),
    }
    context = ModelContext(
        inputs=[input_dict],
        metadatas=[],
    )

    extractor = IdentityFeatureExtractor(expected_timesteps=2)
    model = SimpleTimeSeries(
        encoder=extractor,
        num_timesteps_per_forward_pass=2,
        op="max",
        backbone_channels=[(1, 3)],
    )
    output = model(context)

    # The extractor does temporal mean over 2 timesteps, so:
    # - Batch 0: mean of timesteps 0,1 = (1+2)/2 = 1.5
    # - Batch 1: mean of timesteps 2,3 = (3+4)/2 = 3.5
    # Then SimpleTimeSeries does max over these two batches => 3.5
    assert isinstance(output, FeatureMaps)
    assert len(output.feature_maps) == 1
    feat = output.feature_maps[0]
    assert feat.shape == (1, 3, 8, 8)
    assert feat[0, 0, 0, 0] == pytest.approx(3.5)


def test_simple_time_series_with_image_keys() -> None:
    """Test SimpleTimeSeries with image_keys as a list."""
    extractor = IdentityFeatureExtractor(expected_timesteps=1)
    model = SimpleTimeSeries(
        encoder=extractor,
        num_timesteps_per_forward_pass=1,
        op="mean",
        backbone_channels=[(1, 1)],
        image_keys=["image"],
    )
    # Try with input with two timesteps, one is 5s and one is 6s.
    input_dict = {
        "image": RasterImage(
            torch.stack(
                [
                    5 * torch.ones((1, 4, 4), dtype=torch.float32),
                    6 * torch.ones((1, 4, 4), dtype=torch.float32),
                ],
                dim=1,
            )
        ),
    }
    context = ModelContext(
        inputs=[input_dict],
        metadatas=[],
    )
    output = model(context)
    # The result should have one feature map. Since we use IdentityFeatureExtractor,
    # the values should be mean of the two inputs.
    assert isinstance(output, FeatureMaps)
    assert len(output.feature_maps) == 1
    feat = output.feature_maps[0]
    assert feat.shape == (1, 1, 4, 4)
    assert feat[0, 0, 0, 0] == pytest.approx(5.5)
