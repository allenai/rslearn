"""Tests for rslearn.models.simple_time_series."""

from typing import Any

import pytest
import torch
from typing_extensions import override

from rslearn.models.component import FeatureExtractor, FeatureMaps
from rslearn.models.simple_time_series import SimpleTimeSeries
from rslearn.train.model_context import ModelContext
from rslearn.utils.raster_format import RasterImage


class IdentityFeatureExtractor(FeatureExtractor):
    """A feature extractor for testing that just returns its input."""

    @override
    def forward(self, context: ModelContext) -> Any:
        # Here we stack the inputs to get a BCHW feature map.
        # [:, 0] takes the first (and assumed only) timestep
        feat = torch.stack(
            [input_dict["image"].image[:, 0] for input_dict in context.inputs], dim=0
        )
        return FeatureMaps([feat])


def test_simple_time_series() -> None:
    """Just make sure SimpleTimeSeries performs mean temporal pooling properly."""
    extractor = IdentityFeatureExtractor()
    model = SimpleTimeSeries(
        encoder=extractor,
        image_channels=1,
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


def test_simple_time_series_with_image_keys() -> None:
    """Just make sure SimpleTimeSeries performs mean temporal pooling properly."""
    extractor = IdentityFeatureExtractor()
    model = SimpleTimeSeries(
        encoder=extractor,
        op="mean",
        backbone_channels=[(1, 1)],
        image_keys={"image": 1},
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
