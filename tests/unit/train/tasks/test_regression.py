import pytest
import torch

from rslearn.const import WGS84_PROJECTION
from rslearn.train.tasks.regression import RegressionTask
from rslearn.utils.feature import Feature


def test_process_output() -> None:
    """Ensure that RegressionTask.process_output produces correct Feature."""
    property_name = "property_name"
    scale_factor = 0.01
    task = RegressionTask(
        property_name=property_name,
        scale_factor=scale_factor,
    )
    expected_value = 5
    raw_output = torch.tensor(expected_value * scale_factor)
    metadata = dict(
        projection=WGS84_PROJECTION,
        bounds=[0, 0, 1, 1],
    )
    features = task.process_output(raw_output, metadata)
    assert len(features) == 1
    feature = features[0]
    assert isinstance(feature, Feature)
    assert feature.properties[property_name] == pytest.approx(expected_value)
