from typing import Literal

import pytest
import torch

from rslearn.const import WGS84_PROJECTION
from rslearn.models.component import FeatureVector
from rslearn.train.model_context import ModelContext, SampleMetadata
from rslearn.train.tasks.regression import RegressionHead, RegressionTask
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
    metadata = SampleMetadata(
        window_group="",
        window_name="",
        window_bounds=(0, 0, 1, 1),
        crop_bounds=(0, 0, 1, 1),
        crop_idx=0,
        num_crops_in_window=1,
        time_range=None,
        projection=WGS84_PROJECTION,
        dataset_source=None,
    )
    features = task.process_output(raw_output, metadata)
    assert len(features) == 1
    feature = features[0]
    assert isinstance(feature, Feature)
    assert feature.properties[property_name] == pytest.approx(expected_value)


def test_accuracy_metric() -> None:
    """Verify accuracy metric produces the correct accuracy."""
    # Get the metrics object.
    task = RegressionTask(
        property_name="property_name",
        use_accuracy_metric=True,
        within_factor=0.1,
    )
    metrics = task.get_metrics()

    # Prepare example.
    targets = [
        {
            "value": torch.tensor(100, dtype=torch.float32),
            "valid": torch.tensor(1, dtype=torch.int32),
        },
        {
            "value": torch.tensor(100, dtype=torch.float32),
            "valid": torch.tensor(1, dtype=torch.int32),
        },
        {
            "value": torch.tensor(100, dtype=torch.float32),
            "valid": torch.tensor(1, dtype=torch.int32),
        },
        {
            "value": torch.tensor(100, dtype=torch.float32),
            "valid": torch.tensor(1, dtype=torch.int32),
        },
        {
            "value": torch.tensor(100, dtype=torch.float32),
            "valid": torch.tensor(1, dtype=torch.int32),
        },
    ]
    preds = torch.tensor(
        [
            # Exactly correct.
            100,
            # Within the right factor.
            95,
            109,
            # Incorrect.
            89,
            111,
        ],
        dtype=torch.float32,
    )

    # Accuracy should be 60%.
    metrics.update(preds, targets)
    results = metrics.compute()
    assert results["accuracy"] == pytest.approx(0.6)


def test_rmse_metric_with_scale_factor_and_mask() -> None:
    """Verify RMSE metric respects scale_factor and validity masking."""
    scale_factor = 0.1
    task = RegressionTask(
        property_name="property_name",
        scale_factor=scale_factor,
        metrics=("rmse",),
    )
    metrics = task.get_metrics()

    targets = [
        {"value": torch.tensor(10.0 * scale_factor), "valid": torch.tensor(1.0)},
        {"value": torch.tensor(20.0 * scale_factor), "valid": torch.tensor(1.0)},
        {"value": torch.tensor(0.0), "valid": torch.tensor(0.0)},
    ]
    preds = torch.tensor(
        [12.0 * scale_factor, 18.0 * scale_factor, 999.0 * scale_factor],
        dtype=torch.float32,
    )

    metrics.update(preds, targets)
    results = metrics.compute()
    # Only consider the two valid examples: errors are 2 and -2 => RMSE = 2.
    assert results["rmse"] == pytest.approx(2.0)


def test_mape_metric_with_scale_factor_and_mask() -> None:
    """Verify MAPE metric respects scale_factor and validity masking."""
    scale_factor = 0.01
    task = RegressionTask(
        property_name="property_name",
        scale_factor=scale_factor,
        metrics=("mape",),
    )
    metrics = task.get_metrics()

    targets = [
        {"value": torch.tensor(100.0 * scale_factor), "valid": torch.tensor(1.0)},
        {"value": torch.tensor(200.0 * scale_factor), "valid": torch.tensor(1.0)},
        {"value": torch.tensor(1.0 * scale_factor), "valid": torch.tensor(0.0)},
    ]
    preds = torch.tensor(
        [110.0 * scale_factor, 180.0 * scale_factor, 999.0 * scale_factor],
        dtype=torch.float32,
    )

    metrics.update(preds, targets)
    results = metrics.compute()
    # abs pct errors: |10|/100 = 0.1, |20|/200 = 0.1 => mean = 0.1
    assert results["mape"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("loss_mode", "expected"),
    [
        ("mse", 1.0),
        ("l1", 1.0),
        ("huber", 0.5),
    ],
)
def test_regression_head_loss_modes(
    loss_mode: Literal["mse", "l1", "huber"], expected: float
) -> None:
    head = RegressionHead(loss_mode=loss_mode)
    logits = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
    intermediates = FeatureVector(feature_vector=logits)
    targets = [
        {"value": torch.tensor(2.0), "valid": torch.tensor(1.0)},
        {"value": torch.tensor(2.0), "valid": torch.tensor(1.0)},
    ]
    output = head(
        intermediates=intermediates,
        context=ModelContext(inputs=[], metadatas=[]),
        targets=targets,
    )
    assert output.loss_dict["regress"] == pytest.approx(expected)
