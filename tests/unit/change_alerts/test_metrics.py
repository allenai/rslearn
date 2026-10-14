"""Unit tests for rslearn.change_alerts.metrics."""

import pytest
import torch

from rslearn.change_alerts.metrics import (
    BalancedAccuracy,
    ChangeAUROC,
    TimestepToleranceAccuracy,
)
from rslearn.train.model_context import RasterImage


def _target(classes: list[int], valid: list[int]) -> dict[str, RasterImage]:
    return {
        "classes": RasterImage(torch.tensor(classes).view(1, 1, 1, -1)),
        "valid": RasterImage(
            torch.tensor(valid, dtype=torch.float32).view(1, 1, 1, -1)
        ),
    }


def _probs(predicted: list[int], num_classes: int) -> torch.Tensor:
    """Make CHW (H=1) one-hot-ish probabilities for the predicted classes."""
    probs = torch.full((num_classes, 1, len(predicted)), 0.1)
    for idx, cls in enumerate(predicted):
        probs[cls, 0, idx] = 0.9
    return probs


def test_balanced_accuracy() -> None:
    """Classes are weighted equally, and absent classes and invalid pixels ignored."""
    metric = BalancedAccuracy(num_classes=4)
    # Class 1: 3/4 correct, class 2: 0/1 correct; the last pixel is invalid.
    metric.update(
        [_probs([1, 1, 1, 2, 1, 0], 4)],
        [_target([1, 1, 1, 1, 2, 3], [1, 1, 1, 1, 1, 0])],
    )
    assert metric.compute().item() == pytest.approx((0.75 + 0.0) / 2)


def test_change_auroc() -> None:
    """Changes scored higher than no change gives AUROC 1."""
    metric = ChangeAUROC(none_class=1)
    metric.update([_probs([1, 1, 2, 3], 4)], [_target([1, 1, 2, 3], [1, 1, 1, 1])])
    assert metric.compute().item() == pytest.approx(1.0)


def test_timestep_tolerance_accuracy() -> None:
    """Predictions within the tolerance count as correct."""
    metric = TimestepToleranceAccuracy(tolerance=1)
    metric.update([_probs([3, 5, 0], 12)], [_target([4, 8, 0], [1, 1, 0])])
    assert metric.compute().item() == pytest.approx(0.5)
