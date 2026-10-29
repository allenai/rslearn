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
    """Changes scored higher than no change gives AUROC 1, ignoring invalid pixels."""
    metric = ChangeAUROC(none_class=1)
    # The last pixel is a change scored as no change, but it is invalid.
    metric.update(
        [_probs([1, 1, 2, 3, 1], 4)], [_target([1, 1, 2, 3, 2], [1, 1, 1, 1, 0])]
    )
    assert metric.compute().item() == pytest.approx(1.0)


def test_change_auroc_binned() -> None:
    """The binned AUROC is also 1 when the bins separate the scores."""
    metric = ChangeAUROC(none_class=1, thresholds=100)
    # The last pixel is a change scored as no change, but it is invalid.
    metric.update(
        [_probs([1, 1, 2, 3, 1], 4)], [_target([1, 1, 2, 3, 2], [1, 1, 1, 1, 0])]
    )
    assert metric.compute().item() == pytest.approx(1.0)


def test_change_auroc_binned_coarse() -> None:
    """Coarse bins that do not separate the scores lower the binned AUROC."""
    # Change scores (one minus the no change probability) are 0.1 and 0.6 for the no
    # change pixels and 0.7 and 0.9 for the change pixels.
    scores = torch.tensor([0.1, 0.6, 0.7, 0.9])
    probs = torch.zeros((3, 1, 4))
    probs[1, 0] = 1 - scores
    probs[2, 0] = scores
    target = _target([1, 1, 2, 2], [1, 1, 1, 1])

    exact = ChangeAUROC(none_class=1)
    exact.update([probs], [target])
    assert exact.compute().item() == pytest.approx(1.0)

    # With thresholds 0, 0.5, and 1, the ROC points are (0, 0), (0.5, 1), and (1, 1),
    # since the 0.6 no change score is above 0.5 like the change scores.
    binned = ChangeAUROC(none_class=1, thresholds=3)
    binned.update([probs], [target])
    assert binned.compute().item() == pytest.approx(0.75)


def test_timestep_tolerance_accuracy() -> None:
    """Predictions within the tolerance count as correct."""
    metric = TimestepToleranceAccuracy(tolerance=1)
    metric.update([_probs([3, 5, 0], 12)], [_target([4, 8, 0], [1, 1, 0])])
    assert metric.compute().item() == pytest.approx(0.5)
