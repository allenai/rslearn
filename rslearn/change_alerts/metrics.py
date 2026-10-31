"""Metrics for change alert models.

These are meant to be configured as other_metrics of a SegmentationTask (for the
change category) or a PerPixelTimestepTask (for the change timestep). They receive the
per-example CHW probabilities and the targets with "classes" and "valid".
"""

from typing import Any

import torch
from torchmetrics import Metric
from torchmetrics.classification import BinaryAUROC


def _iter_valid(
    preds: list[torch.Tensor] | torch.Tensor, targets: list[dict[str, Any]]
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Get (probs at valid pixels as NxC, labels at valid pixels) for each example."""
    result = []
    for pred, target in zip(preds, targets):
        labels = target["classes"].get_hw_tensor().long()
        valid = target["valid"].get_hw_tensor() > 0
        result.append((pred.permute(1, 2, 0)[valid], labels[valid]))
    return result


class BalancedAccuracy(Metric):
    """Mean over classes of the per-class recall (argmax accuracy at that class).

    Only classes that appear among the valid target pixels are averaged, so classes
    absent from the evaluation set (or the nodata class) do not affect the score.
    """

    correct: torch.Tensor
    total: torch.Tensor

    def __init__(self, num_classes: int) -> None:
        """Create a new BalancedAccuracy.

        Args:
            num_classes: the number of classes.
        """
        super().__init__()
        self.num_classes = num_classes
        self.add_state(
            "correct", default=torch.zeros(num_classes), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(
        self, preds: list[torch.Tensor] | torch.Tensor, targets: list[dict[str, Any]]
    ) -> None:
        """Update the per-class counts."""
        for probs, labels in _iter_valid(preds, targets):
            if len(labels) == 0:
                continue
            correct = (probs.argmax(dim=1) == labels).float()
            self.total.index_add_(0, labels, torch.ones_like(correct))
            self.correct.index_add_(0, labels, correct)

    def compute(self) -> torch.Tensor:
        """Compute the balanced accuracy."""
        present = self.total > 0
        if not present.any():
            return torch.tensor(0.0)
        return (self.correct[present] / self.total[present]).mean()


class ChangeAUROC(BinaryAUROC):
    """AUROC of detecting change versus no change from the category probabilities.

    The score is one minus the probability of the no change category, and the label is
    whether the target category is a change category.
    """

    def __init__(
        self,
        none_class: int = 1,
        thresholds: int | list[float] | None = None,
    ) -> None:
        """Create a new ChangeAUROC.

        Args:
            none_class: the class ID of the no change category.
            thresholds: passed to BinaryAUROC. If None, all scores at valid pixels are
                stored and the exact AUROC is computed. If set (e.g. to the number of
                thresholds), a binned AUROC that uses constant memory is computed.
        """
        super().__init__(thresholds=thresholds)
        self.none_class = none_class

    def update(  # type: ignore[override]
        self, preds: list[torch.Tensor] | torch.Tensor, targets: list[dict[str, Any]]
    ) -> None:
        """Accumulate the scores and labels at valid pixels."""
        valid = _iter_valid(preds, targets)
        if not valid:
            return
        scores = torch.cat(
            [1 - probs[:, self.none_class].float() for probs, _ in valid]
        )
        labels = torch.cat([(labels != self.none_class).long() for _, labels in valid])
        if len(labels) == 0:
            return
        super().update(scores, labels)


class TimestepToleranceAccuracy(Metric):
    """Fraction of valid pixels whose predicted timestep is within a tolerance."""

    correct: torch.Tensor
    total: torch.Tensor

    def __init__(self, tolerance: int = 1) -> None:
        """Create a new TimestepToleranceAccuracy.

        Args:
            tolerance: the maximum allowed difference in timestep index.
        """
        super().__init__()
        self.tolerance = tolerance
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self, preds: list[torch.Tensor] | torch.Tensor, targets: list[dict[str, Any]]
    ) -> None:
        """Update the counts."""
        for probs, labels in _iter_valid(preds, targets):
            diff = (probs.argmax(dim=1) - labels).abs()
            self.correct += (diff <= self.tolerance).sum()
            self.total += len(labels)

    def compute(self) -> torch.Tensor:
        """Compute the accuracy."""
        if self.total == 0:
            return torch.tensor(0.0)
        return self.correct / self.total
