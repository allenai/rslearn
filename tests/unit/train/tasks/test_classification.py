import pytest
import torch

from rslearn.train.model_context import SampleMetadata
from rslearn.train.tasks.classification import ClassificationTask


def test_positive_class_threshold(empty_sample_metadata: SampleMetadata) -> None:
    # Check that task returns different output depending on the threshold.
    probs = torch.tensor([0.7, 0.3], dtype=torch.float32)

    # Default should use 0.5 threshold.
    task = ClassificationTask(property_name="cls", classes=["positive", "negative"])
    output = task.process_output(probs, empty_sample_metadata)
    assert output[0].properties is not None
    assert output[0].properties["cls"] == "positive"

    task = ClassificationTask(
        property_name="cls",
        classes=["positive", "negative"],
        positive_class="positive",
        positive_class_threshold=0.6,
    )
    output = task.process_output(probs, empty_sample_metadata)
    assert output[0].properties is not None
    assert output[0].properties["cls"] == "positive"

    task = ClassificationTask(
        property_name="cls",
        classes=["positive", "negative"],
        positive_class="positive",
        positive_class_threshold=0.75,
    )
    output = task.process_output(probs, empty_sample_metadata)
    assert output[0].properties is not None
    assert output[0].properties["cls"] == "negative"

    # Now switch the class order.
    task = ClassificationTask(
        property_name="cls",
        classes=["negative", "positive"],
        positive_class="positive",
        positive_class_threshold=0.4,
    )
    output = task.process_output(probs, empty_sample_metadata)
    assert output[0].properties is not None
    assert output[0].properties["cls"] == "negative"

    task = ClassificationTask(
        property_name="cls",
        classes=["negative", "positive"],
        positive_class="positive",
        positive_class_threshold=0.2,
    )
    output = task.process_output(probs, empty_sample_metadata)
    assert output[0].properties is not None
    assert output[0].properties["cls"] == "positive"


def test_per_class_f1() -> None:
    targets = [
        {
            "class": torch.tensor(0, dtype=torch.int32),
            "valid": torch.tensor(1, dtype=torch.int32),
        },
        {
            "class": torch.tensor(0, dtype=torch.int32),
            "valid": torch.tensor(1, dtype=torch.int32),
        },
        {
            "class": torch.tensor(0, dtype=torch.int32),
            "valid": torch.tensor(1, dtype=torch.int32),
        },
        {
            "class": torch.tensor(1, dtype=torch.int32),
            "valid": torch.tensor(1, dtype=torch.int32),
        },
        {
            "class": torch.tensor(2, dtype=torch.int32),
            "valid": torch.tensor(1, dtype=torch.int32),
        },
    ]
    preds = torch.tensor(
        [
            # gt=0
            [0.8, 0.1, 0.1],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            # gt=1
            [0.1, 0.8, 0.1],
            # gt=2
            [0.1, 0.8, 0.1],
        ],
        dtype=torch.float32,
    )

    # gt=0: 2 TP, 0 FP, 1 FN
    # gt=1: 1 TP, 2 FP, 0 FN
    # gt=2: 0 TP, 0 FP, 1 FN
    task = ClassificationTask(
        property_name="ignored", classes=["0", "1", "2"], enable_f1_metric=True
    )
    metrics = task.get_metrics()
    metrics.update(preds, targets)
    results = metrics.compute()
    assert results["0_precision"] == pytest.approx(1)
    assert results["0_recall"] == pytest.approx(2 / 3)
    assert results["1_precision"] == pytest.approx(1 / 3)
    assert results["1_recall"] == pytest.approx(1)
    assert results["2_precision"] == pytest.approx(0)
    assert results["2_recall"] == pytest.approx(0)


def test_auroc_partial() -> None:
    # Binary case with hand-computable macro AUROC.
    # Labels are [0, 0, 1, 1] with class-1 probabilities [0.1, 0.4, 0.35, 0.8].
    # For 2 classes, macro one-vs-rest AUROC equals the class-1 AUROC, i.e. the
    # fraction of (positive, negative) score pairs that are correctly ranked:
    #   positives (label 1) scores: 0.35, 0.8
    #   negatives (label 0) scores: 0.1, 0.4
    #   correctly ranked: (0.35>0.1), (0.8>0.1), (0.8>0.4) -> 3 of 4 -> 0.75
    targets = [
        {
            "class": torch.tensor(0, dtype=torch.int32),
            "valid": torch.tensor(1, dtype=torch.int32),
        },
        {
            "class": torch.tensor(0, dtype=torch.int32),
            "valid": torch.tensor(1, dtype=torch.int32),
        },
        {
            "class": torch.tensor(1, dtype=torch.int32),
            "valid": torch.tensor(1, dtype=torch.int32),
        },
        {
            "class": torch.tensor(1, dtype=torch.int32),
            "valid": torch.tensor(1, dtype=torch.int32),
        },
    ]
    preds = torch.tensor(
        [
            [0.9, 0.1],
            [0.6, 0.4],
            [0.65, 0.35],
            [0.2, 0.8],
        ],
        dtype=torch.float32,
    )

    task = ClassificationTask(
        property_name="ignored", classes=["0", "1"], enable_auroc=True
    )
    metrics = task.get_metrics()
    metrics.update(preds, targets)
    results = metrics.compute()
    assert results["auroc"] == pytest.approx(0.75)


def test_auroc_perfect() -> None:
    # Perfectly separable predictions should yield AUROC of 1.0.
    targets = [
        {
            "class": torch.tensor(0, dtype=torch.int32),
            "valid": torch.tensor(1, dtype=torch.int32),
        },
        {
            "class": torch.tensor(0, dtype=torch.int32),
            "valid": torch.tensor(1, dtype=torch.int32),
        },
        {
            "class": torch.tensor(1, dtype=torch.int32),
            "valid": torch.tensor(1, dtype=torch.int32),
        },
        {
            "class": torch.tensor(1, dtype=torch.int32),
            "valid": torch.tensor(1, dtype=torch.int32),
        },
    ]
    preds = torch.tensor(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
        ],
        dtype=torch.float32,
    )

    task = ClassificationTask(
        property_name="ignored", classes=["0", "1"], enable_auroc=True
    )
    metrics = task.get_metrics()
    metrics.update(preds, targets)
    results = metrics.compute()
    assert results["auroc"] == pytest.approx(1.0)


def test_prauc_partial() -> None:
    # Binary case with hand-computable macro PRAUC (average precision).
    # Labels are [0, 0, 1, 1] with class-1 probabilities [0.1, 0.4, 0.35, 0.8].
    # Average precision uses AP = sum_n (R_n - R_{n-1}) * P_n over score thresholds.
    #   Class 1: ranking by class-1 prob gives 0.8(pos), 0.4(neg), 0.35(pos), 0.1(neg)
    #     -> (1/2)*1 + (1/2)*(2/3) = 5/6.
    #   Class 0: ranking by class-0 prob gives 0.9(pos), 0.65(neg), 0.6(pos), 0.2(neg)
    #     -> (1/2)*1 + (1/2)*(2/3) = 5/6.
    #   Macro average = 5/6 ~= 0.8333.
    targets = [
        {
            "class": torch.tensor(0, dtype=torch.int32),
            "valid": torch.tensor(1, dtype=torch.int32),
        },
        {
            "class": torch.tensor(0, dtype=torch.int32),
            "valid": torch.tensor(1, dtype=torch.int32),
        },
        {
            "class": torch.tensor(1, dtype=torch.int32),
            "valid": torch.tensor(1, dtype=torch.int32),
        },
        {
            "class": torch.tensor(1, dtype=torch.int32),
            "valid": torch.tensor(1, dtype=torch.int32),
        },
    ]
    preds = torch.tensor(
        [
            [0.9, 0.1],
            [0.6, 0.4],
            [0.65, 0.35],
            [0.2, 0.8],
        ],
        dtype=torch.float32,
    )

    task = ClassificationTask(
        property_name="ignored", classes=["0", "1"], enable_prauc=True
    )
    metrics = task.get_metrics()
    metrics.update(preds, targets)
    results = metrics.compute()
    assert results["prauc"] == pytest.approx(5 / 6)


def test_prauc_perfect() -> None:
    # Perfectly separable predictions should yield PRAUC of 1.0.
    targets = [
        {
            "class": torch.tensor(0, dtype=torch.int32),
            "valid": torch.tensor(1, dtype=torch.int32),
        },
        {
            "class": torch.tensor(0, dtype=torch.int32),
            "valid": torch.tensor(1, dtype=torch.int32),
        },
        {
            "class": torch.tensor(1, dtype=torch.int32),
            "valid": torch.tensor(1, dtype=torch.int32),
        },
        {
            "class": torch.tensor(1, dtype=torch.int32),
            "valid": torch.tensor(1, dtype=torch.int32),
        },
    ]
    preds = torch.tensor(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
        ],
        dtype=torch.float32,
    )

    task = ClassificationTask(
        property_name="ignored", classes=["0", "1"], enable_prauc=True
    )
    metrics = task.get_metrics()
    metrics.update(preds, targets)
    results = metrics.compute()
    assert results["prauc"] == pytest.approx(1.0)
