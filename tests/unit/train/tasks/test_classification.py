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
