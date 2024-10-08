import torch

from rslearn.const import WGS84_PROJECTION
from rslearn.train.tasks.classification import ClassificationTask


def test_positive_class_threshold():
    # Check that task returns different output depending on the threshold.
    probs = torch.tensor([0.7, 0.3], dtype=torch.float32)
    metadata = dict(
        projection=WGS84_PROJECTION,
        bounds=[0, 0, 1, 1],
    )

    # Default should use 0.5 threshold.
    task = ClassificationTask(property_name="cls", classes=["positive", "negative"])
    output = task.process_output(probs, metadata)
    assert output[0].properties["cls"] == "positive"

    task = ClassificationTask(
        property_name="cls",
        classes=["positive", "negative"],
        positive_class="positive",
        positive_class_threshold=0.6,
    )
    output = task.process_output(probs, metadata)
    assert output[0].properties["cls"] == "positive"

    task = ClassificationTask(
        property_name="cls",
        classes=["positive", "negative"],
        positive_class="positive",
        positive_class_threshold=0.75,
    )
    output = task.process_output(probs, metadata)
    assert output[0].properties["cls"] == "negative"

    # Now switch the class order.
    task = ClassificationTask(
        property_name="cls",
        classes=["negative", "positive"],
        positive_class="positive",
        positive_class_threshold=0.4,
    )
    output = task.process_output(probs, metadata)
    assert output[0].properties["cls"] == "negative"

    task = ClassificationTask(
        property_name="cls",
        classes=["negative", "positive"],
        positive_class="positive",
        positive_class_threshold=0.2,
    )
    output = task.process_output(probs, metadata)
    assert output[0].properties["cls"] == "positive"
