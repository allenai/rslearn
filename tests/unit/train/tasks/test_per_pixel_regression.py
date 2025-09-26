import pytest
import torch

from rslearn.train.tasks.per_pixel_regression import (
    PerPixelRegressionHead,
    PerPixelRegressionTask,
)


def test_process_inputs() -> None:
    """Verify converting to input works with scale factor and nodata value."""
    task = PerPixelRegressionTask(
        scale_factor=0.1,
        nodata_value=-1,
    )
    # We use 1x2x2 input with one invalid pixel and three different values.
    _, target_dict = task.process_inputs(
        raw_inputs={
            "targets": torch.tensor([[[1, 2], [-1, 3]]]),
        },
        metadata={},
    )
    assert target_dict["values"].shape == (2, 2)
    assert target_dict["values"][0, 0] == pytest.approx(0.1)
    assert target_dict["values"][0, 1] == pytest.approx(0.2)
    assert target_dict["values"][1, 1] == pytest.approx(0.3)
    assert torch.all(target_dict["valid"] == torch.tensor([[1, 1], [0, 1]]))


def test_process_output() -> None:
    """Ensure that PerPixelRegressionTask.process_output works."""
    scale_factor = 0.1
    task = PerPixelRegressionTask(
        scale_factor=scale_factor,
    )
    output = task.process_output(
        raw_output=torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
        metadata={},
    )
    assert torch.all(output == torch.tensor([[[1, 2], [3, 4]]]))


def test_head() -> None:
    """Verify that the head masks out invalid pixels."""
    head = PerPixelRegressionHead(loss_mode="mse")
    logits = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32)[None, None, :, :]
    target_dict = {
        "values": torch.tensor([[2, 2], [2, 4]], dtype=torch.float32),
        "valid": torch.tensor([[1, 1], [1, 0]], dtype=torch.long),
    }
    _, losses = head(
        logits=logits,
        inputs=[{}],
        targets=[target_dict],
    )
    assert losses["regress"] == 1


def test_mse_metric() -> None:
    """Verify mean squared error metric works with customized scale_factor."""
    task = PerPixelRegressionTask(
        scale_factor=0.1,
        metric_mode="mse",
        nodata_value=-1,
    )
    metrics = task.get_metrics()

    # Prepare example.
    _, target_dict = task.process_inputs(
        raw_inputs={
            "targets": torch.tensor([[[1, 2], [-1, 3]]]),
        },
        metadata={},
    )
    preds = torch.tensor([[0.1, 0.1], [0.1, 0.1]])[None, None, :, :]

    # Accuracy should be (0 + 0.01 + 0.04) / 3 = 0.05 / 3.
    metrics.update(preds, [target_dict])
    results = metrics.compute()
    assert results["mse"] == pytest.approx(0.05 / 3)
