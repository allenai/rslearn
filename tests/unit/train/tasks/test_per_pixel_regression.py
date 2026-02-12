from typing import Literal

import numpy as np
import pytest
import torch

from rslearn.models.component import FeatureMaps
from rslearn.train.model_context import ModelContext, RasterImage, SampleMetadata
from rslearn.train.tasks.per_pixel_regression import (
    PerPixelRegressionHead,
    PerPixelRegressionTask,
)
from rslearn.utils.feature import Feature


def _make_metadata(
    empty_sample_metadata: SampleMetadata,
    height: int,
    width: int,
    *,
    window_bounds: tuple[int, int, int, int] | None = None,
    crop_bounds: tuple[int, int, int, int] | None = None,
    window_group: str = "",
    window_name: str = "",
) -> SampleMetadata:
    window_bounds = window_bounds or (0, 0, width, height)
    crop_bounds = crop_bounds or window_bounds
    return SampleMetadata(
        window_group=window_group,
        window_name=window_name,
        window_bounds=window_bounds,
        crop_bounds=crop_bounds,
        crop_idx=0,
        num_crops_in_window=1,
        time_range=None,
        projection=empty_sample_metadata.projection,
        dataset_source=None,
    )


def test_process_inputs(empty_sample_metadata: SampleMetadata) -> None:
    """Verify converting to input works with scale factor and nodata value."""
    task = PerPixelRegressionTask(
        scale_factor=0.1,
        nodata_value=-1,
    )
    metadata = _make_metadata(empty_sample_metadata, height=2, width=2)
    # We use 1x1x2x2 input with one invalid pixel and three different values.
    _, target_dict = task.process_inputs(
        raw_inputs={
            "targets": RasterImage(torch.tensor([[[[1, 2], [-1, 3]]]])),
        },
        metadata=metadata,
    )
    values = target_dict["values"].get_hw_tensor()
    valid = target_dict["valid"].get_hw_tensor()
    assert values.shape == (2, 2)
    assert values[0, 0] == pytest.approx(0.1)
    assert values[0, 1] == pytest.approx(0.2)
    assert values[1, 1] == pytest.approx(0.3)
    assert torch.all(valid == torch.tensor([[1, 1], [0, 1]]))


def test_process_output(empty_sample_metadata: SampleMetadata) -> None:
    """Ensure that PerPixelRegressionTask.process_output works."""
    scale_factor = 0.1
    task = PerPixelRegressionTask(
        scale_factor=scale_factor,
    )
    output = task.process_output(
        raw_output=torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32),
        metadata=empty_sample_metadata,
    )
    assert isinstance(output, np.ndarray)
    assert output.shape == (1, 2, 2)
    assert torch.all(output == torch.tensor([[[1, 2], [3, 4]]]))


def test_head(empty_sample_metadata: SampleMetadata) -> None:
    """Verify that the head masks out invalid pixels."""
    head = PerPixelRegressionHead(loss_mode="mse")
    logits = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32)[None, None, :, :]
    target_dict = {
        "values": RasterImage(
            torch.tensor([[[[2, 2], [2, 4]]]], dtype=torch.float32), timestamps=None
        ),
        "valid": RasterImage(
            torch.tensor([[[[1, 1], [1, 0]]]], dtype=torch.float32), timestamps=None
        ),
    }
    output = head(
        intermediates=FeatureMaps([logits]),
        # Use dummy context since it is anyway not used.
        context=ModelContext(inputs=[], metadatas=[]),
        targets=[target_dict],
    )
    assert output.loss_dict["regress"] == 1


@pytest.mark.parametrize(
    ("loss_mode", "expected"),
    [
        ("mse", 1.0),
        ("l1", 1.0),
        ("huber", 0.5),
    ],
)
def test_head_loss_modes(
    empty_sample_metadata: SampleMetadata,
    loss_mode: Literal["mse", "l1", "huber"],
    expected: float,
) -> None:
    head = PerPixelRegressionHead(loss_mode=loss_mode)
    logits = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32)[None, None, :, :]
    target_dict = {
        "values": RasterImage(
            torch.tensor([[[[2, 2], [2, 4]]]], dtype=torch.float32), timestamps=None
        ),
        "valid": RasterImage(
            torch.tensor([[[[1, 1], [1, 0]]]], dtype=torch.float32), timestamps=None
        ),
    }
    output = head(
        intermediates=FeatureMaps([logits]),
        context=ModelContext(inputs=[], metadatas=[]),
        targets=[target_dict],
    )
    assert output.loss_dict["regress"] == pytest.approx(expected)


def test_mse_metric(empty_sample_metadata: SampleMetadata) -> None:
    """Verify mean squared error metric works with customized scale_factor."""
    task = PerPixelRegressionTask(
        scale_factor=0.1,
        metrics=("mse",),
        nodata_value=-1,
    )
    metrics = task.get_metrics()

    # Prepare example.
    metadata = _make_metadata(empty_sample_metadata, height=2, width=2)
    _, target_dict = task.process_inputs(
        raw_inputs={
            "targets": RasterImage(torch.tensor([[[[1, 2], [-1, 3]]]])),
        },
        metadata=metadata,
    )
    preds = torch.tensor([[0.1, 0.1], [0.1, 0.1]])[None, None, :, :]

    # Accuracy should be (0 + 0.01 + 0.04) / 3 = 0.05 / 3.
    metrics.update(preds, [target_dict])
    results = metrics.compute()
    assert results["mse"] == pytest.approx(0.05 / 3)


def test_rmse_metric(empty_sample_metadata: SampleMetadata) -> None:
    """Verify root mean squared error metric works with customized scale_factor."""
    task = PerPixelRegressionTask(
        scale_factor=0.1,
        metrics=("rmse",),
        nodata_value=-1,
    )
    metrics = task.get_metrics()

    metadata = _make_metadata(empty_sample_metadata, height=2, width=2)
    _, target_dict = task.process_inputs(
        raw_inputs={
            "targets": RasterImage(torch.tensor([[[[1, 2], [-1, 3]]]])),
        },
        metadata=metadata,
    )
    preds = torch.tensor([[0.1, 0.1], [0.1, 0.1]])[None, None, :, :]

    metrics.update(preds, [target_dict])
    results = metrics.compute()
    assert results["rmse"] == pytest.approx(np.sqrt(0.05 / 3))


def test_mape_metric(empty_sample_metadata: SampleMetadata) -> None:
    """Verify mean absolute percentage error metric works."""
    task = PerPixelRegressionTask(
        scale_factor=0.1,
        metrics=("mape",),
        nodata_value=-1,
    )
    metrics = task.get_metrics()

    metadata = _make_metadata(empty_sample_metadata, height=2, width=2)
    _, target_dict = task.process_inputs(
        raw_inputs={
            "targets": RasterImage(torch.tensor([[[[1, 2], [-1, 3]]]])),
        },
        metadata=metadata,
    )
    preds = torch.tensor([[0.1, 0.1], [0.1, 0.1]])[None, None, :, :]

    metrics.update(preds, [target_dict])
    results = metrics.compute()
    # Labels are [0.1, 0.2, 0.3], preds are [0.1, 0.1, 0.1] over valid pixels.
    # MAPE = mean(abs(pred-label)/abs(label)) = (0 + 0.5 + 2/3) / 3 = 7/18.
    assert results["mape"] == pytest.approx(7 / 18)


def test_r2_metric_list(empty_sample_metadata: SampleMetadata) -> None:
    task = PerPixelRegressionTask(
        scale_factor=0.1,
        metrics=("mse", "r2"),
        nodata_value=-1,
    )
    metrics = task.get_metrics()

    metadata = _make_metadata(empty_sample_metadata, height=2, width=2)
    _, target_dict = task.process_inputs(
        raw_inputs={
            "targets": RasterImage(torch.tensor([[[[1, 2], [-1, 3]]]])),
        },
        metadata=metadata,
    )
    preds = torch.tensor([[0.1, 0.1], [0.1, 0.1]])[None, None, :, :]

    metrics.update(preds, [target_dict])
    results = metrics.compute()

    assert results["mse"] == pytest.approx(0.05 / 3)
    # Labels are [0.1, 0.2, 0.3], preds are [0.1, 0.1, 0.1] over valid pixels.
    # SSE = 0.05, SST = 0.02 => R2 = 1 - 0.05/0.02 = -1.5
    assert results["r2"] == pytest.approx(-1.5)


def test_process_inputs_masks_out_of_window_padding(
    empty_sample_metadata: SampleMetadata,
) -> None:
    """Ensure padded (out-of-window) pixels are marked invalid."""
    window_bounds = (0, 0, 10, 10)
    crop_bounds = (-5, -5, 15, 15)

    decoded = torch.zeros((20, 20), dtype=torch.float32)
    decoded[5:15, 5:15] = 5.0

    raw_inputs: dict[str, RasterImage | list[Feature]] = {
        "targets": RasterImage(decoded[None, None, :, :], timestamps=None),
    }
    metadata = _make_metadata(
        empty_sample_metadata,
        height=20,
        width=20,
        window_bounds=window_bounds,
        crop_bounds=crop_bounds,
        window_group="g",
        window_name="w",
    )

    task = PerPixelRegressionTask(nodata_value=-1.0)
    _, target_dict = task.process_inputs(
        raw_inputs, metadata=metadata, load_targets=True
    )

    valid = target_dict["valid"].get_hw_tensor()
    assert int(valid.sum().item()) == 10 * 10
    assert valid[0, 0] == 0
    assert valid[5, 5] == 1
