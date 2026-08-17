"""Tests for metrics module."""

import pytest
import torch

from rslearn.train.metrics import (
    ConfusionMatrixMetric,
    ConfusionMatrixOutput,
    NonScalarMetricOutput,
)


def test_unsupported_loggers_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that unsupported W&B and MLflow logging behave symmetrically."""
    import rslearn.train.metrics as metrics_module

    warnings: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        metrics_module.logger,
        "warning",
        lambda *args: warnings.append(args),
    )

    output = NonScalarMetricOutput()
    output.log_to_wandb("metric")
    output.log_to_mlflow("metric", client=object(), run_id="run-id")

    assert warnings == [
        ("W&B logging is not implemented for metric %s", "metric"),
        ("MLflow logging is not implemented for metric %s", "metric"),
    ]


class TestConfusionMatrixMetric:
    """Tests for ConfusionMatrixMetric."""

    def test_update_compute_reset(self) -> None:
        """Test update with two batches, compute, and reset."""
        metric = ConfusionMatrixMetric(num_classes=3, class_names=["a", "b", "c"])

        # Batch 1: 2 samples
        # Sample 1: true=0, pred=0 (correct)
        # Sample 2: true=1, pred=2 (wrong)
        preds1 = torch.tensor(
            [
                [0.8, 0.1, 0.1],  # Predicted class 0
                [0.1, 0.2, 0.7],  # Predicted class 2
            ]
        )
        labels1 = torch.tensor([0, 1])
        metric.update(preds1, labels1)

        # Batch 2: 2 samples
        # Sample 3: true=2, pred=2 (correct)
        # Sample 4: true=0, pred=1 (wrong)
        preds2 = torch.tensor(
            [
                [0.1, 0.1, 0.8],  # Predicted class 2
                [0.3, 0.5, 0.2],  # Predicted class 1
            ]
        )
        labels2 = torch.tensor([2, 0])
        metric.update(preds2, labels2)

        # Compute and verify confusion matrix
        output = metric.compute()

        assert isinstance(output, ConfusionMatrixOutput)
        assert output.class_names == ["a", "b", "c"]

        # Expected confusion matrix:
        # true\pred  0  1  2
        #    0      [1, 1, 0]
        #    1      [0, 0, 1]
        #    2      [0, 0, 1]
        expected_cm = torch.tensor(
            [
                [1, 1, 0],
                [0, 0, 1],
                [0, 0, 1],
            ],
            dtype=torch.long,
        )
        assert torch.equal(output.confusion_matrix, expected_cm)

        # Test reset clears the state
        metric.reset()
        output_after_reset = metric.compute()
        assert output_after_reset.confusion_matrix.sum() == 0


class TestConfusionMatrixOutput:
    """Tests for ConfusionMatrixOutput."""

    def test_expand_confusion_matrix(self) -> None:
        """Test expanding confusion matrix to (preds, labels) pairs."""
        # Create a confusion matrix:
        # true\pred  0  1
        #    0      [2, 1]
        #    1      [0, 3]
        cm = torch.tensor([[2, 1], [0, 3]], dtype=torch.long)
        output = ConfusionMatrixOutput(confusion_matrix=cm, class_names=["a", "b"])

        preds, labels = output._expand_confusion_matrix()

        # Should have 6 samples total
        assert len(preds) == 6
        assert len(labels) == 6

        assert preds == [0, 0, 1, 1, 1, 1]
        assert labels == [0, 0, 0, 1, 1, 1]

    def test_log_to_wandb(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test log_to_wandb calls wandb with correct arguments."""
        import rslearn.train.metrics as metrics_module

        # Track calls to wandb
        logged_data: dict[str, str] = {}
        cm_call_args: dict[str, list[int] | list[str] | str] = {}

        def mock_log(data: dict[str, str]) -> None:
            logged_data.update(data)

        def mock_confusion_matrix(**kwargs: list[int] | list[str] | str) -> str:
            cm_call_args.update(kwargs)
            return "mock_plot"

        monkeypatch.setattr(metrics_module.wandb, "log", mock_log)
        monkeypatch.setattr(
            metrics_module.wandb.plot, "confusion_matrix", mock_confusion_matrix
        )

        # Create and log confusion matrix
        cm = torch.tensor([[2, 1], [0, 3]], dtype=torch.long)
        output = ConfusionMatrixOutput(confusion_matrix=cm, class_names=["a", "b"])
        output.log_to_wandb("val_confusion_matrix")

        # Verify wandb.log was called with correct key
        assert "val_confusion_matrix" in logged_data
        assert logged_data["val_confusion_matrix"] == "mock_plot"

        # Verify wandb.plot.confusion_matrix was called with correct args
        assert cm_call_args["class_names"] == ["a", "b"]
        assert cm_call_args["title"] == "val_confusion_matrix"
        assert len(cm_call_args["preds"]) == 6
        assert len(cm_call_args["y_true"]) == 6

    def test_log_to_mlflow(self) -> None:
        """Test that MLflow receives confusion-matrix table and figure artifacts."""

        class MockClient:
            def __init__(self) -> None:
                self.table_kwargs: dict = {}
                self.figure_kwargs: dict = {}

            def log_table(self, **kwargs: object) -> None:
                self.table_kwargs = kwargs

            def log_figure(self, **kwargs: object) -> None:
                self.figure_kwargs = kwargs

        client = MockClient()
        output = ConfusionMatrixOutput(
            confusion_matrix=torch.tensor([[2, 1], [0, 3]], dtype=torch.long),
            class_names=["a", "b"],
        )
        output.log_to_mlflow("val_confusion_matrix", client, "run-123")

        assert client.table_kwargs["run_id"] == "run-123"
        assert client.table_kwargs["artifact_file"] == "val_confusion_matrix.json"
        assert client.table_kwargs["data"] == {
            "true_class": ["a", "a", "b", "b"],
            "predicted_class": ["a", "b", "a", "b"],
            "count": [2, 1, 0, 3],
        }
        assert client.figure_kwargs["run_id"] == "run-123"
        assert client.figure_kwargs["artifact_file"] == "val_confusion_matrix.png"
        figure = client.figure_kwargs["figure"]
        assert figure.axes[0].get_title() == "val_confusion_matrix"
        assert [label.get_text() for label in figure.axes[0].get_xticklabels()] == [
            "a",
            "b",
        ]
