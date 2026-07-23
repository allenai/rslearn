"""Unit tests for rslearn.lightning_cli."""

import pathlib
from typing import Any

import pytest

import rslearn.lightning_cli as lightning_cli_module
from rslearn.lightning_cli import MLFLOW_ID_FNAME, SaveMLflowRunIdCallback


@pytest.mark.parametrize(
    ("saved_run_id", "expected_log_count"),
    [(None, 1), ("run-id", 0), ("different-run-id", 1)],
)
def test_mlflow_config_is_not_reuploaded_on_resume(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    saved_run_id: str | None,
    expected_log_count: int,
) -> None:
    """Only skip the config upload when resuming the same MLflow run."""
    run_id_path = tmp_path / MLFLOW_ID_FNAME
    if saved_run_id is not None:
        run_id_path.write_text(saved_run_id)

    logged_configs: list[dict[str, Any]] = []

    class FakeExperiment:
        def log_dict(self, **kwargs: Any) -> None:
            logged_configs.append(kwargs)

    class FakeMLFlowLogger:
        run_id = "run-id"
        experiment = FakeExperiment()

    monkeypatch.setattr(lightning_cli_module, "MLFlowLogger", FakeMLFlowLogger)
    trainer = type("Trainer", (), {"logger": FakeMLFlowLogger()})()
    callback = SaveMLflowRunIdCallback(str(tmp_path), '{"project_name": "project"}')

    callback.on_fit_start(trainer, pl_module=None)  # type: ignore[arg-type]

    assert len(logged_configs) == expected_log_count
    assert run_id_path.read_text() == "run-id"
