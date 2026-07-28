"""Unit tests for rslearn.lightning_cli."""

import pathlib
from typing import Any

import pytest

import rslearn.lightning_cli as lightning_cli_module
from rslearn.lightning_cli import (
    MLFLOW_ID_FNAME,
    RslearnSaveConfigCallback,
    SaveMLflowRunIdCallback,
)


def test_save_config_uses_default_root_dir(tmp_path: pathlib.Path) -> None:
    """Config is saved under project management rather than the logger directory."""

    class FakeParser:
        def save(
            self,
            config: object,
            path: str,
            **kwargs: object,
        ) -> None:
            pathlib.Path(path).write_text("config")

    class FakeStrategy:
        def broadcast(self, value: object) -> object:
            return value

    trainer = type(
        "Trainer",
        (),
        {
            "log_dir": str(tmp_path / "logger"),
            "default_root_dir": str(tmp_path),
            "is_global_zero": True,
            "strategy": FakeStrategy(),
        },
    )()
    callback = RslearnSaveConfigCallback(
        FakeParser(),  # type: ignore[arg-type]
        config={},
        overwrite=True,
    )

    callback.setup(trainer, pl_module=None, stage="fit")  # type: ignore[arg-type]

    assert (tmp_path / "config.yaml").read_text() == "config"
    assert not (tmp_path / "logger" / "config.yaml").exists()
    assert callback.already_saved


def test_save_config_supports_remote_default_root_dir() -> None:
    """Config can be saved to a project management directory through fsspec."""

    class FakeStrategy:
        def broadcast(self, value: object) -> object:
            return value

    root_dir = "memory://rslearn-tests/project"
    trainer = type(
        "Trainer",
        (),
        {
            "log_dir": ".",
            "default_root_dir": root_dir,
            "is_global_zero": True,
            "strategy": FakeStrategy(),
        },
    )()
    callback = RslearnSaveConfigCallback(
        lightning_cli_module.LightningArgumentParser(),
        config=lightning_cli_module.jsonargparse.Namespace(),
        overwrite=True,
    )

    callback.setup(trainer, pl_module=None, stage="fit")  # type: ignore[arg-type]

    fs = lightning_cli_module.fsspec.filesystem("memory")
    assert fs.isfile("rslearn-tests/project/config.yaml")


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
