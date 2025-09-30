"""Callbacks for integrating with Weights & Biases."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from lightning.pytorch.cli import SaveConfigCallback

try:  # pragma: no cover - optional dependency at import time
    from lightning.pytorch.loggers import WandbLogger
except Exception:  # pragma: no cover - lightning may not expose WandbLogger
    WandbLogger = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import wandb
except Exception:  # pragma: no cover - wandb is optional in tests
    wandb = None  # type: ignore

from rslearn.log_utils import get_logger

logger = get_logger(__name__)


class RslearnSaveConfigCallback(SaveConfigCallback):
    """Save the CLI config and upload it to Weights & Biases when available."""

    def __init__(
        self,
        parser,
        config,
        config_filename: str = "config.yaml",
        overwrite: bool = False,
        multifile: bool = False,
        save_to_log_dir: bool = True,
        artifact_name: str = "training-config",
        artifact_type: str = "config",
    ) -> None:
        super().__init__(
            parser=parser,
            config=config,
            config_filename=config_filename,
            overwrite=overwrite,
            multifile=multifile,
            save_to_log_dir=save_to_log_dir,
        )
        self.artifact_name = artifact_name
        self.artifact_type = artifact_type

    def save_config(self, trainer, pl_module, stage: str) -> None:  # pragma: no cover - integration point
        super().save_config(trainer, pl_module, stage)

        if WandbLogger is None:
            return

        run = self._get_active_wandb_run(trainer)
        if run is None:
            return

        config_path = self._get_config_path(trainer)
        if config_path is None:
            logger.debug("Skipping W&B config upload; config path unknown.")
            return
        if not config_path.exists():
            logger.debug("Skipping W&B config upload; config not found at %s", config_path)
            return

        try:
            if wandb is not None:
                artifact = wandb.Artifact(self.artifact_name, type=self.artifact_type)
                artifact.add_file(str(config_path))
                run.log_artifact(artifact)
            else:
                run.save(str(config_path), policy="now")
        except Exception as exc:  # pragma: no cover - external library failure
            logger.warning("Failed to upload config file to W&B: %s", exc)
            return

        logger.info("Logged training config %s to W&B", config_path)

    def _get_config_path(self, trainer) -> Path | None:
        log_dir = getattr(trainer, "log_dir", None)
        if log_dir is None:
            return None
        return Path(log_dir) / self.config_filename

    def _get_active_wandb_run(self, trainer):
        loggers: Iterable | None = getattr(trainer, "loggers", None)
        if loggers is None:
            single_logger = getattr(trainer, "logger", None)
            loggers = [single_logger] if single_logger is not None else []

        if not isinstance(loggers, (list, tuple)):
            loggers = [loggers]

        for logger_instance in loggers:
            if WandbLogger is not None and isinstance(logger_instance, WandbLogger):
                run = getattr(logger_instance, "experiment", None)
                if run is not None:
                    return run
        return None
