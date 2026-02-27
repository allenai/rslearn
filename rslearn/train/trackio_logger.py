"""Lightning logger for Trackio.

Trackio is a lightweight, local drop-in replacement for a subset of the wandb API.
This logger enables Lightning scalar metric logging through Trackio.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment


class TrackioLogger(Logger):
    """A minimal Lightning logger that logs to Trackio."""

    def __init__(
        self,
        project: str,
        name: str | None = None,
        group: str | None = None,
        config: dict[str, Any] | None = None,
        id: str | None = None,
        resume: str = "auto",
        notes: str | None = None,
        save_dir: str | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self._project = project
        self._name = id or name
        self._group = group
        self._config = config or {}
        self._resume = resume
        self._experiment = None
        self._notes = notes
        self._save_dir = save_dir

    @property
    def name(self) -> str:
        return self._project

    @property
    def version(self) -> str:
        exp = self._experiment
        if exp is None:
            return self._name or ""
        return str(getattr(exp, "name", self._name or ""))

    @property
    @rank_zero_experiment
    def experiment(self) -> Any:
        if self._experiment is None:
            try:
                import trackio  # type: ignore[import-not-found]
            except ImportError as e:
                raise ImportError(
                    "TrackioLogger requires `trackio` to be installed."
                ) from e

            resume = self._resume
            if resume == "auto":
                resume = "allow" if self._name is not None else "never"

            self._experiment = trackio.init(
                project=self._project,
                name=self._name,
                group=self._group,
                config=self._config,
                resume=resume,
                embed=False,
            )
            if self._notes:
                # Trackio doesn't currently have a dedicated notes field; store it in config.
                try:
                    self._experiment.config.setdefault("_notes", self._notes)
                except Exception:
                    pass
        return self._experiment

    @property
    def save_dir(self) -> str | None:
        return self._save_dir

    def log_hyperparams(self, params: Any) -> None:
        exp = self.experiment
        if params is None:
            return
        if hasattr(params, "as_dict"):
            params = params.as_dict()
        if isinstance(params, Mapping):
            try:
                exp.config.update(dict(params))
            except Exception:
                pass

    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        import trackio  # type: ignore[import-not-found]

        filtered: dict[str, Any] = {}
        for k, v in metrics.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    v = v.item()
                else:
                    # Avoid logging large tensors by default.
                    continue
            filtered[k] = v

        if not filtered:
            return

        # Ensure run exists (trackio.log requires init first).
        _ = self.experiment
        trackio.log(filtered, step=step)

    def finalize(self, status: str) -> None:
        if self._experiment is None:
            return
        try:
            self._experiment.finish()
        except Exception:
            pass
