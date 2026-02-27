"""Compatibility helpers for experiment tracking backends.

rslearn primarily supports logging through the Weights & Biases (wandb) API. This
module provides a small shim so that `trackio` can be used as a lightweight, local
drop-in replacement for the subset of the wandb API that rslearn relies on.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def is_wandb_available() -> bool:
    return _has_module("wandb")


def is_trackio_available() -> bool:
    return _has_module("trackio")


@dataclass(frozen=True)
class _TrackioRunShim:
    """Expose a minimal `wandb.run`-like interface for trackio runs."""

    run: Any

    @property
    def id(self) -> str:
        # trackio identifies runs by name (string). Use it as an id for resume flows.
        return str(getattr(self.run, "name", ""))

    @property
    def name(self) -> str:
        return str(getattr(self.run, "name", ""))


def _trackio_current_run(trackio_mod: Any) -> Any | None:
    try:
        return trackio_mod.context_vars.current_run.get()
    except Exception:
        return None


def _default_confusion_matrix_plot(**kwargs: Any) -> dict[str, Any]:
    # Minimal placeholder that remains JSON-serializable across backends.
    return {"type": "confusion_matrix", **kwargs}


class _WandbLike:
    """A tiny facade over wandb/trackio for the rslearn call sites."""

    def __init__(self, backend: Any | None) -> None:
        self._backend = backend
        backend_plot = getattr(backend, "plot", None) if backend is not None else None
        if backend_plot is not None and hasattr(backend_plot, "confusion_matrix"):
            self.plot = backend_plot
        else:
            self.plot = SimpleNamespace(confusion_matrix=_default_confusion_matrix_plot)

    @property
    def backend_name(self) -> str:
        if self._backend is None:
            return "none"
        return getattr(self._backend, "__name__", type(self._backend).__name__)

    @property
    def run(self) -> Any | None:
        if self._backend is None:
            return None

        # Native wandb exposes `wandb.run` as the active run object.
        if getattr(self._backend, "__name__", None) == "wandb":
            return getattr(self._backend, "run", None)

        # trackio keeps the active run in a context var.
        if getattr(self._backend, "__name__", None) == "trackio":
            run = _trackio_current_run(self._backend)
            return _TrackioRunShim(run) if run is not None else None

        return None

    @property
    def config(self) -> Any:
        if self._backend is None:
            return {}
        return getattr(self._backend, "config", {})

    def init(self, *args: Any, **kwargs: Any) -> Any:
        if self._backend is None:
            raise RuntimeError(
                "No tracking backend installed. Install `wandb` or `trackio`."
            )
        return self._backend.init(*args, **kwargs)

    def log(self, *args: Any, **kwargs: Any) -> Any:
        if self._backend is None:
            return None
        return self._backend.log(*args, **kwargs)

    def finish(self, *args: Any, **kwargs: Any) -> Any:
        if self._backend is None:
            return None
        finish: Callable[..., Any] | None = getattr(self._backend, "finish", None)
        if finish is None:
            return None
        return finish(*args, **kwargs)


def _select_backend() -> Any | None:
    preference = os.environ.get("RSLEARN_WANDB_BACKEND", "auto").strip().lower()

    if preference not in {"auto", "wandb", "trackio"}:
        preference = "auto"

    if preference in {"auto", "wandb"} and is_wandb_available():
        import wandb as backend  # type: ignore[import-not-found]

        return backend

    if preference in {"auto", "trackio"} and is_trackio_available():
        import trackio as backend  # type: ignore[import-not-found]

        return backend

    return None


wandb = _WandbLike(_select_backend())
