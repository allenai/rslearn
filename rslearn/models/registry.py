"""Model registry."""

from collections.abc import Callable
from typing import Any, TypeVar

Models: dict[str, type[Any]] = {}

_ModelT = TypeVar("_ModelT")


def register_model(name: str) -> Callable[[type[_ModelT]], type[_ModelT]]:
    """Decorator to register a model class."""

    def decorator(cls: type[_ModelT]) -> type[_ModelT]:
        Models[name] = cls
        return cls

    return decorator
