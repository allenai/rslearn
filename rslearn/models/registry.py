"""Model registry."""

from collections.abc import Callable
from typing import Any, TypeVar

_ModelT = TypeVar("_ModelT")


class _ModelRegistry(dict[str, type[Any]]):
    """Registry for Model classes."""

    def register(self, name: str) -> Callable[[type[_ModelT]], type[_ModelT]]:
        """Decorator to register a model class."""

        def decorator(cls: type[_ModelT]) -> type[_ModelT]:
            self[name] = cls
            return cls

        return decorator


Models = _ModelRegistry()
