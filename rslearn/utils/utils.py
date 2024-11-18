"""Miscellaneous utility functions."""

import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any


@contextmanager  # type: ignore
def open_atomic(
    filepath: str, *args: list[Any], **kwargs: dict[str, Any]
) -> Generator[Any, None, None]:
    """Open a file for atomic writing.

    Will write to a temporary file, and rename it to the destination upon success.

    Args:
        filepath: the file path to be opened
        *args: ny valid arguments for :code:`open`
        **kwargs: any valid keyword arguments for :code:`open`
    """
    tmppath = filepath + ".tmp." + str(os.getpid())
    with open(tmppath, *args, **kwargs) as file:  # type: ignore
        yield file
    os.rename(tmppath, filepath)


def parse_disabled_layers(disabled_layers: str) -> list[str]:
    """Parse the disabled layers string."""
    return disabled_layers.split(",") if disabled_layers else []
