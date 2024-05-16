"""Miscellaneous utility functions."""

import os
from contextlib import contextmanager
from typing import Any


@contextmanager
def open_atomic(filepath: str, *args: list[Any], **kwargs: dict[str, Any]):
    """Open a file for atomic writing.

    Will write to a temporary file, and rename it to the destination upon success.

    Args:
        filepath: the file path to be opened
        *args: ny valid arguments for :code:`open`
        **kwargs: any valid keyword arguments for :code:`open`
    """
    tmppath = filepath + ".tmp." + str(os.getpid())
    with open(tmppath, *args, **kwargs) as file:
        yield file
    os.rename(tmppath, filepath)
