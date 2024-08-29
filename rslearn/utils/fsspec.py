"""Utilities related to fsspec and upath libraries."""

import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager

from fsspec.implementations.local import LocalFileSystem
from upath import UPath


@contextmanager
def get_upath_local(
    path: UPath, extra_paths: list[UPath] = []
) -> Generator[str, None, None]:
    """Returns a local filename to access the specified UPath.

    If the path is already local, then its string representation is returned.

    Args:
        path: the UPath to open
        extra_paths: any additional files that should be copied to the same directory
            as the specified path. They will only be copied if the filesystem is not
            local.

    Returns:
        the local filename at which the file can be accessed in this context manager
    """
    if isinstance(path.fs, LocalFileSystem):
        yield path.path

    else:
        with tempfile.TemporaryDirectory() as dir_name:
            basename = os.path.basename(path.name)
            local_fname = os.path.join(dir_name, basename)
            path.fs.get(path.path, local_fname)

            for extra_path in extra_paths:
                extra_basename = os.path.basename(extra_path.name)
                extra_local_fname = os.path.join(dir_name, extra_basename)
                extra_path.fs.get(extra_path.path, extra_local_fname)

            yield local_fname
