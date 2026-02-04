"""Utilities related to fsspec and upath libraries."""

import os
import tempfile
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from typing import Any

import rasterio
import rasterio.io
from fsspec.implementations.local import LocalFileSystem
from upath import UPath

from rslearn.log_utils import get_logger

logger = get_logger(__name__)


def iter_nonhidden(path: UPath) -> Iterator[UPath]:
    """Iterate over non-hidden entries in a directory.

    Hidden entries are those whose basename starts with "." (e.g. ".DS_Store").

    Args:
        path: the directory to iterate.

    Yields:
        non-hidden UPath entries in the directory.
    """
    try:
        it = path.iterdir()
    except (FileNotFoundError, NotADirectoryError):
        return

    for p in it:
        if p.name.startswith("."):
            continue
        yield p


def iter_nonhidden_subdirs(path: UPath) -> Iterator[UPath]:
    """Iterate over non-hidden subdirectories in a directory.

    Args:
        path: the directory to iterate.

    Yields:
        non-hidden subdirectories in the directory.
    """
    for p in iter_nonhidden(path):
        try:
            if p.is_dir():
                yield p
        except OSError:
            continue


def iter_nonhidden_files(path: UPath) -> Iterator[UPath]:
    """Iterate over non-hidden files in a directory.

    Args:
        path: the directory to iterate.

    Yields:
        non-hidden files in the directory.
    """
    for p in iter_nonhidden(path):
        try:
            if p.is_file():
                yield p
        except OSError:
            continue


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
                if not extra_path.exists():
                    continue
                extra_basename = os.path.basename(extra_path.name)
                extra_local_fname = os.path.join(dir_name, extra_basename)
                extra_path.fs.get(extra_path.path, extra_local_fname)

            yield local_fname


def join_upath(path: UPath, suffix: str) -> UPath:
    """Joins a UPath with a suffix that may be absolute or relative to the path.

    Args:
        path: the parent path
        suffix: string suffix. It can include a protocol, in which it is treated as an
            absolute path not relative to the parent. It can also be a
            filesystem-specific absolute path, or a path relative to the parent.

    Returns:
        the joined path
    """
    if "://" in suffix:
        return UPath(suffix)
    else:
        return path / suffix


@contextmanager
def open_atomic(path: UPath, *args: Any, **kwargs: Any) -> Generator[Any, None, None]:
    """Open a path for atomic writing.

    If it is local filesystem, we will write to a temporary file, and rename it to the
    destination upon success.

    Otherwise, we assume it's object storage and none of that is needed.

    Args:
        path: the UPath to be opened
        *args: any valid arguments for :code:`open`
        **kwargs: any valid keyword arguments for :code:`open`
    """
    if isinstance(path.fs, LocalFileSystem):
        logger.debug("open_atomic: writing atomically to local file at %s", path)
        tmppath = path.path + ".tmp." + str(os.getpid())
        with open(tmppath, *args, **kwargs) as file:
            yield file
        os.rename(tmppath, path.path)

    else:
        logger.debug("open_atomic: writing to remote file at %s", path)
        with path.open(*args, **kwargs) as file:
            yield file


@contextmanager
def open_rasterio_upath_reader(
    path: UPath, **kwargs: Any
) -> Generator[rasterio.io.DatasetReader, None, None]:
    """Open a raster for reading.

    If the UPath is local, then we open with rasterio directly, since this is much
    faster. Otherwise, we open the file stream first and then use rasterio with file
    stream.

    Args:
        path: the path to read.
        **kwargs: additional keyword arguments for :code:`rasterio.open`
    """
    if isinstance(path.fs, LocalFileSystem):
        logger.debug("reading from local rasterio dataset at %s", path)
        with rasterio.open(path.path, **kwargs) as raster:
            yield raster

    else:
        logger.debug("reading from remote rasterio dataset at %s", path)
        with path.open("rb") as f:
            with rasterio.open(f, **kwargs) as raster:
                yield raster


@contextmanager
def open_rasterio_upath_writer(
    path: UPath, **kwargs: Any
) -> Generator[rasterio.io.DatasetWriter, None, None]:
    """Open a raster for writing.

    If the UPath is local, then we open with rasterio directly, since this is much
    faster. We also write atomically by writing to temporary file and then renaming it,
    to avoid concurrency issues. Otherwise, we open the file stream first and then use
    rasterio with file stream (and assume that it is object storage so the write will
    be atomic).

    Args:
        path: the path to write.
        **kwargs: additional keyword arguments for :code:`rasterio.open`
    """
    if isinstance(path.fs, LocalFileSystem):
        logger.debug(
            "open_rasterio_upath_writer: writing atomically to local rasterio dataset at %s",
            path,
        )
        tmppath = path.path + ".tmp." + str(os.getpid())
        with rasterio.open(tmppath, "w", **kwargs) as raster:
            yield raster
        os.rename(tmppath, path.path)

    else:
        logger.debug(
            "open_rasterio_upath_writer: writing to remote rasterio dataset at %s", path
        )
        with path.open("wb") as f:
            with rasterio.open(f, "w", **kwargs) as raster:
                yield raster


def get_relative_suffix(base_dir: UPath, fname: UPath) -> str:
    """Get the suffix of fname relative to base_dir.

    Args:
        base_dir: the base directory.
        fname: a filename within the base directory.

    Returns:
        the suffix on base_dir that would yield the given filename.
    """
    if not fname.path.startswith(base_dir.path):
        raise ValueError(
            f"filename {fname.path} must start with base directory {base_dir.path}"
        )
    suffix = fname.path[len(base_dir.path) :]
    if suffix.startswith("/"):
        suffix = suffix[1:]
    return suffix
