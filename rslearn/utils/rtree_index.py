"""RtreeIndex spatial index implementation."""

import os
import shutil
from collections.abc import Callable
from typing import Any

import fsspec
from rtree import index
from upath import UPath

from rslearn.utils.spatial_index import SpatialIndex


class RtreeIndex(SpatialIndex):
    """An index of spatiotemporal geometries backed by an rtree index.

    Both in-memory and on-disk options are supported.
    """

    def __init__(self, fname: str | None = None):
        """Initialize a new RtreeIndex.

        If fname is set, the index is persisted on disk, otherwise it is in-memory.

        For on-disk index, optionally use is_done and mark_done to reuse an existing
        index at the same filename. For reuse, if mark_done was called previously, then
        the existing index will be reused. Use is_done to check if mark_done was
        previously called, and use mark_done to indicate the index is done writing (a
        .done marker file will be created).

        Otherwise, if mark_done was never called previously, then the index is always
        overwritten.

        Args:
            fname: the filename to store the index in, or None to create an in-memory
                index
        """
        self.fname = fname
        self.index = index.Index(fname)
        self.counter = 0

    def insert(self, box: tuple[float, float, float, float], data: Any) -> None:
        """Insert a box into the index.

        Args:
            box: the bounding box of this item (minx, miny, maxx, maxy)
            data: arbitrary object
        """
        self.counter += 1
        self.index.insert(id=self.counter, coordinates=box, obj=data)

    # TODO: Make a named tuple for all the bounding box stuff
    def query(self, box: tuple[float, float, float, float]) -> list[Any]:
        """Query the index for objects intersecting a box.

        Args:
            box: the bounding box query (minx, miny, maxx, maxy)

        Returns:
            a list of objects in the index intersecting the box
        """
        results = self.index.intersection(box, objects=True)
        return [r.object for r in results]


def delete_partially_created_local_files(fname: str):
    """Delete partially created .dat and .idx files."""
    extensions = [".dat", ".idx"]
    for ext in extensions:
        cur_fname = fname + ext
        if os.path.exists(cur_fname):
            os.unlink(cur_fname)


def copy_cache_to_local_dir(cache_dir: UPath, tmp_dir: str, extension_suffix: str):
    """Copy the rtree index files from cache_dir to a local temporary directory."""
    extensions = [".dat", ".idx"]
    for ext in extensions:
        with (cache_dir / f"rtree_index{ext}").open("rb") as src:
            with open(os.path.join(tmp_dir, f"rtree_index{ext}"), "wb") as dst:
                shutil.copyfileobj(src, dst)


def get_cached_rtree(
    cache_dir: UPath, tmp_dir: str, build_fn: Callable[[RtreeIndex], None]
) -> RtreeIndex:
    """Returns an RtreeIndex cached in cache_dir, creating it if needed.

    The .dat and .idx files are cached in cache_dir. Since RtreeIndex expects local
    filesystem, it is copied to a local temporary directory if needed. If the index
    doesn't exist yet, then it is created using build_fn.

    Args:
        cache_dir: directory to cache the index files.
        tmp_dir: temporary local directory to use in case cache_dir is on a remote
            filesystem. The caller is responsible for cleaning this up when they don't
            need the index anymore.
        build_fn: function to build the index in case it doesn't exist yet.

    Returns:
        the RtreeIndex.
    """

    is_local_cache = isinstance(
        cache_dir.fs, fsspec.implementations.local.LocalFileSystem
    )
    extensions = [".dat", ".idx"]
    completed_fname = cache_dir / "rtree_index.done"

    if not completed_fname.exists():
        # Need to build the rtree index.
        # After building, if the cache is non-local, we additionally need to copy it to
        # the cache_dir.
        if is_local_cache:
            local_fname = (cache_dir / "rtree_index").path
        else:
            local_fname = os.path.join(tmp_dir, "rtree_index")
        print(f"{local_fname=}")
        delete_partially_created_local_files(local_fname)

        rtree_index = RtreeIndex(local_fname)
        build_fn(rtree_index)
        del rtree_index

        if not is_local_cache:
            for ext in extensions:
                with open(local_fname + ext, "rb") as src:
                    with (cache_dir / f"rtree_index{ext}").open("wb") as dst:
                        shutil.copyfileobj(src, dst)

        # Create the completed file to indicate index is ready in cache.
        completed_fname.touch()

    else:
        # Initialize the index from the cached version.
        # If the cache is non-local, we need to retrieve it first.
        if is_local_cache:
            local_fname = (cache_dir / "rtree_index").path
        else:
            local_fname = os.path.join(tmp_dir, "rtree_index")
            for ext in extensions:
                with (cache_dir / f"rtree_index{ext}").open("rb") as src:
                    with open(local_fname + ext, "wb") as dst:
                        shutil.copyfileobj(src, dst)

    return RtreeIndex(local_fname)
