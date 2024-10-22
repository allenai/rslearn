import os
import pathlib
import random
import tempfile

import fsspec
from upath import UPath

from rslearn.utils.rtree_index import RtreeIndex, get_cached_rtree


def test_remote_cache(tmp_path: pathlib.Path):
    """Test that we can get the cached rtree index when it's on a remote filesystem."""
    test_id = random.randint(10000, 99999)
    prefix = f"test_{test_id}/"
    fake_gcs = fsspec.filesystem("memory")
    fake_gcs.mkdirs(prefix, exist_ok=True)
    cache_dir = UPath(f"memory://bucket/{prefix}", fs=fake_gcs)

    # Build rtree with one point.
    box = [0, 0, 1, 1]

    def build_rtree1(index: RtreeIndex):
        index.insert(box, "a")

    index = get_cached_rtree(cache_dir, tmp_path, build_rtree1)
    result = index.query(box)
    assert len(result) == 1 and result[0] == "a"

    # Now make sure it is using the cached version.
    os.unlink(os.path.join(tmp_path, "rtree_index.dat"))
    os.unlink(os.path.join(tmp_path, "rtree_index.idx"))

    def build_rtree2(index: RtreeIndex):
        index.insert(box, "b")

    index = get_cached_rtree(cache_dir, tmp_path, build_rtree1)
    result = index.query(box)
    assert len(result) == 1 and result[0] == "a"


def test_local_cache(tmp_path: pathlib.Path):
    """Test that we can get the cached rtree index when it's on a local filesystem."""
    # Build rtree with one point.
    box = [0, 0, 1, 1]

    def build_rtree1(index: RtreeIndex):
        index.insert(box, "a")

    with tempfile.TemporaryDirectory() as cache_dir:
        cached_dir_upath = UPath(cache_dir)
        index = get_cached_rtree(cached_dir_upath, tmp_path, build_rtree1)
        result = index.query(box)
        assert len(result) == 1 and result[0] == "a"
