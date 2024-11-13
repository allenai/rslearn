import os
import random
import tempfile

import fsspec
from upath import UPath

from rslearn.utils.rtree_index import (
    RtreeIndex,
    _get_tmp_dir_for_cached_rtree,
    get_cached_rtree,
)


def test_remote_cache() -> None:
    """Test that we can get the cached rtree index when it's on a remote filesystem."""
    test_id = random.randint(10000, 99999)
    prefix = f"test_{test_id}/"
    fake_gcs = fsspec.filesystem("memory")
    fake_gcs.mkdirs(prefix, exist_ok=True)
    cache_dir = UPath(f"memory://bucket/{prefix}", fs=fake_gcs)

    # Build rtree with one point.
    box: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)

    def build_rtree1(index: RtreeIndex) -> None:
        index.insert(box, "a")

    index = get_cached_rtree(cache_dir, build_rtree1)
    result = index.query(box)
    assert len(result) == 1 and result[0] == "a"

    # Now make sure it is using the cached version.
    local_tmp_dir = _get_tmp_dir_for_cached_rtree(cache_dir)
    os.unlink(os.path.join(local_tmp_dir, "rtree_index.dat"))
    os.unlink(os.path.join(local_tmp_dir, "rtree_index.idx"))

    index = get_cached_rtree(cache_dir, build_rtree1)
    result = index.query(box)
    assert len(result) == 1 and result[0] == "a"


def test_local_cache() -> None:
    """Test that we can get the cached rtree index when it's on a local filesystem."""
    # Build rtree with one point.
    box: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)

    def build_rtree1(index: RtreeIndex) -> None:
        index.insert(box, "a")

    with tempfile.TemporaryDirectory() as cache_dir:
        cached_dir_upath = UPath(cache_dir)
        index = get_cached_rtree(cached_dir_upath, build_rtree1)
        result = index.query(box)
        assert len(result) == 1 and result[0] == "a"
