import os
import pathlib
import random

from upath import UPath

from rslearn.utils.rtree_index import RtreeIndex, get_cached_rtree


def test_remote_cache(tmp_path: pathlib.Path):
    test_id = random.randint(10000, 99999)
    bucket_name = os.environ["TEST_BUCKET"]
    prefix = os.environ["TEST_PREFIX"] + f"test_{test_id}/"
    cache_dir = UPath(f"gcs://{bucket_name}/{prefix}")

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
