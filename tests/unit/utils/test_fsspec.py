import multiprocessing
import os
import pathlib
import time

import pytest
from upath import UPath

from rslearn.utils.fsspec import get_tmp_path, is_tmp_path, open_atomic

MESSAGE = "hello world"
SLEEP_TIME = 1


def sleepy_writer(fname: str) -> None:
    with open_atomic(fname, "w") as f:
        time.sleep(SLEEP_TIME * 2)
        f.write(MESSAGE)


def test_open_atomic(tmp_path: pathlib.Path) -> None:
    # Make sure that open_atomic actually creates file atomically on local filesystem.
    # So we create file, then write to it in another process and sleep and read from
    # first process and make sure it's okay.
    tmp_fname = UPath(tmp_path) / "test.txt"
    with open_atomic(tmp_fname, "w") as f:
        f.write(MESSAGE)
    p = multiprocessing.Process(target=sleepy_writer, args=[tmp_fname])
    p.start()
    time.sleep(SLEEP_TIME)
    with open(tmp_fname) as f:
        message = f.read()
        assert message == MESSAGE
    p.join()
    with open(tmp_fname) as f:
        message = f.read()
        assert message == MESSAGE


def test_get_tmp_path_is_recognized(tmp_path: pathlib.Path) -> None:
    # The path open_atomic writes to must be one that directory listings can identify
    # as a temporary file, so an in-progress write is never mistaken for data.
    path = UPath(tmp_path) / "image.tif"
    tmp_fname = get_tmp_path(path)
    assert tmp_fname.endswith(f".tmp.{os.getpid()}")
    assert is_tmp_path(UPath(tmp_fname))


@pytest.mark.parametrize(
    "name,expected",
    [
        ("image.tif.tmp.1234", True),
        ("summary.json.tmp.1234", True),
        ("image.tif", False),
        ("summary.json", False),
        # Only the suffix open_atomic writes counts, not any mention of "tmp".
        ("image.tmp.tif", False),
        ("image.tif.tmp.abcd", False),
        # The suffix is always appended to a destination name, so there is a file
        # name in front of it.
        ("tmp.1234", False),
    ],
)
def test_is_tmp_path(tmp_path: pathlib.Path, name: str, expected: bool) -> None:
    assert is_tmp_path(UPath(tmp_path) / name) is expected
