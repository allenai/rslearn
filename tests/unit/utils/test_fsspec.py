import multiprocessing
import pathlib
import time

from upath import UPath

from rslearn.utils.fsspec import open_atomic

MESSAGE = "hello world"
SLEEP_TIME = 1


def sleepy_writer(fname: str):
    with open_atomic(fname, "w") as f:
        time.sleep(SLEEP_TIME * 2)
        f.write(MESSAGE)


def test_open_atomic(tmp_path: pathlib.Path):
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
