import multiprocessing

import pytest


@pytest.fixture(scope="session", autouse=True)
def always_forkserver():
    multiprocessing.set_start_method("forkserver")
