import logging
import multiprocessing

import pytest

logging.basicConfig()


@pytest.fixture(scope="session", autouse=True)
def always_spawn():
    multiprocessing.set_start_method("forkserver")
