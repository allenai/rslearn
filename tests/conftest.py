import multiprocessing

import pytest


@pytest.fixture(scope="session", autouse=True)
def always_spawn():
    multiprocessing.set_start_method("spawn")
