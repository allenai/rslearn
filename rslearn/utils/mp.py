"""Multi-processing utilities."""

import multiprocessing
import multiprocessing.pool
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import Any


class StarImapUnorderedWrapper:
    """Wrapper for a function to implement star_imap_unordered.

    A kwargs dict is passed to this wrapper, which then calls the underlying function
    with the unwrapped kwargs.
    """

    def __init__(self, fn: Callable[..., Any]):
        """Create a new StarImapUnordered.

        Args:
            fn: the underlying function to call.
        """
        self.fn = fn

    def __call__(self, kwargs: dict[str, Any]) -> Any:
        """Wrapped call to the underlying function.

        Args:
            kwargs: dict of keyword arguments to pass to the function.
        """
        return self.fn(**kwargs)


def star_imap_unordered(
    p: multiprocessing.pool.Pool,
    fn: Callable[..., Any],
    kwargs_list: list[dict[str, Any]],
) -> multiprocessing.pool.IMapIterator:
    """Wrapper for Pool.imap_unordered that exposes kwargs to the function.

    Args:
        p: the multiprocessing.pool.Pool to use.
        fn: the function to call, which accepts keyword arguments.
        kwargs_list: list of kwargs dicts to pass to the function.

    Returns:
        generator for outputs from the function in arbitrary order.
    """
    return p.imap_unordered(StarImapUnorderedWrapper(fn), kwargs_list)


@contextmanager
def make_pool_and_star_imap_unordered(
    workers: int,
    fn: Callable[..., Any],
    kwargs_list: list[dict[str, Any]],
) -> Iterator[Iterable[Any]]:
    """Context manager that creates a pool and yields an imap_unordered iterable.

    When workers <= 0, runs sequentially in the current process with no pool.

    Args:
        workers: number of worker processes. 0 means no multiprocessing.
        fn: the function to call, which accepts keyword arguments.
        kwargs_list: list of kwargs dicts to pass to the function.

    Yields:
        iterable of outputs from the function (arbitrary order).
    """
    if workers <= 0:
        yield (fn(**kwargs) for kwargs in kwargs_list)
    else:
        p = multiprocessing.Pool(workers)
        try:
            yield p.imap_unordered(StarImapUnorderedWrapper(fn), kwargs_list)
        finally:
            p.close()
            p.join()
