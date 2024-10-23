"""Multi-processing utilities."""

import multiprocessing.pool
from collections.abc import Callable
from multiprocessing.pool import IMapIterator
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
) -> IMapIterator:
    """Wrapper for Pool.imap_unordered that exposes kwargs to the function.

    Args:
        p: the multiprocessing.pool.Pool to use.
        fn: the function to call, which accepts keyword arguments.
        kwargs_list: list of kwargs dicts to pass to the function.

    Returns:
        generator for outputs from the function in arbitrary order.
    """
    return p.imap_unordered(StarImapUnorderedWrapper(fn), kwargs_list)
