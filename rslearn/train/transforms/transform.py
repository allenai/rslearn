"""Transform base class."""

from collections.abc import Callable
from typing import Any

import torch


def get_dict_and_subselector(
    input_dict: dict[str, Any], target_dict: dict[str, Any], selector: str
) -> tuple[dict[str, Any], str]:
    """Determine whether to use input or target dict, and the sub-selector.

    For example, if the selector is "input/x", then we use input dict and the
    sub-selector is "x".

    If neither input/ nor target/ prefixes are present, then we assume it is for
    input dict.

    Args:
        input_dict: the input dict
        target_dict: the target dict
        selector: the full selector configured by the user

    Returns:
        a tuple (referenced dict, sub-selector string)
    """
    input_prefix = "input/"
    target_prefix = "target/"

    if selector.startswith(input_prefix):
        d = input_dict
        selector = selector[len(input_prefix) :]
    elif selector.startswith(target_prefix):
        d = target_dict
        selector = selector[len(target_prefix) :]
    else:
        d = input_dict

    return d, selector


def read_selector(
    input_dict: dict[str, Any], target_dict: dict[str, Any], selector: str
) -> Any:
    """Read the item referenced by the selector.

    Args:
        input_dict: the input dict
        target_dict: the target dict
        selector: the selector specifying the item to read

    Returns:
        the item specified by the selector
    """
    d, selector = get_dict_and_subselector(input_dict, target_dict, selector)
    parts = selector.split("/") if selector else []
    cur = d
    for part in parts:
        cur = cur[part]
    return cur


def write_selector(
    input_dict: dict[str, Any],
    target_dict: dict[str, Any],
    selector: str,
    v: Any,
) -> None:
    """Write the item to the specified selector.

    Args:
        input_dict: the input dict
        target_dict: the target dict
        selector: the selector specifying the item to write
        v: the value to write
    """
    d, selector = get_dict_and_subselector(input_dict, target_dict, selector)
    if selector:
        parts = selector.split("/")
        cur = d
        for part in parts[:-1]:
            cur = cur[part]
        cur[parts[-1]] = v
    else:
        # If the selector references the input or target dictionary directly, then we
        # have a special case where instead of overwriting with v, we replace the keys
        # with those in v. v must be a dictionary here, not a tensor, since otherwise
        # it wouldn't match the type of the input or target dictionary.
        if not isinstance(v, dict):
            raise ValueError(
                "when directly specifying the input or target dict, expected the value to be a dict"
            )
        if d == v:
            # This may happen if the writer did not make a copy of the dictionary. In
            # this case the code below would not update d correctly since it would also
            # clear v.
            return
        d.clear()
        d.update(v)


def selector_exists(
    input_dict: dict[str, Any], target_dict: dict[str, Any], selector: str
) -> bool:
    """Check if the specified selector exists in the dicts.

    Args:
        input_dict: the input dict
        target_dict: the target dict
        selector: the selector specifying the item to check

    Returns:
        True if the selector exists, False otherwise
    """
    d, selector = get_dict_and_subselector(input_dict, target_dict, selector)
    parts = selector.split("/") if selector else []
    cur = d
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


class Transform(torch.nn.Module):
    """An rslearn transform.

    Provides helper functions for subclasses to select input and target keys and to
    transform them.

    Attributes:
        skip_missing: If True, selectors that don't exist in the input/target dicts
            will be silently skipped. Useful when working with optional inputs.
    """

    skip_missing: bool = False

    def apply_fn(
        self,
        fn: Callable,
        input_dict: dict[str, Any],
        target_dict: dict[str, Any],
        selectors: list[str],
        **kwargs: Any,
    ) -> None:
        """Apply the specified function on the selectors in input/target dicts.

        Args:
            fn: the function to apply
            input_dict: the input dict
            target_dict: the target dict
            selectors: the selectors to apply the function on.
            kwargs: additional arguments to pass to the function
        """
        for selector in selectors:
            if self.skip_missing and not selector_exists(
                input_dict, target_dict, selector
            ):
                continue
            v = read_selector(input_dict, target_dict, selector)
            v = fn(v, **kwargs)
            write_selector(input_dict, target_dict, selector, v)


class Identity(Transform):
    """Identity transform."""

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Simply returns the provided input_dict and target_dict.

        Args:
            input_dict: input dict.
            target_dict: target_dict.

        Returns:
            unchanged (input_dict, target_dict)
        """
        return input_dict, target_dict
