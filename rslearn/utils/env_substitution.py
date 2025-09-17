"""Environment variable substitution utilities for configuration files.

This module provides utilities for substituting environment variables in parsed
configuration objects, with YAML-based type inference to preserve correct data types.
"""

import os
import re
from typing import Any

import yaml


def substitute_env_vars_in_tree(obj: Any) -> Any:
    """Recursively walk the object tree and substitute environment variables in string values.

    Replaces ${VAR_NAME} patterns with os.getenv(VAR_NAME) values and attempts
    to infer the correct type for the substituted value.

    Args:
        obj: The object to process (can be dict, list, tuple, object with attributes, etc.)

    Returns:
        The object with environment variables substituted
    """
    if isinstance(obj, str):
        # Apply substitution to string values
        return substitute_env_vars_in_string(obj)

    elif isinstance(obj, dict):
        # Recursively process dictionary values
        return {key: substitute_env_vars_in_tree(value) for key, value in obj.items()}

    elif isinstance(obj, list):
        # Recursively process list items
        return [substitute_env_vars_in_tree(item) for item in obj]

    elif isinstance(obj, tuple):
        # Recursively process tuple items
        return tuple(substitute_env_vars_in_tree(item) for item in obj)

    elif hasattr(obj, "__dict__"):
        # Handle objects with attributes (like Namespace)
        for attr_name in vars(obj):
            if not attr_name.startswith("_"):  # Skip private attributes
                try:
                    attr_value = getattr(obj, attr_name)
                    new_value = substitute_env_vars_in_tree(attr_value)
                    setattr(obj, attr_name, new_value)
                except (AttributeError, TypeError):
                    # Skip attributes that can't be set
                    pass
        return obj

    else:
        # Return other types unchanged (int, float, bool, None, etc.)
        return obj


def substitute_env_vars_in_string(content: str) -> Any:
    """Substitute template variables in content string and parse result as YAML.

    Replaces instances of ${VARIABLE_NAME} with values from environment variables
    and then parses the result as YAML to get the correct type.

    Args:
        content: The string content containing template variables

    Returns:
        The substituted value with type determined by YAML parsing
    """
    pattern = r"\$\{([^}]+)\}"

    def replace_variable(match_obj: re.Match[str]) -> str:
        var_name = match_obj.group(1)
        env_value = os.getenv(var_name, "")
        return env_value if env_value is not None else ""

    interpolated = re.sub(pattern, replace_variable, content)

    if interpolated == content:
        return content

    # Parse the interpolated value as YAML to get correct type.
    # We need this because we're performing substitution after
    # yaml has been parsed, i.e. ${NUM_WORKERS} will have been
    # interpreted as a string, but the actual values being interpolated
    # should be integers.
    try:
        return yaml.safe_load(interpolated)
    except yaml.YAMLError:
        # If YAML parsing fails, return as string
        return interpolated
