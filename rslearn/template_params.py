"""Template parameter substitution utilities for rslearn configuration files."""

import os
import re


def substitute_env_vars_in_string(content: str) -> str:
    """Substitute environment variables in a string.

    Replaces ${VAR_NAME} patterns with os.getenv(VAR_NAME, "") values.
    This works on raw string content before YAML/JSON parsing.

    Args:
        content: The string content containing template variables

    Returns:
        The string with environment variables substituted
    """
    pattern = r"\$\{([^}]+)\}"

    def replace_variable(match_obj: re.Match[str]) -> str:
        var_name = match_obj.group(1)
        env_value = os.getenv(var_name, "")
        return env_value if env_value is not None else ""

    return re.sub(pattern, replace_variable, content)
