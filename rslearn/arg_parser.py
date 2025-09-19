"""Custom Lightning ArgumentParser with environment variable substitution support."""

import os
import re
from typing import Any

from jsonargparse import Namespace
from lightning.pytorch.cli import LightningArgumentParser


def substitute_env_vars_in_string(content: str) -> str:
    """Substitute environment variables in a string.

    Replaces ${VAR_NAME} patterns with os.getenv(VAR_NAME, "") values.
    This works on raw string content before YAML parsing.

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


class RslearnArgumentParser(LightningArgumentParser):
    """Custom ArgumentParser that substitutes environment variables in config files.

    This parser extends LightningArgumentParser to automatically substitute
    ${VAR_NAME} patterns with environment variable values before parsing
    configuration content. This allows config files to use environment
    variables while still passing Lightning's validation.
    """

    def parse_string(
        self,
        cfg_str: str,
        cfg_path: str | os.PathLike = "",
        ext_vars: dict | None = None,
        env: bool | None = None,
        defaults: bool = True,
        with_meta: bool | None = None,
        **kwargs: Any,
    ) -> Namespace:
        """Pre-processes string for environment variable substitution before parsing."""
        # Substitute environment variables in the config string before parsing
        substituted_cfg_str = substitute_env_vars_in_string(cfg_str)

        # Call the parent method with the substituted config
        return super().parse_string(
            substituted_cfg_str, cfg_path, ext_vars, env, defaults, with_meta, **kwargs
        )
