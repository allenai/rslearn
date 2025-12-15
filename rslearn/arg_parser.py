"""Custom Lightning ArgumentParser with environment variable substitution support."""

from typing import Any

from jsonargparse import Namespace
from lightning.pytorch.cli import LightningArgumentParser

from rslearn.template_params import substitute_env_vars_in_string


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
        **kwargs: Any,
    ) -> Namespace:
        """Pre-processes string for environment variable substitution before parsing."""
        # Substitute environment variables in the config string before parsing
        substituted_cfg_str = substitute_env_vars_in_string(cfg_str)

        # Call the parent method with the substituted config
        return super().parse_string(substituted_cfg_str, **kwargs)
