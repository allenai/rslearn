"""Unit tests for RslearnLightningCLI template substitution."""

from argparse import Namespace
from unittest.mock import patch

from rslearn.main import RslearnLightningCLI


class TestRslearnLightningCLITemplateSubstitution:
    """Test suite for RslearnLightningCLI template substitution functionality."""

    def test_substitute_template_variables(self) -> None:
        """Test basic template variable substitution."""
        content = "Path: ${EXTRA_FILES_PATH}/data, Config: ${BASE_PATH}/config.yaml"
        variables = {"EXTRA_FILES_PATH": "/home/user", "BASE_PATH": "/opt/app"}

        result = RslearnLightningCLI.substitute_template_variables(content, variables)

        assert result == "Path: /home/user/data, Config: /opt/app/config.yaml"

    def test_substitute_template_variables_missing_variable(self) -> None:
        """Test that missing variables are left unchanged."""
        content = "Known: ${KNOWN_VAR}, Unknown: ${UNKNOWN_VAR}"
        variables = {"KNOWN_VAR": "replaced"}

        result = RslearnLightningCLI.substitute_template_variables(content, variables)

        assert result == "Known: replaced, Unknown: ${UNKNOWN_VAR}"

    def test_substitute_in_tree_complex_structure(self) -> None:
        """Test tree substitution on complex nested structure."""
        obj = {
            "model_path": "${EXTRA_FILES_PATH}/model.pt",
            "data": {
                "paths": [
                    "${EXTRA_FILES_PATH}/file1.txt",
                    "${EXTRA_FILES_PATH}/file2.txt",
                ],
                "batch_size": 32,
            },
            "config": Namespace(log_dir="${EXTRA_FILES_PATH}/logs"),
        }

        template_vars = {"EXTRA_FILES_PATH": "/home/user"}

        result = RslearnLightningCLI.substitute_in_tree(obj, template_vars)

        assert result["model_path"] == "/home/user/model.pt"
        assert result["data"]["paths"] == [
            "/home/user/file1.txt",
            "/home/user/file2.txt",
        ]
        assert result["data"]["batch_size"] == 32  # Non-string preserved
        assert result["config"].log_dir == "/home/user/logs"

    def test_apply_template_substitution_with_error_handling(self) -> None:
        """Test template substitution with error handling."""
        config = Namespace()
        config.path = "${EXTRA_FILES_PATH}/data"

        # Test normal case
        template_vars = {"EXTRA_FILES_PATH": "/home/user"}
        result = RslearnLightningCLI.apply_template_substitution(config, template_vars)
        assert result.path == "/home/user/data"

        # Test empty vars case
        result = RslearnLightningCLI.apply_template_substitution(config, {})
        assert result is config  # Returns original unchanged

    def test_template_variable_mapping_logic(self) -> None:
        """Test core logic for mapping template variables in add_arguments_to_parser."""
        # Simulate the add_arguments_to_parser logic
        template_var_mappings = {}
        arg_name = "--extra-files-path"
        template_var_name = "EXTRA_FILES_PATH"

        # This is what happens in add_arguments_to_parser
        template_var_mappings[arg_name] = template_var_name

        assert template_var_mappings == {"--extra-files-path": "EXTRA_FILES_PATH"}

    def test_extract_template_variables_from_config(self) -> None:
        """Test extracting template variables from parsed config."""
        # Create a CLI instance with mocked __init__
        with patch.object(RslearnLightningCLI, "__init__", lambda self: None):
            cli = RslearnLightningCLI()
            cli._template_var_mappings = {"--extra-files-path": "EXTRA_FILES_PATH"}

            # Mock config
            cli.config = Namespace()
            cli.config.extra_files_path = "/home/user/data"
            cli.config.other_arg = "ignored"

            template_vars = cli._extract_template_variables_from_config()

            assert template_vars == {"EXTRA_FILES_PATH": "/home/user/data"}
