"""Unit tests for RslearnLightningCLI environment variable substitution."""

from argparse import Namespace
from typing import Any

import pytest

from rslearn.main import RslearnLightningCLI


class TestRslearnLightningCLIEnvSubstitution:
    """Test suite for RslearnLightningCLI environment variable substitution functionality."""

    def test_substitute_env_vars_in_string_basic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test basic template variable substitution with environment variables."""
        monkeypatch.setenv("TEST_PATH", "/home/user")
        monkeypatch.setenv("CONFIG_DIR", "/opt/app")

        content = "Path: ${TEST_PATH}/data, Config: ${CONFIG_DIR}/config.yaml"
        result = RslearnLightningCLI.substitute_env_vars_in_string(content)

        assert result == "Path: /home/user/data, Config: /opt/app/config.yaml"

    def test_substitute_env_vars_in_string_missing_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that missing variables are replaced with empty string."""
        monkeypatch.setenv("KNOWN_VAR", "replaced")
        # UNKNOWN_VAR is not set

        content = "Known: ${KNOWN_VAR}, Unknown: ${UNKNOWN_VAR}"
        result = RslearnLightningCLI.substitute_env_vars_in_string(content)

        assert result == "Known: replaced, Unknown: "

    def test_substitute_env_vars_in_string_no_variables(self) -> None:
        """Test content with no template variables."""
        content = "This is just a regular string with no variables"
        result = RslearnLightningCLI.substitute_env_vars_in_string(content)

        assert result == content

    def test_substitute_env_vars_in_string_empty_string(self) -> None:
        """Test empty string input."""
        result = RslearnLightningCLI.substitute_env_vars_in_string("")
        assert result == ""

    def test_substitute_env_vars_in_string_malformed_patterns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test handling of malformed variable patterns."""
        monkeypatch.setenv("VALID_VAR", "valid")

        content = (
            "Valid: ${VALID_VAR}, Incomplete: ${INCOMPLETE, NoClosing: ${NO_CLOSING"
        )
        result = RslearnLightningCLI.substitute_env_vars_in_string(content)

        # Only properly formatted variables should be replaced
        assert (
            result == "Valid: valid, Incomplete: ${INCOMPLETE, NoClosing: ${NO_CLOSING"
        )

    def test_substitute_env_vars_in_tree_simple_dict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test tree substitution on simple dictionary."""
        monkeypatch.setenv("DATA_PATH", "/home/data")
        monkeypatch.setenv("MODEL_PATH", "/opt/models")

        obj = {
            "data_dir": "${DATA_PATH}",
            "model_dir": "${MODEL_PATH}",
            "batch_size": 32,
        }

        result = RslearnLightningCLI.substitute_env_vars_in_tree(obj)

        assert result["data_dir"] == "/home/data"
        assert result["model_dir"] == "/opt/models"
        assert result["batch_size"] == 32  # Non-string preserved

    def test_substitute_env_vars_in_tree_nested_structure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test tree substitution on complex nested structure."""
        monkeypatch.setenv("BASE_PATH", "/home/user")
        monkeypatch.setenv("LOG_DIR", "/var/log")

        obj = {
            "model_path": "${BASE_PATH}/model.pt",
            "data": {
                "paths": [
                    "${BASE_PATH}/file1.txt",
                    "${BASE_PATH}/file2.txt",
                ],
                "batch_size": 32,
                "config": {"log_dir": "${LOG_DIR}/app.log"},
            },
            "flags": [True, False],
            "metadata": None,
        }

        result = RslearnLightningCLI.substitute_env_vars_in_tree(obj)

        assert result["model_path"] == "/home/user/model.pt"
        assert result["data"]["paths"] == [
            "/home/user/file1.txt",
            "/home/user/file2.txt",
        ]
        assert result["data"]["batch_size"] == 32
        assert result["data"]["config"]["log_dir"] == "/var/log/app.log"
        assert result["flags"] == [True, False]  # Non-string list preserved
        assert result["metadata"] is None

    def test_substitute_env_vars_in_tree_with_namespace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test tree substitution with Namespace objects."""
        monkeypatch.setenv("WORKSPACE_PATH", "/workspace")

        config = Namespace()
        config.log_dir = "${WORKSPACE_PATH}/logs"
        config.data_dir = "${WORKSPACE_PATH}/data"
        config.num_workers = 4

        obj = {"config": config, "other_data": "${WORKSPACE_PATH}/other"}

        result = RslearnLightningCLI.substitute_env_vars_in_tree(obj)

        assert result["config"].log_dir == "/workspace/logs"
        assert result["config"].data_dir == "/workspace/data"
        assert result["config"].num_workers == 4
        assert result["other_data"] == "/workspace/other"

    def test_substitute_env_vars_in_tree_with_tuples(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test tree substitution with tuples."""
        monkeypatch.setenv("COORD_X", "10")
        monkeypatch.setenv("COORD_Y", "20")

        obj = {
            "coordinates": ("${COORD_X}", "${COORD_Y}"),
            "mixed_tuple": ("string", 42, "${COORD_X}"),
        }

        result = RslearnLightningCLI.substitute_env_vars_in_tree(obj)

        # YAML parsing converts "10" to integer 10
        assert result["coordinates"] == (10, 20)
        assert result["mixed_tuple"] == ("string", 42, 10)

    def test_substitute_env_vars_in_tree_empty_objects(self) -> None:
        """Test tree substitution with empty objects."""
        empty_dict: dict[str, Any] = {}
        empty_list: list[Any] = []
        empty_tuple = ()

        assert RslearnLightningCLI.substitute_env_vars_in_tree(empty_dict) == {}
        assert RslearnLightningCLI.substitute_env_vars_in_tree(empty_list) == []
        assert RslearnLightningCLI.substitute_env_vars_in_tree(empty_tuple) == ()

    def test_substitute_env_vars_in_tree_non_string_types(self) -> None:
        """Test that non-string types are preserved unchanged."""
        obj = {
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "none_value": None,
            "complex_number": 1 + 2j,
        }

        result = RslearnLightningCLI.substitute_env_vars_in_tree(obj)

        assert result == obj  # Should be identical

    def test_substitute_env_vars_in_tree_with_missing_env_vars(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test tree substitution when environment variables are missing."""
        monkeypatch.setenv("EXISTING_VAR", "exists")
        # MISSING_VAR is not set

        obj = {
            "existing": "${EXISTING_VAR}",
            "missing": "${MISSING_VAR}",
            "mixed": "prefix_${EXISTING_VAR}_${MISSING_VAR}_suffix",
        }

        result = RslearnLightningCLI.substitute_env_vars_in_tree(obj)

        assert result["existing"] == "exists"
        assert result["missing"] is None  # Empty string parsed as None by YAML
        assert result["mixed"] == "prefix_exists__suffix"  # Mixed substitution

    def test_substitute_env_vars_in_tree_with_special_characters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test substitution with special characters in environment variables."""
        monkeypatch.setenv("SPECIAL_PATH", "/path/with spaces/and-dashes_underscores")
        monkeypatch.setenv("SYMBOLS", "!@#$%^&*()")

        obj = {"path": "${SPECIAL_PATH}", "symbols": "${SYMBOLS}"}

        result = RslearnLightningCLI.substitute_env_vars_in_tree(obj)

        assert result["path"] == "/path/with spaces/and-dashes_underscores"
        assert result["symbols"] == "!@#$%^&*()"

    def test_substitute_env_vars_in_tree_preserves_object_identity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that objects without string attributes preserve their identity."""

        class CustomObject:
            def __init__(self) -> None:
                self.numeric_attr = 42
                self._private_attr = "should be ignored"

        custom_obj = CustomObject()
        obj = {"custom": custom_obj, "number": 123}

        result = RslearnLightningCLI.substitute_env_vars_in_tree(obj)

        # The custom object should be the same instance since it has no string attrs to substitute
        assert result["custom"] is custom_obj
        assert result["custom"].numeric_attr == 42
        assert result["number"] == 123

    def test_substitute_env_vars_in_string_yaml_type_inference(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that substituted values are parsed as YAML for correct type inference."""
        monkeypatch.setenv("INT_VAR", "42")
        monkeypatch.setenv("FLOAT_VAR", "3.14")
        monkeypatch.setenv("BOOL_TRUE_VAR", "true")
        monkeypatch.setenv("BOOL_FALSE_VAR", "false")
        monkeypatch.setenv("NULL_VAR", "null")
        monkeypatch.setenv("STRING_VAR", "hello world")
        monkeypatch.setenv("LIST_VAR", "[1, 2, 3]")
        monkeypatch.setenv("DICT_VAR", "{key: value}")

        # Test pure substitution with type inference
        assert RslearnLightningCLI.substitute_env_vars_in_string("${INT_VAR}") == 42
        assert RslearnLightningCLI.substitute_env_vars_in_string("${FLOAT_VAR}") == 3.14
        assert (
            RslearnLightningCLI.substitute_env_vars_in_string("${BOOL_TRUE_VAR}")
            is True
        )
        assert (
            RslearnLightningCLI.substitute_env_vars_in_string("${BOOL_FALSE_VAR}")
            is False
        )
        assert RslearnLightningCLI.substitute_env_vars_in_string("${NULL_VAR}") is None
        assert (
            RslearnLightningCLI.substitute_env_vars_in_string("${STRING_VAR}")
            == "hello world"
        )
        assert RslearnLightningCLI.substitute_env_vars_in_string("${LIST_VAR}") == [
            1,
            2,
            3,
        ]
        assert RslearnLightningCLI.substitute_env_vars_in_string("${DICT_VAR}") == {
            "key": "value"
        }

        # Test partial substitution (should remain strings)
        assert (
            RslearnLightningCLI.substitute_env_vars_in_string(
                "prefix_${INT_VAR}_suffix"
            )
            == "prefix_42_suffix"
        )
        assert (
            RslearnLightningCLI.substitute_env_vars_in_string("path/${STRING_VAR}/file")
            == "path/hello world/file"
        )

        # Test no substitution
        assert (
            RslearnLightningCLI.substitute_env_vars_in_string("no variables here")
            == "no variables here"
        )

    def test_substitute_env_vars_in_string_yaml_error_handling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that invalid YAML in substituted values falls back to string."""
        monkeypatch.setenv("INVALID_YAML", "{ invalid: yaml: syntax }")

        # Should fall back to string if YAML parsing fails
        result = RslearnLightningCLI.substitute_env_vars_in_string("${INVALID_YAML}")
        assert result == "{ invalid: yaml: syntax }"
        assert isinstance(result, str)

    def test_substitute_env_vars_in_string_multiple_variables(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that multiple variables in a single string are all substituted."""
        monkeypatch.setenv("VAR_1", "first")
        monkeypatch.setenv("VAR_2", "second")
        monkeypatch.setenv("VAR_3", "third")

        # Test multiple substitutions in various patterns
        test_cases = [
            ("${VAR_1}_asdf_${VAR_2}", "first_asdf_second"),
            ("${VAR_1}/${VAR_2}/${VAR_3}", "first/second/third"),
            (
                "prefix_${VAR_1}_middle_${VAR_2}_suffix",
                "prefix_first_middle_second_suffix",
            ),
            ("${VAR_1}${VAR_2}${VAR_3}", "firstsecondthird"),  # No separators
            ("start${VAR_1}${VAR_2}end", "startfirstsecondend"),
        ]

        for input_str, expected in test_cases:
            result = RslearnLightningCLI.substitute_env_vars_in_string(input_str)
            assert result == expected, f"Failed for input: {input_str}"
            assert isinstance(result, str)

    def test_substitute_env_vars_in_string_mixed_present_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test multiple variables where some are present and some are missing."""
        monkeypatch.setenv("PRESENT_VAR", "exists")
        # MISSING_VAR is not set

        test_cases = [
            ("${PRESENT_VAR}_${MISSING_VAR}", "exists_"),
            ("${MISSING_VAR}_${PRESENT_VAR}", "_exists"),
            ("${PRESENT_VAR}_middle_${MISSING_VAR}_end", "exists_middle__end"),
            (
                "${MISSING_VAR}${MISSING_VAR}${PRESENT_VAR}",
                "exists",
            ),  # Multiple missing
        ]

        for input_str, expected in test_cases:
            result = RslearnLightningCLI.substitute_env_vars_in_string(input_str)
            assert result == expected, f"Failed for input: {input_str}"
            assert isinstance(result, str)

    def test_substitute_env_vars_in_string_same_variable_multiple_times(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test the same variable used multiple times in one string."""
        monkeypatch.setenv("REPEAT_VAR", "repeated")

        test_cases = [
            ("${REPEAT_VAR}_${REPEAT_VAR}", "repeated_repeated"),
            ("${REPEAT_VAR}/path/${REPEAT_VAR}/file", "repeated/path/repeated/file"),
            ("${REPEAT_VAR}${REPEAT_VAR}${REPEAT_VAR}", "repeatedrepeatedrepeated"),
        ]

        for input_str, expected in test_cases:
            result = RslearnLightningCLI.substitute_env_vars_in_string(input_str)
            assert result == expected, f"Failed for input: {input_str}"
            assert isinstance(result, str)
