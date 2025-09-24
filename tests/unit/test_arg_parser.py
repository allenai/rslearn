"""Unit tests for RslearnArgumentParser with environment variable substitution."""

import pytest
from lightning.pytorch import LightningDataModule, LightningModule

from rslearn.arg_parser import RslearnArgumentParser, substitute_env_vars_in_string


class DummyModel(LightningModule):
    """Dummy model for testing."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        num_workers: int = 4,
        model_path: str = "model.pt",
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.model_path = model_path


class DummyDataModule(LightningDataModule):
    """Dummy data module for testing."""

    def __init__(self, batch_size: int = 32, data_root: str = "/tmp"):
        super().__init__()
        self.batch_size = batch_size
        self.data_root = data_root


class TestRslearnArgumentParser:
    """Test suite for RslearnArgumentParser environment variable substitution."""

    def test_substitute_env_vars_in_string_basic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test basic template variable substitution with environment variables."""
        monkeypatch.setenv("TEST_PATH", "/home/user")
        monkeypatch.setenv("CONFIG_DIR", "/opt/app")

        content = "Path: ${TEST_PATH}/data, Config: ${CONFIG_DIR}/config.yaml"
        result = substitute_env_vars_in_string(content)

        assert result == "Path: /home/user/data, Config: /opt/app/config.yaml"

    def test_substitute_env_vars_in_string_missing_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that missing variables are replaced with empty string."""
        monkeypatch.setenv("KNOWN_VAR", "replaced")
        # UNKNOWN_VAR is not set

        content = "Known: ${KNOWN_VAR}, Unknown: ${UNKNOWN_VAR}"
        result = substitute_env_vars_in_string(content)

        assert result == "Known: replaced, Unknown: "

    def test_substitute_env_vars_in_string_no_variables(self) -> None:
        """Test content with no template variables."""
        content = "This is just a regular string with no variables"
        result = substitute_env_vars_in_string(content)

        assert result == content

    def test_environment_variable_substitution_before_validation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that parse_string works correctly with Lightning's class_path/init_args structure."""
        # Set up environment variables
        monkeypatch.setenv("LEARNING_RATE", "0.001")
        monkeypatch.setenv("NUM_WORKERS", "8")
        monkeypatch.setenv("BATCH_SIZE", "64")
        monkeypatch.setenv("DATA_ROOT", "/tmp/test_data")

        # Create a parser like Lightning CLI does - with subclass mode
        parser = RslearnArgumentParser()

        # Use Lightning's subclass configuration - this supports class_path/init_args structure
        parser.add_subclass_arguments(DummyModel, "model")
        parser.add_subclass_arguments(DummyDataModule, "data")

        # Create config content with environment variables - using Lightning's init_args structure
        yaml_content = """
        model:
          class_path: tests.unit.test_arg_parser.DummyModel
          init_args:
            learning_rate: ${LEARNING_RATE}
            num_workers: ${NUM_WORKERS}
        data:
          class_path: tests.unit.test_arg_parser.DummyDataModule
          init_args:
            batch_size: ${BATCH_SIZE}
            data_root: ${DATA_ROOT}
        """

        # This is the critical test - parse_string should:
        # 1. Substitute environment variables BEFORE validation
        # 2. Parse successfully with correct types
        result = parser.parse_string(yaml_content)

        # Verify that environment variables were substituted and parsed correctly
        assert result.model.class_path == "tests.unit.test_arg_parser.DummyModel"
        assert (
            result.model.init_args.learning_rate == 0.001
        )  # String "0.001" -> float 0.001
        assert result.model.init_args.num_workers == 8  # String "8" -> int 8

        assert result.data.class_path == "tests.unit.test_arg_parser.DummyDataModule"
        assert result.data.init_args.batch_size == 64  # String "64" -> int 64
        assert (
            result.data.init_args.data_root == "/tmp/test_data"
        )  # String remains string
