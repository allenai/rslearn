"""Unit tests for rslearn.dataset.index."""

import json
from pathlib import Path

from upath import UPath

from rslearn.dataset.index import DatasetIndex


class MockDataInput:
    """Mock DataInput for testing index key generation."""

    def __init__(
        self,
        layers: list[str],
        required: bool = True,
        load_all_layers: bool = False,
        is_target: bool = False,
    ):
        self.layers = layers
        self.required = required
        self.load_all_layers = load_all_layers
        self.is_target = is_target


class TestDatasetIndex:
    """Test suite for DatasetIndex."""

    def test_get_index_key_deterministic(self, tmp_path: Path) -> None:
        """Test that index key generation is deterministic."""
        index = DatasetIndex(UPath(tmp_path))
        inputs = {
            "image": MockDataInput(["layer1", "layer2"]),
            "label": MockDataInput(["label_layer"], is_target=True),
        }

        key1 = index.get_index_key(
            groups=["train"],
            names=None,
            tags={"split": "train"},
            num_samples=1000,
            skip_targets=False,
            inputs=inputs,
            disabled_layers=[],
        )
        key2 = index.get_index_key(
            groups=["train"],
            names=None,
            tags={"split": "train"},
            num_samples=1000,
            skip_targets=False,
            inputs=inputs,
            disabled_layers=[],
        )

        assert key1 == key2
        assert len(key1) == 16  # 16 hex characters

    def test_get_index_key_different_configs(self, tmp_path: Path) -> None:
        """Test that different configs produce different index keys."""
        index = DatasetIndex(UPath(tmp_path))
        inputs = {"image": MockDataInput(["layer1"])}

        key1 = index.get_index_key(
            groups=["train"],
            names=None,
            tags=None,
            num_samples=None,
            skip_targets=False,
            inputs=inputs,
            disabled_layers=[],
        )
        key2 = index.get_index_key(
            groups=["val"],  # Different group
            names=None,
            tags=None,
            num_samples=None,
            skip_targets=False,
            inputs=inputs,
            disabled_layers=[],
        )
        key3 = index.get_index_key(
            groups=["train"],
            names=None,
            tags=None,
            num_samples=100,  # Different num_samples
            skip_targets=False,
            inputs=inputs,
            disabled_layers=[],
        )

        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

    def test_get_config_hash_with_config(self, tmp_path: Path) -> None:
        """Test config hash when config.json exists."""
        config_content = {"layers": {"test": {"type": "vector"}}}
        with (tmp_path / "config.json").open("w") as f:
            json.dump(config_content, f)

        index = DatasetIndex(UPath(tmp_path))
        hash1 = index.get_config_hash()

        assert hash1 != ""
        assert len(hash1) == 16

        # Same content should produce same hash
        hash2 = index.get_config_hash()
        assert hash1 == hash2

    def test_get_config_hash_without_config(self, tmp_path: Path) -> None:
        """Test config hash when config.json doesn't exist."""
        index = DatasetIndex(UPath(tmp_path))
        assert index.get_config_hash() == ""

    def test_save_and_load_windows(self, tmp_path: Path) -> None:
        """Test saving and loading indexed windows."""
        index = DatasetIndex(UPath(tmp_path))
        index_key = "test_index_key"

        windows = [
            {"group": "train", "name": "window1", "bounds": [0, 0, 256, 256]},
            {"group": "train", "name": "window2", "bounds": [256, 0, 512, 256]},
        ]

        # Save windows
        index.save_windows(index_key, windows)

        # Load windows
        loaded = index.load_windows(index_key)

        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0]["name"] == "window1"
        assert loaded[1]["name"] == "window2"

    def test_load_nonexistent_index(self, tmp_path: Path) -> None:
        """Test loading returns None when index doesn't exist."""
        index = DatasetIndex(UPath(tmp_path))
        result = index.load_windows("nonexistent_key")
        assert result is None

    def test_refresh_index_ignores_existing(self, tmp_path: Path) -> None:
        """Test that refresh_index=True ignores existing index."""
        index = DatasetIndex(UPath(tmp_path))
        index_key = "test_index_key"

        windows = [{"group": "train", "name": "window1"}]
        index.save_windows(index_key, windows)

        # With refresh_index=False, should load
        loaded = index.load_windows(index_key, refresh_index=False)
        assert loaded is not None

        # With refresh_index=True, should return None
        loaded = index.load_windows(index_key, refresh_index=True)
        assert loaded is None

    def test_index_invalidated_on_config_change(self, tmp_path: Path) -> None:
        """Test that index is invalidated when config.json changes."""
        # Create initial config
        config1 = {"layers": {"layer1": {"type": "vector"}}}
        with (tmp_path / "config.json").open("w") as f:
            json.dump(config1, f)

        index = DatasetIndex(UPath(tmp_path))
        index_key = "test_index_key"

        windows = [{"group": "train", "name": "window1"}]
        index.save_windows(index_key, windows)

        # Should load successfully
        loaded = index.load_windows(index_key)
        assert loaded is not None

        # Change config
        config2 = {"layers": {"layer1": {"type": "raster"}}}  # Different type
        with (tmp_path / "config.json").open("w") as f:
            json.dump(config2, f)

        # Should return None (index invalidated)
        loaded = index.load_windows(index_key)
        assert loaded is None

    def test_corrupted_index_returns_none(self, tmp_path: Path) -> None:
        """Test that corrupted index file returns None."""
        index = DatasetIndex(UPath(tmp_path))
        index_key = "test_index_key"

        # Create index directory and write invalid JSON
        index_dir = tmp_path / ".rslearn_index"
        index_dir.mkdir(parents=True, exist_ok=True)
        index_file = index_dir / f"{index_key}.json"
        with index_file.open("w") as f:
            f.write("not valid json {{{")

        # Should return None, not raise an error
        loaded = index.load_windows(index_key)
        assert loaded is None

    def test_atomic_write(self, tmp_path: Path) -> None:
        """Test that save uses atomic write (temp file + rename)."""
        index = DatasetIndex(UPath(tmp_path))
        index_key = "test_index_key"

        windows = [{"group": "train", "name": "window1"}]
        index.save_windows(index_key, windows)

        # Check that final file exists and temp file doesn't
        index_file = tmp_path / ".rslearn_index" / f"{index_key}.json"
        tmp_file = tmp_path / ".rslearn_index" / f"{index_key}.tmp"

        assert index_file.exists()
        assert not tmp_file.exists()

        # Verify content is valid JSON
        with index_file.open() as f:
            data = json.load(f)
        assert "windows" in data
        assert "config_hash" in data
        assert "created_at" in data
        assert "num_windows" in data
        assert data["num_windows"] == 1
