"""Unit tests for rslearn.train.dataset_index."""

import json
from pathlib import Path

from upath import UPath

from rslearn.train.dataset_index import INDEX_DIR_NAME, DatasetIndex


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


class MockStorage:
    """Mock WindowStorage for testing."""

    pass


def make_index(
    tmp_path: Path,
    groups: list[str] | None = None,
    names: list[str] | None = None,
    tags: dict | None = None,
    num_samples: int | None = None,
    skip_targets: bool = False,
    inputs: dict | None = None,
) -> DatasetIndex:
    """Helper to create a DatasetIndex with common defaults."""
    if inputs is None:
        inputs = {"image": MockDataInput(["layer1"])}
    return DatasetIndex(
        storage=MockStorage(),  # type: ignore
        dataset_path=UPath(tmp_path),
        groups=groups,
        names=names,
        tags=tags,
        num_samples=num_samples,
        skip_targets=skip_targets,
        inputs=inputs,
    )


class TestDatasetIndex:
    """Test suite for DatasetIndex."""

    def test_index_key_deterministic(self, tmp_path: Path) -> None:
        """Test that index key generation is deterministic."""
        inputs = {
            "image": MockDataInput(["layer1", "layer2"]),
            "label": MockDataInput(["label_layer"], is_target=True),
        }

        index1 = make_index(
            tmp_path,
            groups=["train"],
            tags={"split": "train"},
            num_samples=1000,
            inputs=inputs,
        )
        index2 = make_index(
            tmp_path,
            groups=["train"],
            tags={"split": "train"},
            num_samples=1000,
            inputs=inputs,
        )

        assert index1.index_key == index2.index_key

    def test_index_key_different_configs(self, tmp_path: Path) -> None:
        """Test that different configs produce different index keys."""
        inputs = {"image": MockDataInput(["layer1"])}

        index1 = make_index(tmp_path, groups=["train"], inputs=inputs)
        index2 = make_index(tmp_path, groups=["val"], inputs=inputs)  # Different group
        index3 = make_index(
            tmp_path, groups=["train"], num_samples=100, inputs=inputs
        )  # Different num_samples

        assert index1.index_key != index2.index_key
        assert index1.index_key != index3.index_key
        assert index2.index_key != index3.index_key

    def test_config_hash_with_config(self, tmp_path: Path) -> None:
        """Test config hash when config.json exists."""
        config_content = {"layers": {"test": {"type": "vector"}}}
        with (tmp_path / "config.json").open("w") as f:
            json.dump(config_content, f)

        index = make_index(tmp_path)
        hash1 = index._get_config_hash()

        assert hash1 != ""
        assert len(hash1) == 16

        # Same content should produce same hash
        hash2 = index._get_config_hash()
        assert hash1 == hash2

    def test_config_hash_without_config(self, tmp_path: Path) -> None:
        """Test config hash when config.json doesn't exist."""
        index = make_index(tmp_path)
        assert index._get_config_hash() == ""

    def test_load_nonexistent_index(self, tmp_path: Path) -> None:
        """Test loading returns None when index doesn't exist."""
        index = make_index(tmp_path)
        result = index.load_windows()
        assert result is None

    def test_refresh_ignores_existing(self, tmp_path: Path) -> None:
        """Test that refresh=True ignores existing index."""
        # We can't fully test save/load without real Window objects,
        # but we can test the refresh logic with a manually created index file.
        index = make_index(tmp_path)

        # Create a mock index file
        index_dir = tmp_path / INDEX_DIR_NAME
        index_dir.mkdir(parents=True, exist_ok=True)
        index_file = index_dir / f"{index.index_key}.json"
        with index_file.open("w") as f:
            json.dump(
                {
                    "version": 1,
                    "config_hash": index._get_config_hash(),
                    "num_windows": 0,
                    "windows": [],
                },
                f,
            )

        # With refresh=False, should load
        loaded = index.load_windows(refresh=False)
        assert loaded is not None
        assert len(loaded) == 0

        # With refresh=True, should return None
        loaded = index.load_windows(refresh=True)
        assert loaded is None

    def test_index_invalidated_on_config_change(self, tmp_path: Path) -> None:
        """Test that index is invalidated when config.json changes."""
        # Create initial config
        config1 = {"layers": {"layer1": {"type": "vector"}}}
        with (tmp_path / "config.json").open("w") as f:
            json.dump(config1, f)

        index = make_index(tmp_path)

        # Create a mock index file with the current config hash
        index_dir = tmp_path / INDEX_DIR_NAME
        index_dir.mkdir(parents=True, exist_ok=True)
        index_file = index_dir / f"{index.index_key}.json"
        with index_file.open("w") as f:
            json.dump(
                {
                    "version": 1,
                    "config_hash": index._get_config_hash(),
                    "num_windows": 0,
                    "windows": [],
                },
                f,
            )

        # Should load successfully
        loaded = index.load_windows()
        assert loaded is not None

        # Change config
        config2 = {"layers": {"layer1": {"type": "raster"}}}  # Different type
        with (tmp_path / "config.json").open("w") as f:
            json.dump(config2, f)

        # Should return None (index invalidated due to config hash mismatch)
        loaded = index.load_windows()
        assert loaded is None

    def test_corrupted_index_returns_none(self, tmp_path: Path) -> None:
        """Test that corrupted index file returns None."""
        index = make_index(tmp_path)

        # Create index directory and write invalid JSON
        index_dir = tmp_path / INDEX_DIR_NAME
        index_dir.mkdir(parents=True, exist_ok=True)
        index_file = index_dir / f"{index.index_key}.json"
        with index_file.open("w") as f:
            f.write("not valid json {{{")

        # Should return None, not raise an error
        loaded = index.load_windows()
        assert loaded is None

    def test_version_mismatch_invalidates_index(self, tmp_path: Path) -> None:
        """Test that index with wrong version is invalidated."""
        index = make_index(tmp_path)

        # Create index with wrong version
        index_dir = tmp_path / INDEX_DIR_NAME
        index_dir.mkdir(parents=True, exist_ok=True)
        index_file = index_dir / f"{index.index_key}.json"
        with index_file.open("w") as f:
            json.dump(
                {
                    "version": 999,  # Wrong version
                    "config_hash": "",
                    "num_windows": 1,
                    "windows": [{"name": "test"}],
                },
                f,
            )

        # Should return None due to version mismatch
        loaded = index.load_windows()
        assert loaded is None
