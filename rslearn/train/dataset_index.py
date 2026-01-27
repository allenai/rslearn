"""Dataset index for caching window lists to speed up ModelDataset initialization."""

import hashlib
import json
from datetime import datetime
from typing import Any

from upath import UPath

from rslearn.log_utils import get_logger

logger = get_logger(__name__)

# Increment this when the index format changes to force rebuild
INDEX_VERSION = 1

# Directory name for storing index files
INDEX_DIR_NAME = ".rslearn_dataset_index"


class DatasetIndex:
    """Manages indexed window lists for faster ModelDataset initialization.

    Note: The index does NOT automatically detect when windows are added or removed
    from the dataset. Use refresh_index=True after modifying dataset windows.
    """

    def __init__(self, dataset_path: UPath) -> None:
        """Initialize DatasetIndex.

        Args:
            dataset_path: Path to the dataset directory.
        """
        self.dataset_path = dataset_path
        self.index_dir = dataset_path / INDEX_DIR_NAME

    def get_index_key(
        self,
        groups: list[str] | None,
        names: list[str] | None,
        tags: dict[str, Any] | None,
        num_samples: int | None,
        skip_targets: bool,
        inputs: dict[str, Any],
    ) -> str:
        """Generate deterministic index key from configuration.

        Args:
            groups: list of window groups to include.
            names: list of window names to include.
            tags: tags to filter windows by.
            num_samples: limit on number of samples.
            skip_targets: whether targets are skipped.
            inputs: dict mapping input names to DataInput objects.

        Returns:
            A 16-character hex string index key.
        """
        # Serialize inputs to dict (extract the relevant fields)
        inputs_data = {}
        for name, inp in inputs.items():
            inputs_data[name] = {
                "layers": inp.layers,
                "required": inp.required,
                "load_all_layers": inp.load_all_layers,
                "is_target": inp.is_target,
            }

        key_data = {
            "groups": groups,
            "names": names,
            "tags": tags,
            "num_samples": num_samples,
            "skip_targets": skip_targets,
            "inputs": inputs_data,
        }
        return hashlib.sha256(
            json.dumps(key_data, sort_keys=True).encode()
        ).hexdigest()[:16]

    def get_config_hash(self) -> str:
        """Get hash of config.json for quick validation.

        Returns:
            A 16-character hex string hash of the config, or empty string if no config.
        """
        config_path = self.dataset_path / "config.json"
        if config_path.exists():
            with config_path.open() as f:
                return hashlib.sha256(f.read().encode()).hexdigest()[:16]
        return ""

    def load_windows(
        self,
        index_key: str,
        refresh_index: bool = False,
    ) -> list[dict[str, Any]] | None:
        """Load indexed window list if valid, else return None.

        Args:
            index_key: The index key to look up.
            refresh_index: If True, ignore existing index and return None.

        Returns:
            List of serialized window dicts if index is valid, None otherwise.
        """
        if refresh_index:
            logger.info("refresh_index=True, rebuilding index")
            return None

        index_file = self.index_dir / f"{index_key}.json"
        if not index_file.exists():
            logger.info(f"No index found at {index_file}, will build")
            return None

        try:
            with index_file.open() as f:
                index_data = json.load(f)
        except (OSError, json.JSONDecodeError):
            logger.warning(f"Corrupted index file at {index_file}, will rebuild")
            return None

        # Check index version
        if index_data.get("version") != INDEX_VERSION:
            logger.info(
                f"Index version mismatch (got {index_data.get('version')}, "
                f"expected {INDEX_VERSION}), will rebuild"
            )
            return None

        # Quick validation: check config hash
        if index_data.get("config_hash") != self.get_config_hash():
            logger.info("Config hash mismatch, index invalidated")
            return None

        return index_data["windows"]

    def save_windows(
        self,
        index_key: str,
        windows: list[dict[str, Any]],
    ) -> None:
        """Save processed windows to index with atomic write.

        Args:
            index_key: The index key to save under.
            windows: List of serialized window dicts to index.
        """
        self.index_dir.mkdir(parents=True, exist_ok=True)
        index_file = self.index_dir / f"{index_key}.json"
        index_data = {
            "version": INDEX_VERSION,
            "config_hash": self.get_config_hash(),
            "created_at": datetime.now().isoformat(),
            "num_windows": len(windows),
            "windows": windows,
        }
        # Atomic write via temp file + rename
        tmp_file = index_file.with_suffix(".tmp")
        with tmp_file.open("w") as f:
            json.dump(index_data, f)
        tmp_file.rename(index_file)
        logger.info(f"Saved {len(windows)} windows to index at {index_file}")
