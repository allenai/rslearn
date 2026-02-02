"""Dataset index for caching window lists to speed up ModelDataset initialization."""

import hashlib
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from upath import UPath

from rslearn.dataset.window import Window
from rslearn.log_utils import get_logger
from rslearn.utils.fsspec import open_atomic

if TYPE_CHECKING:
    from rslearn.dataset.storage.storage import WindowStorage

logger = get_logger(__name__)

# Increment this when the index format changes to force rebuild
INDEX_VERSION = 1

# Directory name for storing index files
INDEX_DIR_NAME = ".rslearn_dataset_index"


class DatasetIndex:
    """Manages indexed window lists for faster ModelDataset initialization.

    Note: The index does NOT automatically detect when windows are added or removed
    from the dataset. Use refresh=True after modifying dataset windows.
    """

    def __init__(
        self,
        storage: "WindowStorage",
        dataset_path: UPath,
        groups: list[str] | None,
        names: list[str] | None,
        tags: dict[str, Any] | None,
        num_samples: int | None,
        skip_targets: bool,
        inputs: dict[str, Any],
    ) -> None:
        """Initialize DatasetIndex with specific configuration.

        Args:
            storage: WindowStorage for deserializing windows.
            dataset_path: Path to the dataset directory.
            groups: list of window groups to include.
            names: list of window names to include.
            tags: tags to filter windows by.
            num_samples: limit on number of samples.
            skip_targets: whether targets are skipped.
            inputs: dict mapping input names to DataInput objects.
        """
        self.storage = storage
        self.dataset_path = dataset_path
        self.index_dir = dataset_path / INDEX_DIR_NAME

        # Compute index key from configuration
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
        self.index_key = hashlib.sha256(
            json.dumps(key_data, sort_keys=True).encode()
        ).hexdigest()

    def _get_config_hash(self) -> str:
        """Get hash of config.json for quick validation.

        Returns:
            A 16-character hex string hash of the config, or empty string if no config.
        """
        config_path = self.dataset_path / "config.json"
        if config_path.exists():
            with config_path.open() as f:
                return hashlib.sha256(f.read().encode()).hexdigest()[:16]
        return ""

    def load_windows(self, refresh: bool = False) -> list[Window] | None:
        """Load indexed window list if valid, else return None.

        Args:
            refresh: If True, ignore existing index and return None.

        Returns:
            List of Window objects if index is valid, None otherwise.
        """
        if refresh:
            logger.info("refresh=True, rebuilding index")
            return None

        index_file = self.index_dir / f"{self.index_key}.json"
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
        if index_data.get("config_hash") != self._get_config_hash():
            logger.info("Config hash mismatch, index invalidated")
            return None

        # Deserialize windows
        return [Window.from_metadata(self.storage, w) for w in index_data["windows"]]

    def save_windows(self, windows: list[Window]) -> None:
        """Save processed windows to index with atomic write.

        Args:
            windows: List of Window objects to index.
        """
        self.index_dir.mkdir(parents=True, exist_ok=True)
        index_file = self.index_dir / f"{self.index_key}.json"

        # Serialize windows
        serialized_windows = [w.get_metadata() for w in windows]

        index_data = {
            "version": INDEX_VERSION,
            "config_hash": self._get_config_hash(),
            "created_at": datetime.now().isoformat(),
            "num_windows": len(windows),
            "windows": serialized_windows,
        }
        with open_atomic(index_file, "w") as f:
            json.dump(index_data, f)
        logger.info(f"Saved {len(windows)} windows to index at {index_file}")
