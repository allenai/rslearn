"""rslearn dataset class."""

import json
from typing import Any

from upath import UPath

from rslearn.config import DatasetConfig
from rslearn.log_utils import get_logger
from rslearn.template_params import substitute_env_vars_in_string
from rslearn.tile_stores import TileStore, load_tile_store

from .window import Window

logger = get_logger(__name__)


class Dataset:
    """A rslearn dataset.

    Datasets are stored in a directory with the following structure:

    .. code-block:: none

        dataset/
            config.json  # optional, if config provided as runtime object
            windows/
                group1/
                    epsg:3857_10_623565_1528020/
                        metadata.json
                        layers/
                            sentinel2/
                                0_0_tci.tif
                            label/
                                0_0_tci.json
                    ...
                ...

    The dataset loads its configuration and supports actions like prepare, ingest, and
    materialize.
    """

    def __init__(
        self,
        path: UPath,
        disabled_layers: list[str] = [],
        dataset_config: DatasetConfig | None = None,
    ) -> None:
        """Initializes a new Dataset.

        Args:
            path: the root directory of the dataset
            disabled_layers: list of layers to disable
            dataset_config: optional dataset configuration to use instead of loading from the dataset directory
        """
        self.path = path

        if dataset_config is None:
            # Load dataset configuration from the dataset directory.
            with (self.path / "config.json").open("r") as f:
                config_content = f.read()
                config_content = substitute_env_vars_in_string(config_content)
                dataset_config = DatasetConfig.model_validate(
                    json.loads(config_content)
                )

        self.layers = {}
        for layer_name, layer_config in dataset_config.layers.items():
            if layer_name in disabled_layers:
                logger.warning(f"Layer {layer_name} is disabled")
                continue
            self.layers[layer_name] = layer_config

        self.tile_store_config = dataset_config.tile_store
        self.storage = (
            dataset_config.storage.instantiate_window_storage_factory().get_storage(
                self.path
            )
        )

    def load_windows(
        self,
        groups: list[str] | None = None,
        names: list[str] | None = None,
        **kwargs: Any,
    ) -> list[Window]:
        """Load the windows in the dataset.

        Args:
            groups: an optional list of groups to filter loading
            names: an optional list of window names to filter loading
            kwargs: optional keyword arguments to pass to WindowStorage.get_windows.
        """
        return self.storage.get_windows(groups=groups, names=names, **kwargs)

    def get_tile_store(self) -> TileStore:
        """Get the tile store associated with this dataset.

        Returns:
            the TileStore
        """
        return load_tile_store(self.tile_store_config, self.path)
