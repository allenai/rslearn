"""Default LightningDataModule for rslearn."""

from typing import Any, Optional

import lightning as L
import torch
from torch.utils.data import DataLoader

from rslearn.dataset import Dataset
from rslearn.train.tasks import Task
from rslearn.utils import FileAPI, parse_file_api_string

from .dataset import DataInput, ModelDataset, SplitConfig


class RslearnDataModule(L.LightningDataModule):
    """Default rslearn LightningDataModule.

    It initializes a ModelDataset based on configured tasks, splits, etc.
    """

    def __init__(
        self,
        inputs: dict[str, DataInput],
        task: Task,
        root_dir: Optional[str] = None,
        file_api: Optional[FileAPI] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        default_config: SplitConfig = SplitConfig(),
        train_config: SplitConfig = SplitConfig(),
        val_config: SplitConfig = SplitConfig(),
        test_config: SplitConfig = SplitConfig(),
    ):
        """Initialize a new RslearnDataModule.

        Args:
            inputs: what to read from the underlying dataset
            task: the task to train on
            root_dir: the root directory of the dataset. One of root_dir or file_api
                must be provided.
            file_api: a FileAPI containing dataset root. One of root_dir or file_api
                must be provided.
            batch_size: the batch size
            num_workers: number of data loader worker processes, or 0 to use main
                process only
            default_config: default split configuration
            train_config: split config for train split
            val_config: split config for val split
            test_config: split config for test split
        """
        super().__init__()
        self.inputs = inputs
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers

        if not file_api:
            file_api = parse_file_api_string(root_dir)
        self.file_api = file_api

        self.split_configs = {
            "train": default_config.update(train_config),
            "val": default_config.update(val_config),
            "test": default_config.update(test_config),
        }

    def setup(self, stage: str):
        """Set up datasets and samplers.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        stage_to_splits = {
            "fit": ["train", "val"],
            "validate": ["val"],
            "test": ["test"],
            "predict": ["test"],
        }
        self.datasets = {}
        for split in stage_to_splits[stage]:
            self.datasets[split] = ModelDataset(
                dataset=Dataset(file_api=self.file_api),
                split_config=self.split_configs[split],
                inputs=self.inputs,
                task=self.task,
                workers=self.num_workers,
            )
            print(f"got {len(self.datasets[split])} examples in split {split}")

    def _get_dataloader(self, split: str) -> DataLoader[dict[str, torch.Tensor]]:
        dataset = self.datasets[split]
        kwargs = dict(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        sampler_factory = self.split_configs[split].sampler
        if sampler_factory:
            kwargs["sampler"] = sampler_factory.get_sampler(dataset)
        elif split == "train":
            kwargs["shuffle"] = True
        return DataLoader(**kwargs)

    def train_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        """Implement one or more PyTorch DataLoaders for training.

        Returns:
            A collection of data loaders specifying training samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        return self._get_dataloader("train")

    def val_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        """Implement one or more PyTorch DataLoaders for validation.

        Returns:
            A collection of data loaders specifying validation samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        return self._get_dataloader("val")

    def test_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        """Implement one or more PyTorch DataLoaders for testing.

        Returns:
            A collection of data loaders specifying testing samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        return self._get_dataloader("test")

    def collate_fn(
        self, batch: list[tuple[dict[str, Any], dict[str, Any]]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Collate batch of training examples.

        We just make list of the inputs and another of the targets.

        Args:
            batch: list of input/target for each example

        Returns:
            a tuple (inputs, targets)
        """
        return tuple(zip(*batch))
