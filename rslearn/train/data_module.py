"""Default LightningDataModule for rslearn."""

import random
from typing import Any

import lightning as L
import torch
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset
from upath import UPath

from rslearn.dataset import Dataset
from rslearn.train.tasks import Task
from rslearn.train.tasks.multi_task import MultiDatasetTask

from .dataset import DataInput, ModelDataset, RetryDataset, SplitConfig


def collate_fn(
    batch: list[tuple[dict[str, Any], dict[str, Any]]],
) -> tuple:
    """Collate batch of training examples.

    We just make list of the inputs and another of the targets.

    Args:
        batch: list of input/target for each example

    Returns:
        a tuple (inputs, targets)
    """
    return tuple(zip(*batch))


class RslearnDataModule(L.LightningDataModule):
    """Default rslearn LightningDataModule.

    It initializes a ModelDataset based on configured tasks, splits, etc.
    """

    def __init__(
        self,
        inputs: dict[str, DataInput],
        task: Task,
        path: str,
        path_options: dict[str, Any] = {},
        batch_size: int = 1,
        num_workers: int = 0,
        default_config: SplitConfig = SplitConfig(),
        train_config: SplitConfig = SplitConfig(),
        val_config: SplitConfig = SplitConfig(),
        test_config: SplitConfig = SplitConfig(),
        predict_config: SplitConfig = SplitConfig(),
        name: str | None = None,
    ) -> None:
        """Initialize a new RslearnDataModule.

        Args:
            inputs: what to read from the underlying dataset
            task: the task to train on
            path: the dataset path.
            path_options: additional options for path to pass to fsspec.
            batch_size: the batch size
            num_workers: number of data loader worker processes, or 0 to use main
                process only
            default_config: default split configuration
            train_config: split config for train split
            val_config: split config for val split
            test_config: split config for test split
            predict_config: split config for predict split
            name: name of the dataset (default: None)
        """
        super().__init__()
        self.inputs = inputs
        self.task = task
        self.path = UPath(path, **path_options)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.name = name
        self.split_configs = {
            "train": default_config.update(train_config),
            "val": default_config.update(val_config),
            "test": default_config.update(test_config),
            "predict": default_config.update(predict_config),
        }

    def setup(self, stage: str) -> None:
        """Set up datasets and samplers.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        stage_to_splits = {
            "fit": ["train", "val"],
            "validate": ["val"],
            "test": ["test"],
            "predict": ["predict"],
        }
        self.datasets = {}
        for split in stage_to_splits[stage]:
            dataset = ModelDataset(
                dataset=Dataset(path=self.path),
                split_config=self.split_configs[split],
                inputs=self.inputs,
                task=self.task,
                workers=self.num_workers,
                name=self.name,
            )
            dataset = RetryDataset(dataset)
            self.datasets[split] = dataset
            print(f"got {len(self.datasets[split])} examples in split {split}")

    def set_name(self, name: str) -> None:
        """Set the name of the dataset.

        Args:
            name: the name of the dataset
        """
        self.name = name
        for dataset in self.datasets.values():
            dataset.set_name(name)

    def _get_dataloader(self, split: str) -> DataLoader[dict[str, torch.Tensor]]:
        dataset = self.datasets[split]
        persistent_workers = self.num_workers > 0
        kwargs = dict(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=persistent_workers,
        )
        sampler_factory = self.split_configs[split].sampler
        should_shuffle = split == "train"
        if sampler_factory:
            kwargs["sampler"] = sampler_factory.get_sampler(dataset)
        elif (
            self.trainer is not None
            and self.trainer.world_size is not None
            and self.trainer.world_size > 1
        ):
            print(
                f"INFO: using distributed sampler with {self.trainer.world_size} replicas and rank {self.trainer.global_rank}"
            )
            # NOTE: when doing single-gpu multi-dataset training, self.trainer is None
            # unsure if this is only for single-gpu setting or multi-dataset
            # is broken for distributed traning
            # Use distributed sampler in case ddp is enabled.
            kwargs["sampler"] = DistributedSampler(
                dataset,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=should_shuffle,
            )
        else:
            kwargs["shuffle"] = should_shuffle
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

    def predict_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        """Implement one or more PyTorch DataLoaders for testing.

        Returns:
            A collection of data loaders specifying predicting samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        return self._get_dataloader("predict")


class MultiWrapperDataset(IterableDataset):
    """Multi-dataset data module for training on multiple datasets with different modalities.
    Basic data flow:
    1. Build individual RslearnDataModule instances from dataset configs
    2. Build a MultiWrapperDataset from the DataLoaders of these RslearnDataModule instances
    3. Wrap the MultiWrapperDataset in a DataLoader shell
    4. Pass the DataLoader to the LightningDataModule and use as normal
    """

    def __init__(
        self,
        dataloaders: list[DataLoader],
        datasets: list[str],
        strategy: str = "random",
    ):
        """Args:
        dataloaders: list of DataLoader objects
        tasks: list of dataset names, one for each dataloader
        strategy: "random" or "round_robin"
        """
        assert len(dataloaders) == len(
            datasets
        ), "number of dataloaders and tasks must match"
        self.dataloaders = dataloaders
        self.strategy = strategy
        self.datasets = datasets.copy()

    def __len__(self):
        return sum(len(dl) for dl in self.dataloaders)

    def __iter__(self):
        self.iterators = [iter(dl) for dl in self.dataloaders]
        datasets = self.datasets.copy()  # allow restart on new iter call
        while True:
            if self.strategy == "random":
                idx = random.randint(0, len(self.iterators) - 1)
            elif self.strategy == "round_robin":
                idx = getattr(self, "last_idx", -1) + 1
                idx %= len(self.iterators)
                self.last_idx = idx
            else:
                raise ValueError("Unknown strategy")

            try:
                batch = next(self.iterators[idx])
                for instance in batch[0]:  # modify the inputs directly
                    instance["dataset_source"] = datasets[idx]
                yield batch
            except StopIteration:
                self.iterators.pop(idx)
                datasets.pop(idx)
                if len(self.iterators) == 0:
                    break
                if self.strategy == "round_robin":
                    self.last_idx -= 1


class MultiDatasetDataModule(L.LightningDataModule):
    """Data module that manages multiple RslearnDataModule instances.

    This module creates and manages multiple RslearnDataModule instances, each handling
    a different dataset. It provides a unified interface for training on multiple datasets
    with different modalities and labels.

    Each dataset can have different:
    - Input modalities (e.g., Sentinel-2 vs Landsat)
    - Label schemas (e.g., different classification classes)
    - Task types (e.g., classification vs detection)
    - Transforms and preprocessing
    """

    def __init__(
        self,
        dataset_configs: dict[str, RslearnDataModule],
        task: MultiDatasetTask,
        **kwargs,
    ):
        super().__init__()
        self.datasets = list(dataset_configs.keys())
        self.data_modules = dataset_configs

    def setup(self, stage: str | None = None):
        """Set up the datasets for the given stage. Also assign dataset-specific names.

        Args:
            stage: The stage to set up ('fit', 'validate', 'test', 'predict')
        """
        for name, data_module in self.data_modules.items():
            data_module.setup(stage)  # type: ignore
            data_module.set_name(name)

    def _get_dataloader(self, split: str) -> DataLoader[dict[str, torch.Tensor]]:
        dataloaders = []
        for data_module in self.data_modules.values():
            dataloaders.append(data_module._get_dataloader(split))
        dataset = MultiWrapperDataset(
            dataloaders,
            self.datasets,
            strategy="random" if split == "train" else "round_robin",
            # ensure that during testing, we see all datasets
        )
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=0,  # handle splitting/multiprocessing in RslearnDataModule
            pin_memory=True,  # unclear if we need this, haven't checked properly
            shuffle=False,
        )

    def train_dataloader(self) -> DataLoader:
        """Get the training dataloader.

        Returns:
            A DataLoader for the training set.
        """
        return self._get_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        """Get the validation dataloader.

        Returns:
            A DataLoader for the validation set.
        """
        return self._get_dataloader("val")

    def test_dataloader(self) -> DataLoader:
        """Get the test dataloader.

        Returns:
            A DataLoader for the test set.
        """
        return self._get_dataloader("test")

    def predict_dataloader(self) -> DataLoader:
        """Get the predict dataloader.

        Returns:
            A DataLoader for the predict set.
        """
        return self._get_dataloader("predict")
