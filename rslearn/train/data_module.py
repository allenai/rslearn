"""Default LightningDataModule for rslearn."""

import math
import random
from collections.abc import Iterator
from typing import Any

import lightning as L
import torch
from torch.utils.data import DataLoader, DistributedSampler
from upath import UPath

from rslearn.dataset import Dataset
from rslearn.train.tasks import Task

from .dataset import DataInput, ModelDataset, MultiDataset, RetryDataset, SplitConfig


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

    def _get_dataloader(
        self,
        split: str,
    ) -> DataLoader[dict[str, torch.Tensor]]:
        """Get a dataloader for the given split.

        Args:
            split: the split to get a dataloader for
        """
        dataset = self.datasets[split]
        persistent_workers = self.num_workers > 0
        kwargs: dict[str, Any] = dict(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=persistent_workers,
        )
        should_shuffle = split == "train"

        sampler_factory = self.split_configs[split].sampler
        if sampler_factory:
            kwargs["sampler"] = sampler_factory.get_sampler(dataset)
        elif (
            self.trainer is not None
            and self.trainer.world_size is not None
            and self.trainer.world_size > 1
        ):
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
        data_modules: dict[str, RslearnDataModule],
        num_workers: int = 32,
        **kwargs: Any,
    ) -> None:
        """Initialize a new MultiDatasetDataModule.

        Args:
            data_modules: dict mapping dataset names to RslearnDataModule objects
            num_workers: the maximum number of workers to use for the dataloader
            kwargs: additional keyword arguments
        """
        super().__init__()
        self.data_modules = data_modules
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        """Set up the datasets for the given stage. Also assign dataset-specific names.

        Args:
            stage: The stage to set up ('fit', 'validate', 'test', 'predict')
        """
        for name, data_module in self.data_modules.items():
            data_module.setup(stage)  # type: ignore
            data_module.set_name(name)

    def _get_dataloader(self, split: str) -> DataLoader[dict[str, torch.Tensor]]:
        datasets = {name: dm.datasets[split] for name, dm in self.data_modules.items()}
        batch_size = max(
            self.data_modules.values(), key=lambda dm: dm.batch_size
        ).batch_size
        print(f"INFO: using batch_size {batch_size} for split {split}")
        dataset = MultiDataset(datasets)
        return DataLoader(
            dataset=dataset,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=collate_fn,
            batch_sampler=DistributedPerDatasetBatchSampler(
                multi_dataset=dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_replicas=self.trainer.world_size,  # type: ignore
                rank=self.trainer.global_rank,  # type: ignore
            ),
        )

    def train_dataloader(self) -> DataLoader:
        """Get the training dataloader."""
        return self._get_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        """Get the validation dataloader."""
        return self._get_dataloader("val")

    def test_dataloader(self) -> DataLoader:
        """Get the test dataloader."""
        return self._get_dataloader("test")

    def predict_dataloader(self) -> DataLoader:
        """Get the predict dataloader."""
        return self._get_dataloader("predict")


class DistributedPerDatasetBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Distributed batch sampler yielding batches from one sub-dataset per batch.

    Wraps torch DistributedSampler to first split indices across ranks,
    then does "one-subdataset-per-batch" sampling in each process.
    """

    def __init__(
        self,
        multi_dataset: MultiDataset,
        batch_size: int,
        shuffle: bool = True,
        num_replicas: int | None = None,
        rank: int | None = None,
    ) -> None:
        """Initialize a new DistributedPerDatasetBatchSampler.

        Args:
            multi_dataset: the MultiDataset to sample from
            batch_size: the batch size
            shuffle: whether to shuffle the indices
            num_replicas: the number of replicas
            rank: the rank
        """
        self.multi_dataset = multi_dataset
        self.batch_size = batch_size
        self.epoch = 0
        # Using one DistributedSampler per dataset guarantees equal splitting
        # across all datasets across all ranks
        self.dist_samplers = {
            name: DistributedSampler(
                dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=shuffle,
                drop_last=False,
            )
            for name, dataset in multi_dataset.datasets.items()
        }

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for the distributed sampler.

        Args:
            epoch: the epoch to set
        """
        self.epoch = epoch
        for dist_sampler in self.dist_samplers.values():
            dist_sampler.set_epoch(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        """Iterate over the batches."""
        # Get the per-rank, per-epoch list of properly offset multi-dataset indices
        partitioned: dict[str, list[int]] = {}
        for name, sampler in self.dist_samplers.items():
            offset = self.multi_dataset.buckets[name].start
            partitioned[name] = [idx + offset for idx in list(iter(sampler))]

        # Seed is shared aross all ranks but shuffled per epoch
        rng = random.Random(self.epoch)

        # Randomly pick a non-empty (or sufficiently full) bucket each time
        # NOTE: this samples uniformly across all datasets, which may or may not be desirable
        while True:
            available = [name for name, idxs in partitioned.items() if idxs]
            if not available:
                break

            name = rng.choice(available)
            idxs = partitioned[name]

            if len(idxs) >= self.batch_size:
                batch, partitioned[name] = (
                    idxs[: self.batch_size],
                    idxs[self.batch_size :],
                )
            else:
                batch, partitioned[name] = idxs[:], []

            yield batch

    def __len__(self) -> int:
        """Return the number of batches."""
        return sum(
            math.ceil(len(sampler) / self.batch_size)
            for sampler in self.dist_samplers.values()
        )
