"""Default LightningDataModule for rslearn."""

import os
import random
from collections.abc import Iterator
from typing import Any

import lightning as L
import torch
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset
from upath import UPath

from rslearn.dataset import Dataset
from rslearn.train.tasks import Task
from rslearn.train.tasks.multi_task import MultiTask

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

    def _get_dataloader(
        self,
        split: str,
        is_inner_process: bool = False,
        generator: torch.Generator | None = None,
    ) -> DataLoader[dict[str, torch.Tensor]]:
        """Get a dataloader for the given split.

        If is_inner_process is True, then this dataloader is in an inner process, and we
        don't want to spawn nested processes. In this case, we use a generator to shuffle
        the data, and we don't use a sampler. We need a generator to ensure that each
        child dataloader gets the same shuffle, otherwise rank/processes won't synch.

        Args:
            split: the split to get a dataloader for
            is_inner_process: whether this dataloader is in an inner process
            generator: a generator to use for shuffling (only used if is_inner_process is True)
        """
        # is_inner_process is used if we are in a multi dataset setting, and this dataloader
        # is one of a set instantiated PER process (so we don't want to spawn nested processes)
        dataset = self.datasets[split]
        persistent_workers = self.num_workers > 0
        kwargs = dict(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=persistent_workers,
        )
        # Can still shuffle in inner process since we rebuild dataloaders
        should_shuffle = split == "train"
        if is_inner_process:
            assert generator is not None, "generator must be provided for inner process"
            kwargs["num_workers"] = 0
            kwargs["persistent_workers"] = False
            kwargs["shuffle"] = should_shuffle
            kwargs["pin_memory"] = False
            kwargs["generator"] = generator  # type: ignore
            return DataLoader(**kwargs)  # type: ignore

        sampler_factory = self.split_configs[split].sampler
        if sampler_factory:
            kwargs["sampler"] = sampler_factory.get_sampler(dataset)  # type: ignore
        elif (
            self.trainer is not None
            and self.trainer.world_size is not None
            and self.trainer.world_size > 1
        ):
            # Use distributed sampler in case ddp is enabled.
            kwargs["sampler"] = DistributedSampler(  # type: ignore
                dataset,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=should_shuffle,
            )
        else:
            kwargs["shuffle"] = should_shuffle
        return DataLoader(**kwargs)  # type: ignore

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
    """IterableDataset that wraps several RslearnDataModule DataLoaders.

    Wraps several RslearnDataModule DataLoaders and emits batches in random order
    with correct sharding across DDP ranks (GPU) and DataLoader workers (CPU).
    Manual sharding only (seems to be faster).

    Basic data flow:
    1. Build individual RslearnDataModule instances from dataset configs
    2. Build a MultiWrapperDataset from the DataLoaders of these RslearnDataModule instances
    3. Wrap the MultiWrapperDataset in a DataLoader shell
    4. Pass the DataLoader to the LightningDataModule and use as normal
    """

    def __init__(
        self,
        data_modules: dict[str, "RslearnDataModule"],
        rank: int,
        world_size: int,
        split: str = "train",
        epoch: int = 0,
    ) -> None:
        """Initialize a new MultiWrapperDataset.

        Args:
            data_modules: dict mapping dataset names to RslearnDataModule objects
            rank: the rank of the current process
            world_size: the total number of processes
            split: "train", "val", "test", or "predict"
            epoch: the current epoch
        """
        super().__init__()
        self.data_modules = data_modules
        self.split = split
        self.rank = rank
        self.world_size = world_size
        self.epoch = epoch

    def _build_dataloaders(self, seed: int = 0) -> list[DataLoader]:
        """Build the dataloaders for the given split.

        Lightning will call this every epoch when it re‑creates the DataLoaders.
        Note that the base seed should be consistent across all ranks/processes,
        but is modified for each spawn's dataloader.

        The seed argument is only used for shuffling (needs to be identical across
        all the child dataloaders), but is not used to seed transforms.

        Args:
            seed: a seed to use for shuffling across all dataloaders

        Returns:
            A list of DataLoaders for the given split.
        """
        dataloaders = []
        for i, dm in enumerate(self.data_modules.values()):
            generator = torch.Generator()
            generator.manual_seed(seed + i)
            dataloaders.append(
                dm._get_dataloader(
                    self.split,
                    is_inner_process=True,
                    generator=generator,
                )
            )
        return dataloaders

    def _worker(self) -> tuple[int, int]:
        """Get the worker id and number of workers for the current process."""
        w = torch.utils.data.get_worker_info()
        return (w.id, w.num_workers) if w else (0, 1)

    def __len__(self) -> int:
        """Get the length of the dataset.

        This information is per rank, after sharding. It's not exact (may be off
        due to uneven splits - see docs in __iter__) but is okay for logging.

        Do NOT use this for operations that require exact counts (e.g. anything
        with distributed training)!

        Returns:
            The length of the dataset.
        """
        total_batches = sum(
            int(len(dm.datasets[self.split]) / dm.batch_size)
            for dm in self.data_modules.values()
        )
        return int(total_batches / self.world_size)

    def __iter__(self) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
        """Iterate over the dataset.

        For each GPU rank (process), we have a pool of workers. For each worker in each
        rank, we have a disjoint set of batches we will route to the data queue.
        We need to ensure that 1) each worker per rank gets a unique set of batches, and
        2) all batches are routed, 3) shuffling is consistent across workers in all ranks.

        NOTE: Condition 2 is not really satisfied, since we need to enforce exact batch
        counts across ranks/workers to prevent hangs. There might be a few dropped batches.

        Returns:
            An iterator over the dataset for this particular rank/worker.
        """
        # Set a consistent seed across all ranks and all workers
        rng = random.Random(self.epoch)
        loaders = self._build_dataloaders(self.epoch)
        total_raw_batches = sum(len(dl) for dl in loaders)

        # Compute this rank/worker's id compared to the global batch counter
        wid, num_w = self._worker()
        my_id = self.rank * num_w + wid
        parts = self.world_size * num_w
        iters = [iter(dl) for dl in loaders]
        local_yielded = 0
        max_local = total_raw_batches // parts
        self.epoch += 1

        # Only yield batches that belong to this rank/worker
        for global_idx in range(total_raw_batches):
            if not iters or local_yielded >= max_local:
                # Ensure that all ranks/workers yield the same number of batches
                # Otherwise, NCCL hangs waiting for an all-reduce (gather?) even
                # though some dataloaders have already quit the yield loop
                break
            batch = None
            while batch is None and iters:
                child_i = rng.randrange(len(iters))
                try:
                    batch = next(iters[child_i])
                except StopIteration:
                    iters.pop(child_i)
            if batch is not None and global_idx % parts == my_id:
                yield batch
                local_yielded += 1


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
        task: MultiTask,
        max_num_workers: int = 32,
        **kwargs: Any,
    ) -> None:
        """Initialize a new MultiDatasetDataModule.

        Args:
            dataset_configs: dict mapping dataset names to RslearnDataModule objects
            task: the task to train on
            max_num_workers: the maximum number of workers to use for the dataloader
            kwargs: additional keyword arguments
        """
        super().__init__()
        self.data_modules = dataset_configs
        self.max_num_workers = max_num_workers

    def setup(self, stage: str | None = None) -> None:
        """Set up the datasets for the given stage. Also assign dataset-specific names.

        Args:
            stage: The stage to set up ('fit', 'validate', 'test', 'predict')
        """
        for name, data_module in self.data_modules.items():
            data_module.setup(stage)  # type: ignore
            data_module.set_name(name)

    def _get_dataloader(self, split: str) -> DataLoader[dict[str, torch.Tensor]]:
        num_workers = min(len(os.sched_getaffinity(0)), self.max_num_workers)
        print(f"INFO: using num_workers={num_workers} for {split} split")
        print(f"INFO: restarting shuffling from epoch {self.trainer.current_epoch}")  # type: ignore
        return DataLoader(
            dataset=MultiWrapperDataset(
                self.data_modules,
                split=split,
                rank=self.trainer.global_rank,  # type: ignore
                world_size=self.trainer.world_size,  # type: ignore
                epoch=self.trainer.current_epoch,  # type: ignore
            ),
            batch_size=None,
            pin_memory=True,
            num_workers=num_workers,
            shuffle=False,  # already randomly chose next dataloader + per-dataloader shuffler
            persistent_workers=True,  # won't do much since we rebuild dataloaders on each epoch
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
