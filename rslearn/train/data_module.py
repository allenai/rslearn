"""Default LightningDataModule for rslearn."""

from typing import Any, Optional

import lightning as L
import torch
from torch.utils.data import DataLoader

from rslearn.train.tasks import Task

from .dataset import DatasetConfig, ModelDataset, TaskConfig


class SamplerFactory:
    """Factory to produce a Sampler.

    This enables configuring a sampler without needing to pass the dataset.
    """

    def get_sampler(
        self, dataset: torch.utils.data.Dataset
    ) -> torch.utils.data.Sampler:
        """Create a sampler for the given dataset.

        Args:
            dataset: the dataset

        Returns:
            a sampler
        """
        raise NotImplementedError


class RandomSamplerFactory(SamplerFactory):
    """A sampler factory for RandomSampler."""

    def __init__(self, replacement: bool = False, num_samples: Optional[int] = None):
        """Initialize a RandomSamplerFactory.

        Args:
            replacement: whether to pick with replacement, default false
            num_samples: optional number of dataset samples to limit iteration to,
                otherwise picks random samples equal to the dataset size
        """
        self.replacement = replacement
        self.num_samples = num_samples

    def get_sampler(
        self, dataset: torch.utils.data.Dataset
    ) -> torch.utils.data.Sampler:
        """Create a sampler for the given dataset.

        Args:
            dataset: the dataset

        Returns:
            a RandomSampler
        """
        return torch.utils.data.RandomSampler(
            dataset, replacement=self.replacement, num_samples=self.num_samples
        )


class RslearnDataModule(L.LightningDataModule):
    """Default rslearn LightningDataModule.

    It initializes a ModelDataset based on configured tasks, splits, etc.
    """

    def __init__(
        self,
        tasks: dict[str, Task],
        task_config: TaskConfig,
        dataset_config: DatasetConfig,
        batch_size: int = 1,
        num_workers: int = 0,
        train_sampler: Optional[SamplerFactory] = None,
        val_sampler: Optional[SamplerFactory] = None,
    ):
        """Initialize a new RslearnDataModule.

        Args:
            tasks: the Tasks to train on
            task_config: specification of the data types of inputs and targets to read
            dataset_config: which rslearn dataset(s) and layer(s) to read from
            batch_size: the batch size
            num_workers: number of data loader worker processes, or 0 to use main
                process only
            train_sampler: SamplerFactor for training
            val_sampler: SamplerFactory for validation
        """
        super().__init__()
        self.tasks = tasks
        self.task_config = task_config
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sampler_factories = {}
        if train_sampler:
            self.sampler_factories["train"] = train_sampler
        if val_sampler:
            self.sampler_factories["val"] = val_sampler

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
                self.tasks, self.task_config, self.dataset_config, split
            )

    def _get_dataloader(self, split) -> DataLoader[dict[str, torch.Tensor]]:
        dataset = self.datasets[split]
        kwargs = dict(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        if split in self.sampler_factories:
            kwargs["sampler"] = self.sampler_factories[split].get_sampler(dataset)
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
