"""LightningDataModule for YieldSAT preprocessed Zarr stores."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import lightning as L
import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from rslearn.train.data_module import collate_fn
from rslearn.train.model_context import SampleMetadata
from rslearn.train.tasks import Task
from rslearn.utils.geometry import WGS84_PROJECTION


def _open_zarr_kwargs(
    storage_options: dict[str, Any] | None,
    consolidated: bool | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "consolidated": consolidated,
        "chunks": None,
    }
    if storage_options:
        kwargs["storage_options"] = storage_options
    return kwargs


class YieldSatZarrDataset(Dataset):
    """Map-style PyTorch dataset over a YieldSAT Zarr store."""

    def __init__(
        self,
        store: str,
        indices: Sequence[int],
        storage_options: dict[str, Any] | None = None,
        consolidated: bool | None = True,
        input_key: str = "sample",
        target_key: str = "target",
        normalize: bool = True,
    ) -> None:
        self.store = store
        self.indices = np.asarray(indices, dtype=np.int64)
        self.storage_options = storage_options or {}
        self.consolidated = consolidated
        self.input_key = input_key
        self.target_key = target_key
        self.normalize = normalize
        self._ds: xr.Dataset | None = None
        self._stats_mean: np.ndarray | None = None
        self._stats_std: np.ndarray | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_ds"] = None
        state["_stats_mean"] = None
        state["_stats_std"] = None
        return state

    def _ensure_open(self) -> xr.Dataset:
        if self._ds is None:
            self._ds = xr.open_zarr(
                self.store,
                **_open_zarr_kwargs(self.storage_options, self.consolidated),
            )
            if self.normalize:
                self._stats_mean = np.asarray(self._ds["stats-mean"].values, dtype=np.float32)
                self._stats_std = np.asarray(self._ds["stats-std"].values, dtype=np.float32)
                self._stats_std = np.where(self._stats_std == 0, 1, self._stats_std)
        return self._ds

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(
        self, idx: int
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor], SampleMetadata]:
        ds = self._ensure_open()
        source_idx = int(self.indices[idx])

        sample = np.asarray(ds[self.input_key].isel(index=source_idx).values, dtype=np.float32)
        if self.normalize:
            if self._stats_mean is None or self._stats_std is None:
                raise RuntimeError("normalization stats were not initialized")
            sample = (sample - self._stats_mean[None, :]) / self._stats_std[None, :]

        target = float(ds[self.target_key].isel(index=source_idx).values)
        sample_id = str(ds["index"].isel(index=source_idx).values)

        inputs = {
            self.input_key: torch.from_numpy(sample.copy()),
        }
        targets = {
            "value": torch.tensor(target, dtype=torch.float32),
            "valid": torch.tensor(np.isfinite(target), dtype=torch.float32),
        }
        metadata = SampleMetadata(
            window_group="yieldsat",
            window_name=sample_id,
            window_bounds=(0, 0, 1, 1),
            crop_bounds=(0, 0, 1, 1),
            crop_idx=0,
            num_crops_in_window=1,
            time_range=None,
            projection=WGS84_PROJECTION,
            dataset_source="yieldsat_zarr",
        )
        return inputs, targets, metadata


class YieldSatZarrDataModule(L.LightningDataModule):
    """DataModule that feeds YieldSAT Zarr samples into rslearn models.

    This intentionally bypasses rslearn's GeoTIFF/window materialization path. It is
    for the preprocessed benchmark tensor layout where each row already contains a
    complete sample time series and target.
    """

    def __init__(
        self,
        store: str,
        task: Task,
        storage_options: dict[str, Any] | None = None,
        consolidated: bool | None = True,
        batch_size: int = 64,
        num_workers: int = 0,
        input_key: str = "sample",
        target_key: str = "target",
        normalize: bool = True,
        split_indices_path: str | None = None,
        train_fraction: float = 0.8,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        split_seed: int = 0,
        split_group_var: str | None = "field_shared_name",
        predict_split: str = "test",
    ) -> None:
        super().__init__()
        self.store = store
        self.task = task
        self.storage_options = storage_options or {}
        self.consolidated = consolidated
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_key = input_key
        self.target_key = target_key
        self.normalize = normalize
        self.split_indices_path = split_indices_path
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.split_seed = split_seed
        self.split_group_var = split_group_var
        self.predict_split = predict_split
        self.datasets: dict[str, YieldSatZarrDataset] = {}

    def _split_positions(self, positions: np.ndarray) -> dict[str, list[int]]:
        train_end = int(round(len(positions) * self.train_fraction))
        val_end = train_end + int(round(len(positions) * self.val_fraction))
        return {
            "train": positions[:train_end].tolist(),
            "val": positions[train_end:val_end].tolist(),
            "test": positions[val_end:].tolist(),
        }

    def _load_split_indices(self, ds: xr.Dataset) -> dict[str, list[int]]:
        if self.split_indices_path:
            with open(self.split_indices_path) as f:
                data = json.load(f)
            return {k: [int(v) for v in vals] for k, vals in data.items()}

        total = self.train_fraction + self.val_fraction + self.test_fraction
        if not np.isclose(total, 1.0):
            raise ValueError("train_fraction + val_fraction + test_fraction must equal 1")

        rng = np.random.default_rng(self.split_seed)
        n = int(ds.sizes["index"])

        if self.split_group_var and self.split_group_var in ds:
            group_values = np.asarray(ds[self.split_group_var].values)
            unique_groups = rng.permutation(np.unique(group_values))
            group_splits = self._split_positions(unique_groups)
            return {
                split: np.flatnonzero(np.isin(group_values, groups)).tolist()
                for split, groups in group_splits.items()
            }

        return self._split_positions(rng.permutation(n))

    def setup(self, stage: str) -> None:
        ds = xr.open_zarr(
            self.store,
            **_open_zarr_kwargs(self.storage_options, self.consolidated),
        )
        splits = self._load_split_indices(ds)
        ds.close()

        stage_splits = {
            "fit": ["train", "val"],
            "validate": ["val"],
            "test": ["test"],
            "predict": [self.predict_split],
        }[stage]

        self.datasets = {}
        for split in stage_splits:
            self.datasets[split] = YieldSatZarrDataset(
                store=self.store,
                indices=splits[split],
                storage_options=self.storage_options,
                consolidated=self.consolidated,
                input_key=self.input_key,
                target_key=self.target_key,
                normalize=self.normalize,
            )

    def _get_dataloader(self, split: str, shuffle: bool) -> DataLoader:
        dataset = self.datasets[split]
        kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "collate_fn": collate_fn,
            "persistent_workers": self.num_workers > 0,
        }

        if (
            self.trainer is not None
            and self.trainer.world_size is not None
            and self.trainer.world_size > 1
        ):
            kwargs["sampler"] = DistributedSampler(
                dataset,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=shuffle,
            )
        else:
            kwargs["shuffle"] = shuffle

        return DataLoader(**kwargs)

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader("test", shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.predict_split, shuffle=False)
