"""Default LightningModule for rslearn."""

import os
from typing import Any, Optional

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MetricCollection

from .tasks import Task


class RslearnLightningModule(L.LightningModule):
    """Default LightningModule for rslearn.

    The loss is computed by provided model while metrics are configured by the provided
    task.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        metrics: dict[str, MetricCollection],
        tasks: dict[str, Task],
        lr: float = 1e-3,
        plateau: bool = False,
        plateau_factor: float = 0.1,
        plateau_patience: int = 10,
        plateau_min_lr: float = 0,
        plateau_cooldown: int = 0,
        visualize_dir: Optional[str] = None,
    ):
        """Initialize a new RslearnLightningModule.

        Args:
            model: the model
            metrics: the metrics for val/test
            tasks: the tasks to train on
            lr: the initial learning rate
            plateau: whether to enable plateau scheduler (default false)
            plateau_factor: on plateau, factor to multiply learning rate by
            plateau_patience: number of iterations with no improvement in val loss
                before reducing learning rate
            plateau_min_lr: minimum learning rate to reduce to
            plateau_cooldown: number of iterations after reducing learning rate before
                resetting plateau scheduler
            visualize_dir: during validation or testing, output visualizations to this
                directory
        """
        super().__init__()
        self.model = model
        self.tasks = tasks
        self.lr = lr
        self.plateau = plateau
        self.plateau_factor = plateau_factor
        self.plateau_patience = plateau_patience
        self.plateau_min_lr = plateau_min_lr
        self.plateau_cooldown = plateau_cooldown
        self.visualize_dir = visualize_dir

        def clone_metrics(prefix: str) -> dict[str, MetricCollection]:
            return torch.nn.ModuleDict(
                {
                    name: metric_collection.clone(prefix=prefix)
                    for name, metric_collection in metrics.items()
                }
            )

        self.val_metrics = clone_metrics(prefix="val_")
        self.test_metrics = clone_metrics(prefix="test_")

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            Optimizer and learning rate scheduler.
        """
        optimizer = AdamW(self.parameters(), lr=self.lr)
        d = dict(
            optimizer=optimizer,
        )
        if self.plateau:
            scheduler = ReduceLROnPlateau(
                optimizer,
                factor=self.plateau_factor,
                patience=self.plateau_patience,
                min_lr=self.plateau_min_lr,
                cooldown=self.plateau_cooldown,
            )
            d["lr_scheduler"] = {
                "scheduler": scheduler,
                "monitor": "val_loss",
            }
        return d

    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Compute the training loss.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            The loss tensor.
        """
        inputs, targets = batch
        batch_size = len(inputs)
        _, loss_dict = self(inputs, targets)
        train_loss = sum(loss_dict.values())
        self.log_dict(
            {"train_" + k: v for k, v in loss_dict.items()},
            batch_size=batch_size,
            prog_bar=True,
        )
        self.log("train_loss", train_loss, batch_size=batch_size)
        return train_loss

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        inputs, targets = batch
        batch_size = len(inputs)
        outputs, loss_dict = self(inputs, targets)
        val_loss = sum(loss_dict.values())
        self.log_dict(
            {"val_" + k: v for k, v in loss_dict.items()}, batch_size=batch_size
        )
        self.log("val_loss", val_loss, batch_size=batch_size, prog_bar=True)
        for name, metric_collection in self.val_metrics.items():
            cur_outputs = outputs[name]
            cur_targets = [target[name] for target in targets]
            metric_collection(cur_outputs, cur_targets)
            self.log_dict(metric_collection, batch_size=batch_size)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        inputs, targets = batch
        batch_size = len(inputs)
        outputs, loss_dict = self(inputs, targets)
        test_loss = sum(loss_dict.values())
        self.log_dict(
            {"test_" + k: v for k, v in loss_dict.items()}, batch_size=batch_size
        )
        self.log("test_loss", test_loss, batch_size=batch_size)
        for name, metric_collection in self.test_metrics.items():
            cur_outputs = outputs[name]
            cur_targets = [target[name] for target in targets]
            metric_collection(cur_outputs, cur_targets)
            self.log_dict(metric_collection, batch_size=batch_size)

        if self.visualize_dir:
            for name, task in self.tasks.items():
                cur_outputs = outputs[name]
                cur_targets = [target[name] for target in targets]
                for idx, (input_dict, output, target) in enumerate(
                    zip(inputs, cur_outputs, cur_targets)
                ):
                    images = task.visualize(input_dict, output, target)
                    for image_suffix, image in images.items():
                        out_fname = os.path.join(
                            self.visualize_dir, f"{batch_idx}_{idx}_{image_suffix}.png"
                        )
                        Image.fromarray(image).save(out_fname)

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Compute the predicted class probabilities.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Output predicted probabilities.
        """
        inputs, _ = batch
        outputs = self(inputs)
        return outputs

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the model.

        Args:
            args: Arguments to pass to model.
            kwargs: Keyword arguments to pass to model.

        Returns:
            Output of the model.
        """
        return self.model(*args, **kwargs)
