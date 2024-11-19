"""Default LightningModule for rslearn."""

import os
from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from upath import UPath

from .tasks import Task


class RestoreConfig:
    """Configuration for restoring model parameters.

    This is intended to restore from torch files that are not Lightning checkpoints.
    Only the model parameters state dict are restored, not optimizers and such.
    """

    def __init__(
        self,
        restore_path: str,
        restore_path_options: dict[str, Any] = {},
        selector: list[str] = [],
        ignore_prefixes: list[str] = [],
        remap_prefixes: list[tuple[str, str]] = [],
    ):
        """Create a new RestoreConfig.

        Args:
            restore_path: the filename to restore the file from.
            restore_path_options: additional options for the restore_path to pass to
                fsspec.
            selector: path in the torch dict containing the model parameters.
            ignore_prefixes: prefixes to restore.
            remap_prefixes: list of (old_prefix, new_prefix) to rename parameters
                starting with old_prefix to start with new_prefix instead.
        """
        self.restore_path = UPath(restore_path, **restore_path_options)
        self.selector = selector
        self.ignore_prefixes = ignore_prefixes
        self.remap_prefixes = remap_prefixes

    def get_state_dict(self) -> dict[str, Any]:
        """Returns the state dict configured in this RestoreConfig."""
        print(f"loading state dict from {self.restore_path}")
        with self.restore_path.open("rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        for k in self.selector:
            state_dict = state_dict[k]

        for prefix in self.ignore_prefixes:
            for k in list(state_dict.keys()):
                if not k.startswith(prefix):
                    continue
                del state_dict[k]

        for old_prefix, new_prefix in self.remap_prefixes:
            for k in list(state_dict.keys()):
                if not k.startswith(old_prefix):
                    continue
                new_k = new_prefix + k[len(old_prefix) :]
                v = state_dict[k]
                del state_dict[k]
                state_dict[new_k] = v

        return state_dict


class RslearnLightningModule(L.LightningModule):
    """Default LightningModule for rslearn.

    The loss is computed by provided model while metrics are configured by the provided
    task.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        task: Task,
        lr: float = 1e-3,
        plateau: bool = False,
        plateau_factor: float = 0.1,
        plateau_patience: int = 10,
        plateau_min_lr: float = 0,
        plateau_cooldown: int = 0,
        visualize_dir: str | None = None,
        restore_config: RestoreConfig | None = None,
        print_parameters: bool = False,
        print_model: bool = False,
    ):
        """Initialize a new RslearnLightningModule.

        Args:
            model: the model
            task: the task to train on
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
            restore_config: specification of configuration to restore parameters from
                a non-Lightning checkpoint.
            print_parameters: whether to print the list of model parameters after model
                initialization
            print_model: whether to print the model after model initialization
        """
        super().__init__()
        self.model = model
        self.task = task
        self.lr = lr
        self.plateau = plateau
        self.plateau_factor = plateau_factor
        self.plateau_patience = plateau_patience
        self.plateau_min_lr = plateau_min_lr
        self.plateau_cooldown = plateau_cooldown
        self.visualize_dir = visualize_dir
        self.restore_config = restore_config

        if print_parameters:
            for name, param in self.named_parameters():
                print(name, param.shape)

        if print_model:
            print(self.model)

        self.epochs = 0

        metrics = self.task.get_metrics()
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        self.schedulers: dict = {}

    def on_fit_start(self) -> None:
        """Called when the fit begins."""
        # Only restore if doing a fresh fit.
        if self.trainer.ckpt_path is None and self.restore_config:
            state_dict = self.restore_config.get_state_dict()
            missing_keys, unexpected_keys = self.model.load_state_dict(
                state_dict, strict=False
            )
            if missing_keys or unexpected_keys:
                print(
                    f"warning: restore yielded missing_keys={missing_keys} and unexpected_keys={unexpected_keys}"
                )

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            Optimizer and learning rate scheduler.
        """
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=self.lr)
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
                "monitor": "train_loss",
                "interval": "epoch",
            }
            self.schedulers["plateau"] = scheduler
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
        inputs, targets, _ = batch
        batch_size = len(inputs)
        _, loss_dict = self(inputs, targets)
        train_loss = sum(loss_dict.values())
        self.log_dict(
            {"train_" + k: v for k, v in loss_dict.items()},
            batch_size=batch_size,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_loss",
            train_loss,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
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
        inputs, targets, _ = batch
        batch_size = len(inputs)
        outputs, loss_dict = self(inputs, targets)
        val_loss = sum(loss_dict.values())
        self.log_dict(
            {"val_" + k: v for k, v in loss_dict.items()},
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_loss",
            val_loss,
            batch_size=batch_size,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.val_metrics.update(outputs, targets)
        self.log_dict(self.val_metrics, batch_size=batch_size, on_epoch=True)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        inputs, targets, metadatas = batch
        batch_size = len(inputs)
        outputs, loss_dict = self(inputs, targets)
        test_loss = sum(loss_dict.values())
        self.log_dict(
            {"test_" + k: v for k, v in loss_dict.items()},
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_loss", test_loss, batch_size=batch_size, on_step=False, on_epoch=True
        )
        self.test_metrics.update(outputs, targets)
        self.log_dict(self.test_metrics, batch_size=batch_size, on_epoch=True)

        if self.visualize_dir:
            for idx, (inp, target, output, metadata) in enumerate(
                zip(inputs, targets, outputs, metadatas)
            ):
                images = self.task.visualize(inp, target, output)
                for image_suffix, image in images.items():
                    out_fname = os.path.join(
                        self.visualize_dir,
                        f'{metadata["window_name"]}_{metadata["bounds"][0]}_{metadata["bounds"][1]}_{image_suffix}.png',
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
        inputs, _, _ = batch
        outputs, _ = self(inputs)
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
