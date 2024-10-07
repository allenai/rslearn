"""FreezeUnfreeze callback."""

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import BaseFinetuning
from torch.optim.optimizer import Optimizer


class FreezeUnfreeze(BaseFinetuning):
    """Freezes a module and optionally unfreezes it after a number of epochs."""

    def __init__(
        self, module_selector: list[str | int], unfreeze_at_epoch: int | None = None
    ):
        """Creates a new FreezeUnfreeze.

        Args:
            module_selector: list of keys to access the target module to freeze. For
                example, the selector for backbone.encoder is ["backbone", "encoder"].
            unfreeze_at_epoch: optionally unfreeze the target module after this many
                epochs.
        """
        super().__init__()
        self.module_selector = module_selector
        self.unfreeze_at_epoch = unfreeze_at_epoch

    def _get_target_module(self, pl_module: LightningModule) -> torch.nn.Module:
        target_module = pl_module
        for k in self.module_selector:
            if isinstance(k, int):
                target_module = target_module[k]
            else:
                target_module = getattr(target_module, k)
        return target_module

    def freeze_before_training(self, pl_module: LightningModule):
        """Freeze the model at the beginning of training.

        Args:
            pl_module: the LightningModule.
        """
        print(f"freezing model at {self.module_selector}")
        self.freeze(self._get_target_module(pl_module))

    def finetune_function(
        self, pl_module: LightningModule, current_epoch: int, optimizer: Optimizer
    ):
        """Check whether we should unfreeze the model on each epoch.

        Args:
            pl_module: the LightningModule.
            current_epoch: the current epoch number.
            optimizer: the optimizer
        """
        if self.unfreeze_at_epoch is None:
            return
        if current_epoch != self.unfreeze_at_epoch:
            return
        print(
            f"unfreezing model at {self.module_selector} since we are on epoch {current_epoch}"
        )
        self.unfreeze_and_add_param_group(
            modules=self._get_target_module(pl_module),
            optimizer=optimizer,
        )
