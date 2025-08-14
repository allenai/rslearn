"""FreezeUnfreeze callback."""

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import BaseFinetuning
from lightning.pytorch.callbacks.stochastic_weight_avg import StochasticWeightAveraging
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer

from rslearn.log_utils import get_logger

logger = get_logger(__name__)


class FreezeUnfreeze(BaseFinetuning):
    """Freezes a module and optionally unfreezes it after a number of epochs."""

    def __init__(
        self,
        module_selector: list[str | int],
        unfreeze_at_epoch: int | None = None,
        unfreeze_lr_factor: float = 1,
    ) -> None:
        """Creates a new FreezeUnfreeze.

        Args:
            module_selector: list of keys to access the target module to freeze. For
                example, the selector for backbone.encoder is ["backbone", "encoder"].
            unfreeze_at_epoch: optionally unfreeze the target module after this many
                epochs.
            unfreeze_lr_factor: if unfreezing, how much lower to set the learning rate
                of this module compared to the default learning rate after unfreezing,
                e.g. 10 to set it 10x lower. Default is 1 to use same learning rate.
        """
        super().__init__()
        self.module_selector = module_selector
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.unfreeze_lr_factor = unfreeze_lr_factor
        if unfreeze_at_epoch == 0:
            raise ValueError("unfreeze_at_epoch cannot be 0")

    def _get_target_module(self, pl_module: LightningModule) -> torch.nn.Module:
        target_module = pl_module
        for k in self.module_selector:
            if isinstance(k, int):
                target_module = target_module[k]
            else:
                target_module = getattr(target_module, k)
        return target_module

    def freeze_before_training(self, pl_module: LightningModule) -> None:
        """Freeze the model at the beginning of training.

        Args:
            pl_module: the LightningModule.
        """
        logger.info(f"freezing model at {self.module_selector}")
        self.freeze(self._get_target_module(pl_module))

    def finetune_function(
        self, pl_module: LightningModule, current_epoch: int, optimizer: Optimizer
    ) -> None:
        """Check whether we should unfreeze the model on each epoch.

        Args:
            pl_module: the LightningModule.
            current_epoch: the current epoch number.
            optimizer: the optimizer
        """
        if self.unfreeze_at_epoch is None:
            return
        elif current_epoch == self.unfreeze_at_epoch:
            logger.info(
                f"unfreezing model at {self.module_selector} since we are on epoch {current_epoch}"
            )
            self.unfreeze_and_add_param_group(
                modules=self._get_target_module(pl_module),
                optimizer=optimizer,
                initial_denom_lr=self.unfreeze_lr_factor,
            )
            if "scheduler" in pl_module.schedulers:
                scheduler = pl_module.schedulers["scheduler"]
                if isinstance(scheduler, ReduceLROnPlateau):
                    while len(scheduler.min_lrs) < len(optimizer.param_groups):
                        logger.info(
                            "appending to ReduceLROnPlateau scheduler min_lrs for unfreeze"
                        )
                        scheduler.min_lrs.append(scheduler.min_lrs[0])
            for callback in pl_module.trainer.callbacks:
                if isinstance(callback, StochasticWeightAveraging):
                    if isinstance(callback._swa_lrs, list):
                        assert (
                            len(callback._swa_lrs) == 1
                        ), "only one swa lr is supported"
                        swa_lr = callback._swa_lrs[0]
                    elif isinstance(callback._swa_lrs, float):
                        swa_lr = callback._swa_lrs
                    else:
                        raise ValueError(
                            f"unknown swa lr type: {type(callback._swa_lrs)}"
                        )
                    for param_group in optimizer.param_groups:
                        param_group["swa_lr"] = swa_lr
                    logger.info(f"setting all swa lrs to {swa_lr}")
        elif current_epoch > self.unfreeze_at_epoch:
            # always do this because overhead is minimal, and it allows restoring
            # from a checkpoint (resuming a run) without messing up unfreezing
            BaseFinetuning.make_trainable(self._get_target_module(pl_module))
