"""Callback to activate/deactivate adapter layers."""

from typing import Any

from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer import Trainer

from rslearn.log_utils import get_logger

logger = get_logger(__name__)


class ActivateLayers(Callback):
    """Activates adapter layers on a given epoch."""

    def __init__(self, selectors: list[dict[str, Any]]) -> None:
        """Initialize the callback.

        Args:
            selectors: List of selectors to activate.
                Each selector is a dictionary with the following keys:
                - "name": Substring selector of modules to activate (str).
                - "at_epoch": The epoch at which to activate (int).
        """
        self.selectors = selectors

    def on_train_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Activate adapter layers on a given epoch.

        Adapter layers are activated/deactivate by setting the `active` attribute.

        Args:
            trainer: The trainer object.
            pl_module: The LightningModule object.
            batch: The batch of data.
            batch_idx: The index of the batch.
        """
        status = {}
        for name, module in pl_module.named_modules():
            for selector in self.selectors:
                if selector["name"] in name:
                    module.active = trainer.current_epoch >= selector["at_epoch"]
                    status[selector["name"]] = "active" if module.active else "inactive"
        logger.info(f"Updated adapter status: {status}")
