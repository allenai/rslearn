import lightning.pytorch as pl
import pytest
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

from rslearn.models.singletask import SingleTaskModel
from rslearn.train.callbacks.freeze_unfreeze import FreezeUnfreeze
from rslearn.train.data_module import RslearnDataModule
from rslearn.train.lightning_module import RslearnLightningModule

INITIAL_LR = 1e-3


class RecordParamsCallback(pl.Callback):
    def __init__(self) -> None:
        self.recorded_params: list = []

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: RslearnLightningModule
    ) -> None:
        self.recorded_params.append(
            pl_module.model.encoder[0].model.features[0][0].weight.tolist()
        )


class LMWithCustomPlateau(RslearnLightningModule):
    """RslearnLightningModule but adjust the plateau scheduler if it is enabled.

    Specifically, set threshold to negative value so that plateau is triggered on every
    epoch.
    """

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            Optimizer and learning rate scheduler.
        """
        d = super().configure_optimizers()
        if "lr_scheduler" in d:
            # Plateau scheduler will be set up with relative mode.
            # We set threshold to 1 so it should activate plateau on every epoch.
            # It should plateau unless:
            #     cur_train_loss < best_train_loss * (1 - threshold)
            d["lr_scheduler"]["scheduler"].threshold = 1
        return d


def test_freeze_unfreeze(
    image_to_class_data_module: RslearnDataModule, image_to_class_model: SingleTaskModel
) -> None:
    """Test the FreezeUnfreeze callback by making sure the weights don't change in the
    first epoch but then unfreeze and do change in the second epoch."""
    pl_module = LMWithCustomPlateau(
        model=image_to_class_model,
        task=image_to_class_data_module.task,
        print_parameters=True,
        lr=INITIAL_LR,
    )
    freeze_unfreeze = FreezeUnfreeze(
        module_selector=["model", "encoder"],
        unfreeze_at_epoch=1,
    )
    record_callback = RecordParamsCallback()
    trainer = pl.Trainer(
        max_epochs=3,
        callbacks=[
            freeze_unfreeze,
            record_callback,
        ],
    )
    trainer.fit(pl_module, datamodule=image_to_class_data_module)
    assert record_callback.recorded_params[0] == record_callback.recorded_params[1]
    assert record_callback.recorded_params[0] != record_callback.recorded_params[2]


def test_unfreeze_lr_factor(
    image_to_class_data_module: RslearnDataModule, image_to_class_model: SingleTaskModel
) -> None:
    """Make sure learning rate is set correctly after unfreezing."""
    plateau_factor = 0.5
    unfreeze_lr_factor = 3
    pl_module = LMWithCustomPlateau(
        model=image_to_class_model,
        task=image_to_class_data_module.task,
        print_parameters=True,
        lr=INITIAL_LR,
        plateau=True,
        plateau_factor=plateau_factor,
        plateau_patience=0,
    )
    freeze_unfreeze = FreezeUnfreeze(
        module_selector=["model", "encoder"],
        unfreeze_at_epoch=1,
        unfreeze_lr_factor=unfreeze_lr_factor,
    )
    trainer = pl.Trainer(
        max_epochs=2,
        callbacks=[freeze_unfreeze],
    )
    trainer.fit(pl_module, datamodule=image_to_class_data_module)
    param_groups = trainer.optimizers[0].param_groups
    # Default parameters should undergo two plateaus.
    assert param_groups[0]["lr"] == pytest.approx(INITIAL_LR * (plateau_factor**2))
    # Other one should be affected by two plateaus + the unfreeze factor.
    assert param_groups[1]["lr"] == pytest.approx(
        INITIAL_LR * (plateau_factor**2) / unfreeze_lr_factor
    )
