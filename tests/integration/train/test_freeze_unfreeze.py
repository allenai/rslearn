import lightning.pytorch as pl

from rslearn.dataset import Dataset
from rslearn.models.pooling_decoder import PoolingDecoder
from rslearn.models.singletask import SingleTaskModel
from rslearn.models.swin import Swin
from rslearn.train.callbacks.freeze_unfreeze import FreezeUnfreeze
from rslearn.train.data_module import RslearnDataModule
from rslearn.train.dataset import DataInput
from rslearn.train.lightning_module import RslearnLightningModule
from rslearn.train.tasks.classification import ClassificationHead, ClassificationTask


class RecordParamsCallback(pl.Callback):
    def __init__(self):
        self.recorded_params = []

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self.recorded_params.append(
            pl_module.model.encoder[0].model.features[0][0].weight.tolist()
        )


def test_freeze_unfreeze(image_to_class_dataset: Dataset):
    # Test the FreezeUnfreeze callback by making sure the weights don't change in the
    # first epoch but then unfreeze and do change in the second epoch.
    image_data_input = DataInput("raster", ["image"], bands=["band"], passthrough=True)
    target_data_input = DataInput("vector", ["label"])
    task = ClassificationTask("label", ["cls0", "cls1"], read_class_id=True)
    data_module = RslearnDataModule(
        path=image_to_class_dataset.path,
        inputs={
            "image": image_data_input,
            "targets": target_data_input,
        },
        task=task,
        num_workers=1,
    )
    model = SingleTaskModel(
        encoder=[
            Swin(arch="swin_v2_t", input_channels=1, output_layers=[3]),
        ],
        decoder=[
            PoolingDecoder(in_channels=192, out_channels=2),
            ClassificationHead(),
        ],
    )
    pl_module = RslearnLightningModule(
        model=model,
        task=task,
        print_parameters=True,
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
    trainer.fit(pl_module, datamodule=data_module)
    assert record_callback.recorded_params[0] == record_callback.recorded_params[1]
    assert record_callback.recorded_params[0] != record_callback.recorded_params[2]
