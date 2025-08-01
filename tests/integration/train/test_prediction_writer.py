"""Integration tests for rslearn.train.prediction_writer."""

import lightning.pytorch as pl

from rslearn.dataset.dataset import Dataset
from rslearn.models.multitask import MultiTaskModel
from rslearn.models.pooling_decoder import PoolingDecoder
from rslearn.models.singletask import SingleTaskModel
from rslearn.models.swin import Swin
from rslearn.train.data_module import RslearnDataModule
from rslearn.train.dataset import DataInput
from rslearn.train.lightning_module import RslearnLightningModule
from rslearn.train.optimizer import AdamW
from rslearn.train.prediction_writer import RslearnWriter
from rslearn.train.tasks.classification import ClassificationHead, ClassificationTask
from rslearn.train.tasks.multi_task import MultiTask


def test_predict(
    image_to_class_data_module: RslearnDataModule, image_to_class_model: SingleTaskModel
) -> None:
    """Ensure prediction works."""
    # Set up the basic RslearnLightningModule for the image_to_class task.
    pl_module = RslearnLightningModule(
        model=image_to_class_model,
        task=image_to_class_data_module.task,
        optimizer=AdamW(),
    )
    # Now create Trainer with an RslearnWriter.
    writer = RslearnWriter(
        path=image_to_class_data_module.path,
        output_layer="output",
    )
    trainer = pl.Trainer(
        callbacks=[writer],
    )
    trainer.predict(pl_module, datamodule=image_to_class_data_module)
    window = Dataset(writer.path).load_windows()[0]
    assert window.is_layer_completed("output")


def test_predict_multi_task(image_to_class_dataset: Dataset) -> None:
    """Ensure prediction writing still works with MultiTaskModel."""
    image_data_input = DataInput("raster", ["image"], bands=["band"], passthrough=True)
    target_data_input = DataInput("vector", ["label"])
    task = MultiTask(
        tasks={
            "mytask": ClassificationTask("label", ["cls0", "cls1"], read_class_id=True)
        },
        input_mapping={
            "mytask": {
                "targets": "targets",
            }
        },
    )
    data_module = RslearnDataModule(
        path=image_to_class_dataset.path,
        inputs={
            "image": image_data_input,
            "targets": target_data_input,
        },
        task=task,
    )
    model = MultiTaskModel(
        encoder=[
            Swin(arch="swin_v2_t", input_channels=1, output_layers=[3]),
        ],
        decoders={
            "mytask": [
                PoolingDecoder(in_channels=192, out_channels=2),
                ClassificationHead(),
            ],
        },
    )
    pl_module = RslearnLightningModule(
        model=model,
        task=task,
        optimizer=AdamW(),
    )
    # Now create Trainer with an RslearnWriter.
    writer = RslearnWriter(
        path=image_to_class_dataset.path,
        output_layer="output",
        selector=["mytask"],
    )
    trainer = pl.Trainer(
        callbacks=[writer],
    )
    trainer.predict(pl_module, datamodule=data_module)
    window = Dataset(writer.path).load_windows()[0]
    assert window.is_layer_completed("output")
