"""Integration tests for rslearn.train.prediction_writer."""

import lightning.pytorch as pl

from rslearn.dataset.dataset import Dataset
from rslearn.models.conv import Conv
from rslearn.models.module_wrapper import EncoderModuleWrapper
from rslearn.models.multitask import MultiTaskModel
from rslearn.models.pooling_decoder import PoolingDecoder
from rslearn.models.singletask import SingleTaskModel
from rslearn.models.swin import Swin
from rslearn.train.data_module import RslearnDataModule
from rslearn.train.dataset import DataInput, SplitConfig
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


def test_predict_with_all_patches(image_to_class_dataset: Dataset) -> None:
    """Ensure prediction works with IterableAllPatchesDataset and multiple workers.

    If __len__ is defined on IterableAllPatchesDataset, and gives different lengths based on
    the number of workers active, then this can cause problems because Lightning will
    call it before spawning workers, and get a length, but more padding may be needed
    afterward but Lightning will cut off the prediction.
    """
    image_data_input = DataInput("raster", ["image"], bands=["band"], passthrough=True)
    target_data_input = DataInput("vector", ["label"])
    task = ClassificationTask("label", ["cls0", "cls1"], read_class_id=True)

    # There should be 16 1x1 patches in the 4x4 window in the dataset.
    # So with 1 worker it would say length is 8 (with bs=2), but with 4 workers the
    # length is still 8 since we split windows across workers but there is only one
    # window. This can cause problems if IterableAllPatchesDataset.__len__ is defined.
    data_module = RslearnDataModule(
        path=image_to_class_dataset.path,
        inputs={
            "image": image_data_input,
            "targets": target_data_input,
        },
        task=task,
        predict_config=SplitConfig(
            patch_size=1,
            load_all_patches=True,
        ),
        num_workers=4,
        batch_size=2,
    )

    model = SingleTaskModel(
        encoder=[
            EncoderModuleWrapper(
                module=Conv(in_channels=1, out_channels=32, kernel_size=1)
            )
        ],
        decoder=[
            PoolingDecoder(in_channels=32, out_channels=2),
            ClassificationHead(),
        ],
    )

    pl_module = RslearnLightningModule(
        model=model,
        task=data_module.task,
        optimizer=AdamW(),
    )
    writer = RslearnWriter(
        path=data_module.path,
        output_layer="output",
    )
    trainer = pl.Trainer(
        callbacks=[writer],
    )
    trainer.predict(pl_module, datamodule=data_module)
    window = Dataset(writer.path).load_windows()[0]
    assert window.is_layer_completed("output")
