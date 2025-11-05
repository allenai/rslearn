"""LightningCLI for rslearn."""

import sys

from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from rslearn.arg_parser import RslearnArgumentParser
from rslearn.train.data_module import RslearnDataModule
from rslearn.train.lightning_module import RslearnLightningModule


class RslearnLightningCLI(LightningCLI):
    """LightningCLI that links data.tasks to model.tasks and supports environment variables."""

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Link data.tasks to model.tasks.

        Args:
            parser: the argument parser
        """
        # Link data.tasks to model.tasks
        parser.link_arguments(
            "data.init_args.task", "model.init_args.task", apply_on="instantiate"
        )

    def before_instantiate_classes(self) -> None:
        """Called before Lightning class initialization.

        Sets the dataset path for any configured RslearnPredictionWriter callbacks.
        """
        subcommand = self.config.subcommand
        c = self.config[subcommand]

        # If there is a RslearnPredictionWriter, set its path.
        prediction_writer_callback = None
        if "callbacks" in c.trainer:
            for existing_callback in c.trainer.callbacks:
                if (
                    existing_callback.class_path
                    == "rslearn.train.prediction_writer.RslearnWriter"
                ):
                    prediction_writer_callback = existing_callback
        if prediction_writer_callback:
            prediction_writer_callback.init_args.path = c.data.init_args.path

        # Disable the sampler replacement, since the rslearn data module will set the
        # sampler as needed.
        c.trainer.use_distributed_sampler = False

        # For predict, make sure that return_predictions is False.
        # Otherwise all the predictions would be stored in memory which can lead to
        # high memory consumption.
        if subcommand == "predict":
            c.return_predictions = False


def model_handler() -> None:
    """Handler for any rslearn model X commands."""
    RslearnLightningCLI(
        model_class=RslearnLightningModule,
        datamodule_class=RslearnDataModule,
        args=sys.argv[2:],
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        parser_class=RslearnArgumentParser,
    )
