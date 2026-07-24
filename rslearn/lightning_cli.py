"""LightningCLI for rslearn."""

import hashlib
import json
import multiprocessing
import os
import shutil
import sys
import tempfile

import fsspec
import jsonargparse
import wandb
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.cli import (
    LightningArgumentParser,
    LightningCLI,
    SaveConfigCallback,
)
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.utilities import rank_zero_only
from upath import UPath

from rslearn.arg_parser import RslearnArgumentParser
from rslearn.log_utils import get_logger
from rslearn.utils.fsspec import open_atomic
from rslearn.utils.jsonargparse import init_jsonargparse

WANDB_ID_FNAME = "wandb_id"
MLFLOW_ID_FNAME = "mlflow_id"

logger = get_logger(__name__)


class RslearnSaveConfigCallback(SaveConfigCallback):
    """Save the CLI config even when the logger has no local log directory."""

    save_to_log_dir: bool

    def setup(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: str,
    ) -> None:
        """Fall back to ``default_root_dir`` when ``log_dir`` is unavailable."""
        if self.already_saved or not self.save_to_log_dir:
            super().setup(trainer, pl_module, stage)
            return

        log_dir = trainer.log_dir
        if log_dir is not None:
            super().setup(trainer, pl_module, stage)
            return

        log_dir = trainer.default_root_dir
        config_path = os.path.join(log_dir, self.config_filename)
        fs = get_filesystem(log_dir)
        if not self.overwrite:
            file_exists = fs.isfile(config_path) if trainer.is_global_zero else False
            file_exists = trainer.strategy.broadcast(file_exists)
            if file_exists:
                raise RuntimeError(
                    f"{self.__class__.__name__} expected {config_path} to NOT exist"
                )

        if trainer.is_global_zero:
            fs.makedirs(log_dir, exist_ok=True)
            self.parser.save(
                self.config,
                config_path,
                skip_none=False,
                overwrite=self.overwrite,
                multifile=self.multifile,
            )

        # Let Lightning perform the save_config hook and synchronization without
        # asking trainer.log_dir for a second time.
        self.save_to_log_dir = False
        try:
            super().setup(trainer, pl_module, stage)
        finally:
            self.save_to_log_dir = True


def get_cached_checkpoint(checkpoint_fname: UPath) -> str:
    """Get a local cached version of the specified checkpoint.

    If checkpoint_fname is already local, then it is returned. Otherwise, it is saved
    in a deterministic local cache directory under the system temporary directory, and
    the cached filename is returned.

    Note that the cache is not deleted when the program exits.

    Args:
        checkpoint_fname: the potentially non-local checkpoint file to load.

    Returns:
        a local filename containing the same checkpoint.
    """
    is_local = isinstance(
        checkpoint_fname.fs, fsspec.implementations.local.LocalFileSystem
    )
    if is_local:
        return checkpoint_fname.path

    cache_id = hashlib.sha256(str(checkpoint_fname).encode()).hexdigest()
    local_fname = os.path.join(
        tempfile.gettempdir(), "rslearn_cache", "checkpoints", f"{cache_id}.ckpt"
    )

    if os.path.exists(local_fname):
        logger.info(
            "using cached checkpoint for %s at %s", str(checkpoint_fname), local_fname
        )
        return local_fname

    logger.info("caching checkpoint %s to %s", str(checkpoint_fname), local_fname)
    os.makedirs(os.path.dirname(local_fname), exist_ok=True)
    with checkpoint_fname.open("rb") as src:
        with open_atomic(UPath(local_fname), "wb") as dst:
            shutil.copyfileobj(src, dst)

    return local_fname


class SaveWandbRunIdCallback(Callback):
    """Callback to save the wandb run ID to project directory in case of resume."""

    def __init__(
        self,
        project_dir: str,
        config_str: str,
    ) -> None:
        """Create a new SaveWandbRunIdCallback.

        Args:
            project_dir: the project directory.
            config_str: the JSON-encoded configuration of this experiment
        """
        self.project_dir = project_dir
        self.config_str = config_str

    @rank_zero_only
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called just before fit starts I think.

        Args:
            trainer: the Trainer object.
            pl_module: the LightningModule object.
        """
        wandb_id = wandb.run.id

        project_dir = UPath(self.project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)
        with (project_dir / WANDB_ID_FNAME).open("w") as f:
            f.write(wandb_id)

        if self.config_str is not None and "project_name" not in wandb.config:
            wandb.config.update(json.loads(self.config_str))


class SaveMLflowRunIdCallback(Callback):
    """Save the MLflow run ID and experiment configuration for resuming."""

    def __init__(self, project_dir: str, config_str: str) -> None:
        """Create a new SaveMLflowRunIdCallback.

        Args:
            project_dir: the project directory.
            config_str: the JSON-encoded configuration of this experiment.
        """
        self.project_dir = project_dir
        self.config_str = config_str

    @rank_zero_only
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Persist the run ID and log the configuration as an artifact.

        Args:
            trainer: the Trainer object.
            pl_module: the LightningModule object.
        """
        if not isinstance(trainer.logger, MLFlowLogger):
            raise ValueError("SaveMLflowRunIdCallback requires an MLFlowLogger")

        run_id = trainer.logger.run_id
        if run_id is None:
            raise ValueError("MLFlowLogger did not provide a run ID")

        project_dir = UPath(self.project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)
        run_id_path = project_dir / MLFLOW_ID_FNAME
        is_resuming = False
        if run_id_path.exists():
            with run_id_path.open("r") as f:
                is_resuming = f.read().strip() == run_id

        with run_id_path.open("w") as f:
            f.write(run_id)

        if self.config_str is not None and not is_resuming:
            trainer.logger.experiment.log_dict(
                run_id=run_id,
                dictionary=json.loads(self.config_str),
                artifact_file="config.json",
            )


class RslearnLightningCLI(LightningCLI):
    """LightningCLI that links data.tasks to model.tasks and supports environment variables."""

    save_config_callback: type[SaveConfigCallback] | None

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Link data.tasks to model.tasks and set sensible defaults.

        Args:
            parser: the argument parser
        """
        parser.link_arguments(
            "data.init_args.task", "model.init_args.task", apply_on="instantiate"
        )

        parser.set_defaults(
            {
                # Disable the sampler replacement, since the rslearn data module will set the
                # sampler as needed.
                "trainer.use_distributed_sampler": False,
                # Default to DDP with find_unused_parameters, since it works in most settings.
                "trainer.strategy": {
                    "class_path": "lightning.pytorch.strategies.DDPStrategy",
                    "init_args": {"find_unused_parameters": True},
                },
            }
        )

        # Project management option to have rslearn manage checkpoints and logger run.
        parser.add_argument(
            "--management_dir",
            type=str | None,
            help="Enable project management, and use this directory to store checkpoints and configs. If enabled, rslearn will automatically manage checkpoint directory/loading and the configured logger run",
            default=None,
        )
        parser.add_argument(
            "--project_name",
            type=str | None,
            help="The project name (used with --management_dir)",
            default=None,
        )
        parser.add_argument(
            "--run_name",
            type=str | None,
            help="A unique name for this experiment (used with --management_dir)",
            default=None,
        )
        parser.add_argument(
            "--run_description",
            type=str,
            help="Optional description of this experiment (used with --management_dir)",
            default="",
        )
        parser.add_argument(
            "--load_checkpoint_mode",
            type=str,
            help="Which checkpoint to load, if any (used with --management_dir). 'none' never loads any checkpoint, 'last' loads the most recent checkpoint, and 'best' loads the best checkpoint. 'auto' will use 'last' during fit and 'best' during val/test/predict.",
            default="auto",
        )
        parser.add_argument(
            "--load_checkpoint_required",
            type=str,
            help="Whether to fail if the expected checkpoint based on load_checkpoint_mode does not exist (used with --management_dir). 'yes' will fail while 'no' won't. 'auto' will use 'no' during fit and 'yes' during val/test/predict.",
            default="auto",
        )
        parser.add_argument(
            "--log_mode",
            type=str,
            help="Whether to use the configured experiment logger (used with --management_dir). 'yes' will enable logging, 'no' will disable logging, and 'auto' will use 'yes' during fit and 'no' during val/test/predict.",
            default="auto",
        )

    def _get_checkpoint_path(
        self,
        project_dir: UPath,
        load_checkpoint_mode: str,
        load_checkpoint_required: str,
        stage: str,
    ) -> str | None:
        """Get path to checkpoint to load from, or None to not restore checkpoint.

        Args:
            project_dir: the project directory determined from the project management
                directory.
            load_checkpoint_mode: "none" to not load any checkpoint, "last" to load the
                most recent checkpoint, "best" to load the best checkpoint. "auto" to
                use "last" during fit and "best" during val/test/predict.
            load_checkpoint_required: "yes" to fail if no checkpoint exists, "no" to
                ignore. "auto" will use "no" during fit and "yes" during
                val/test/predict.
            stage: the lightning stage (fit/val/test/predict).

        Returns:
            the path to the checkpoint for setting c.ckpt_path, or None if no
                checkpoint should be restored.
        """
        # Resolve auto options if used.
        if load_checkpoint_mode == "auto":
            if stage == "fit":
                load_checkpoint_mode = "last"
            else:
                load_checkpoint_mode = "best"
        if load_checkpoint_required == "auto":
            if stage == "fit":
                load_checkpoint_required = "no"
            else:
                load_checkpoint_required = "yes"

        if load_checkpoint_required == "yes" and load_checkpoint_mode == "none":
            raise ValueError(
                "load_checkpoint_required cannot be set when load_checkpoint_mode is none"
            )

        ckpt_path: str | None = None

        if load_checkpoint_mode == "best":
            # Checkpoints should be either:
            # - last.ckpt
            # - of the form "A=B-C=D-....ckpt" with one key being epoch=X
            # So we want the one with the highest epoch, and only use last.ckpt if
            # it's the only option.
            # User should set save_top_k=1 so there's just one, otherwise we won't
            # actually know which one is the best.
            best_checkpoint = None
            best_epochs = None

            # Avoid error in case project_dir doesn't exist.
            fnames = project_dir.iterdir() if project_dir.exists() else []

            for option in fnames:
                if not option.name.endswith(".ckpt"):
                    continue

                # Try to see what epochs this checkpoint is at.
                # If it is some other format, then set it 0 so we only use it if it's
                # the only option.
                # If it is last.ckpt then we set it -100 to only use it if there is not
                # even another format like "best.ckpt".
                extracted_epochs = 0
                if option.name == "last.ckpt":
                    extracted_epochs = -100

                parts = option.name.split(".ckpt")[0].split("-")
                for part in parts:
                    kv_parts = part.split("=")
                    if len(kv_parts) != 2:
                        continue
                    if kv_parts[0] != "epoch":
                        continue
                    extracted_epochs = int(kv_parts[1])

                if best_epochs is None or extracted_epochs > best_epochs:
                    best_checkpoint = option
                    best_epochs = extracted_epochs

            if best_checkpoint is not None:
                # Cache the checkpoint so we only need to download once in case we
                # reuse it later.
                # We only cache with --load_best since this is the only scenario where we
                # expect to keep reusing the same checkpoint.
                ckpt_path = get_cached_checkpoint(best_checkpoint)

        elif load_checkpoint_mode == "last":
            last_checkpoint_path = project_dir / "last.ckpt"
            if last_checkpoint_path.exists():
                ckpt_path = str(last_checkpoint_path)

        else:
            raise ValueError(f"unknown load_checkpoint_mode {load_checkpoint_mode}")

        if load_checkpoint_required == "yes" and ckpt_path is None:
            raise ValueError(
                f"load_checkpoint_required is set but no checkpoint was found under {project_dir}"
            )

        return ckpt_path

    def enable_project_management(self, management_dir: str) -> None:
        """Enable project management in the specified directory.

        Sets default_root_dir on the trainer (used by ManagedBestLastCheckpoint),
        configures W&B or MLflow logging, and handles checkpoint loading/resuming.

        Args:
            management_dir: the directory to store checkpoints and logger metadata.
        """
        subcommand = self.config.subcommand
        c = self.config[subcommand]

        # Project name and run name are required with project management.
        if not c.project_name or not c.run_name:
            raise ValueError(
                "project name and run name must be set when using project management"
            )

        # Get project directory within the project management directory.
        project_dir = UPath(management_dir) / c.project_name / c.run_name

        # Set default_root_dir so ManagedBestLastCheckpoint can resolve it.
        c.trainer.default_root_dir = str(project_dir)

        # Disable Lightning's default ModelCheckpoint since we use our own
        # (ManagedBestLastCheckpoint). Without this, Lightning adds a redundant
        # ModelCheckpoint that writes to the logger's directory.
        c.trainer.enable_checkpointing = False

        # Configure experiment logging. W&B remains the default for compatibility;
        # users select MLflow with trainer.logger.class_path in their config.
        should_log = False
        if c.log_mode == "yes":
            should_log = True
        elif c.log_mode == "auto":
            should_log = subcommand == "fit"

        if should_log:
            if not c.trainer.logger:
                c.trainer.logger = jsonargparse.Namespace(
                    {
                        "class_path": "lightning.pytorch.loggers.WandbLogger",
                        "init_args": jsonargparse.Namespace(),
                    }
                )
            logger_class = c.trainer.logger.class_path.rsplit(".", 1)[-1]
            if logger_class == "WandbLogger":
                c.trainer.logger.init_args.project = c.project_name
                c.trainer.logger.init_args.name = c.run_name
                if c.run_description:
                    c.trainer.logger.init_args.notes = c.run_description
            elif logger_class == "MLFlowLogger":
                c.trainer.logger.init_args.experiment_name = c.project_name
                c.trainer.logger.init_args.run_name = c.run_name
                if c.run_description:
                    tags = c.trainer.logger.init_args.get("tags", {}) or {}
                    tags["mlflow.note.content"] = c.run_description
                    c.trainer.logger.init_args.tags = tags
            else:
                raise ValueError(
                    "Project management logging only supports WandbLogger and "
                    f"MLFlowLogger, got {c.trainer.logger.class_path}"
                )

            # Add a callback to save the logger run ID for resume.
            if "callbacks" not in c.trainer or not c.trainer.callbacks:
                c.trainer.callbacks = []

            callback_class = (
                "SaveWandbRunIdCallback"
                if logger_class == "WandbLogger"
                else "SaveMLflowRunIdCallback"
            )
            has_logger_callback = any(
                cb.class_path.rsplit(".", 1)[-1] == callback_class
                for cb in c.trainer.callbacks
            )
            if not has_logger_callback:
                config_str = json.dumps(
                    c.as_dict(), default=lambda _: "<not serializable>"
                )
                c.trainer.callbacks.append(
                    jsonargparse.Namespace(
                        {
                            "class_path": callback_class,
                            "init_args": jsonargparse.Namespace(
                                {
                                    "project_dir": str(project_dir),
                                    "config_str": config_str,
                                }
                            ),
                        }
                    )
                )
        elif c.trainer.logger:
            logger.warning(
                "Model management is enabled and logging should be off, but the model config specifies a logger. "
                "The logger should be removed from the model config, since it will not be automatically disabled."
            )

        # Load existing checkpoint.
        checkpoint_path = self._get_checkpoint_path(
            project_dir=project_dir,
            load_checkpoint_mode=c.load_checkpoint_mode,
            load_checkpoint_required=c.load_checkpoint_required,
            stage=subcommand,
        )
        if checkpoint_path is not None:
            logger.info(f"found checkpoint to resume from at {checkpoint_path}")
            c.ckpt_path = checkpoint_path

            # When training resumes, reconnect to the original experiment run.
            if subcommand == "fit" and should_log:
                logger_class = c.trainer.logger.class_path.rsplit(".", 1)[-1]
                if logger_class == "WandbLogger":
                    run_id_path = project_dir / WANDB_ID_FNAME
                    if run_id_path.exists():
                        with run_id_path.open("r") as f:
                            c.trainer.logger.init_args.id = f.read().strip()
                        c.trainer.logger.init_args.resume = "must"
                elif logger_class == "MLFlowLogger":
                    run_id_path = project_dir / MLFLOW_ID_FNAME
                    if run_id_path.exists():
                        with run_id_path.open("r") as f:
                            c.trainer.logger.init_args.run_id = f.read().strip()

    def before_instantiate_classes(self) -> None:
        """Called before Lightning class initialization."""
        if self.save_config_callback is SaveConfigCallback:
            self.save_config_callback = RslearnSaveConfigCallback

        if not hasattr(self.config, "subcommand"):
            logger.warning(
                "Config does not have subcommand attribute, assuming we are in run=False mode"
            )
            subcommand = None
            c = self.config
        else:
            subcommand = self.config.subcommand
            c = self.config[subcommand]

        # For predict, make sure that return_predictions is False.
        # Otherwise all the predictions would be stored in memory which can lead to
        # high memory consumption.
        if subcommand == "predict":
            c.return_predictions = False

        if c.management_dir:
            self.enable_project_management(c.management_dir)


def model_handler() -> None:
    """Handler for any rslearn model X commands."""
    init_jsonargparse()

    # Set forkserver preload since torch can take a long time to load in each process.
    logger.debug("Configuring forkserver preload")
    multiprocessing.set_forkserver_preload(
        [
            "fiona",
            "jsonargparse",
            "numpy",
            "pickle",
            "PIL",
            "torch",
            "torch.multiprocessing",
            "torchvision",
            "upath",
            "wandb",
            "rslearn.main",
            "rslearn.train.dataset",
            "rslearn.train.data_module",
        ]
    )

    RslearnLightningCLI(
        model_class=LightningModule,
        datamodule_class=LightningDataModule,
        args=sys.argv[2:],
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        parser_class=RslearnArgumentParser,
    )
