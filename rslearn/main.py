"""Entrypoint for the rslearn command-line interface."""

import argparse
import multiprocessing
import random
import sys
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any, TypeVar

import tqdm
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from rasterio.crs import CRS
from upath import UPath

from rslearn.config import LayerConfig
from rslearn.const import WGS84_EPSG
from rslearn.data_sources import Item, data_source_from_config
from rslearn.dataset import Dataset, Window, WindowLayerData
from rslearn.dataset.add_windows import add_windows_from_box, add_windows_from_file
from rslearn.dataset.manage import (
    materialize_dataset_windows,
    prepare_dataset_windows,
    retry,
)
from rslearn.log_utils import get_logger
from rslearn.tile_stores import get_tile_store_with_layer
from rslearn.train.data_module import RslearnDataModule
from rslearn.train.lightning_module import RslearnLightningModule
from rslearn.utils import Projection, STGeometry

logger = get_logger(__name__)

handler_registry = {}

ItemType = TypeVar("ItemType", bound="Item")

MULTIPROCESSING_CONTEXT = "forkserver"


def register_handler(category: Any, command: str) -> Callable:
    """Register a new handler for a command."""

    def decorator(f: Callable) -> Callable:
        handler_registry[(category, command)] = f
        return f

    return decorator


def parse_time(time_str: str) -> datetime:
    """Parse an ISO-formatted time string into datetime while ensuring timezone is set.

    The timezone defaults to UTC.
    """
    ts = datetime.fromisoformat(time_str)
    if not ts.tzinfo:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def parse_time_range(
    start: str | None, end: str | None
) -> tuple[datetime, datetime] | None:
    """Parse a start and end time string into a time range tuple."""
    if not start or not end:
        return None
    return (parse_time(start), parse_time(end))


def parse_disabled_layers(disabled_layers: str) -> list[str]:
    """Parse the disabled layers string."""
    return disabled_layers.split(",") if disabled_layers else []


@register_handler("dataset", "add_windows")
def add_windows() -> None:
    """Handler for the rslearn dataset add_windows command."""
    parser = argparse.ArgumentParser(
        prog="rslearn dataset add_windows",
        description="rslearn dataset add_windows: add windows to a dataset",
    )
    parser.add_argument(
        "--root", type=str, required=True, help="Dataset root directory"
    )
    parser.add_argument(
        "--group", type=str, required=True, help="Add windows to this group"
    )
    parser.add_argument(
        "--box",
        type=str,
        default=None,
        help="Specify extent by bounding box (comma-separated coordinates x1,y1,x2,y2)",
    )
    parser.add_argument(
        "--fname", type=str, default=None, help="Specify extent(s) by vector file"
    )
    parser.add_argument(
        "--crs", type=str, default=None, help="The CRS of the output windows"
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="The resolution of the output windows",
    )
    parser.add_argument(
        "--x_res", type=float, default=1, help="The X resolution of the output windows"
    )
    parser.add_argument(
        "--y_res", type=float, default=-1, help="The Y resolution of the output windows"
    )
    parser.add_argument(
        "--src_crs",
        type=str,
        default=None,
        help="The CRS of the input extents (only if box is provided)",
    )
    parser.add_argument(
        "--src_resolution",
        type=float,
        default=None,
        help="The resolution of the input extents",
    )
    parser.add_argument(
        "--src_x_res",
        type=float,
        default=1,
        help="The X resolution of the input extents",
    )
    parser.add_argument(
        "--src_y_res",
        type=float,
        default=1,
        help="The Y resolution of the input extents",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of output window (or prefix of output windows)",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=None,
        help=(
            "Instead of creating one window per input extent (default), "
            + "create windows along a grid of this cell size"
        ),
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help=(
            "Instead of creating windows the size of the input extents, "
            + "create windows of this fixed size centered at each extent's center"
        ),
    )
    parser.add_argument("--start", type=str, default=None, help="Optional start time")
    parser.add_argument("--end", type=str, default=None, help="Optional end time")
    parser.add_argument(
        "--utm",
        type=bool,
        default=False,
        help="Create windows in an appropriate UTM projection",
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args(args=sys.argv[3:])

    def parse_projection(
        crs_str: str | None,
        resolution: float | None,
        x_res: float,
        y_res: float,
        default_crs: CRS | None = None,
    ) -> Projection | None:
        if not crs_str:
            if default_crs:
                crs = default_crs
            else:
                return None
        else:
            crs = CRS.from_string(crs_str)

        if resolution:
            return Projection(crs, resolution, -resolution)
        else:
            return Projection(crs, x_res, y_res)

    # CRS for dst is not needed if we are auto-selecting a UTM projection.
    # So here we make sure that parse_projection returns a non-null Projection with
    # placeholder CRS.
    dst_projection = parse_projection(
        args.crs,
        args.resolution,
        args.x_res,
        args.y_res,
        default_crs=CRS.from_epsg(WGS84_EPSG),
    )

    kwargs = dict(
        dataset=Dataset(UPath(args.root)),
        group=args.group,
        projection=dst_projection,
        name=args.name,
        grid_size=args.grid_size,
        window_size=args.window_size,
        time_range=parse_time_range(args.start, args.end),
        use_utm=args.utm,
    )

    if args.box:
        # Parse box.
        box = [float(value) for value in args.box.split(",")]

        windows = add_windows_from_box(
            # TODO: we should have an object for box
            box=box,  # type: ignore
            src_projection=parse_projection(
                args.src_crs, args.src_resolution, args.src_x_res, args.src_y_res
            ),
            **kwargs,
        )

    elif args.fname:
        windows = add_windows_from_file(fname=args.fname, **kwargs)

    else:
        raise Exception("one of box or fname must be specified")

    logger.info(f"created {len(windows)} windows")


def add_apply_on_windows_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for handlers that use the apply_on_windows helper.

    Args:
        parser: the argument parser
    """
    parser.add_argument(
        "--root", type=str, required=True, help="Dataset root directory"
    )
    parser.add_argument(
        "--group", type=str, default=None, help="Only prepare windows in this group"
    )
    parser.add_argument(
        "--window", type=str, nargs="*", default=None, help="Only prepare these windows"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes (default 0 to use main process only)",
    )
    parser.add_argument(
        "--load-workers",
        type=int,
        default=None,
        help="Number of workers for loading windows (defaults to --workers)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of windows to process in each batch (default 1)",
    )
    parser.add_argument(
        "--jobs-per-process",
        type=int,
        default=None,
        help="Number of jobs to run in each worker process before restarting",
    )
    parser.add_argument(
        "--use-initial-job",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
    )


def apply_on_windows(
    f: Callable[[list[Window]], None],
    dataset: Dataset,
    group: str | None = None,
    names: list[str] | None = None,
    workers: int = 0,
    load_workers: int | None = None,
    batch_size: int = 1,
    jobs_per_process: int | None = None,
    use_initial_job: bool = True,
) -> None:
    """A helper to apply a function on windows in a dataset.

    Args:
        f: the function to apply on lists of windows.
        dataset: the dataset.
        group: optional, only apply on windows in this group.
        names: optional, only apply on windows with these names.
        workers: the number of parallel workers to use, default 0 (main thread only).
        load_workers: optional different number of workers to use for loading the
            windows. If set, workers controls the number of workers to process the
            jobs, while load_workers controls the number of workers to use for reading
            windows from the rslearn dataset.
        batch_size: if workers > 0, the maximum number of windows to pass to the
            function.
        jobs_per_process: optional, terminate processes after they have handled this
            many jobs. This is useful if there is a memory leak in a dependency.
        use_initial_job: if workers > 0, by default, an initial job is run on the first
            batch in the main thread before spawning workers. This can handle things
            like building indexes that should not be done in parallel. Set this false
            to disable using the initial job.
    """
    if hasattr(f, "set_dataset"):
        f.set_dataset(dataset)

    groups = None
    if group:
        groups = [group]
    if load_workers is None:
        load_workers = workers
    windows = dataset.load_windows(
        groups=groups, names=names, workers=load_workers, show_progress=True
    )
    logger.info(f"found {len(windows)} windows")

    if hasattr(f, "get_jobs"):
        jobs = f.get_jobs(windows, load_workers)
        logger.info(f"got {len(jobs)} jobs")
    else:
        jobs = windows

    random.shuffle(jobs)

    if use_initial_job and len(jobs) > 0:
        # Apply directly on first window to get any initialization out of the way.
        f([jobs[0]])
        jobs = jobs[1:]

    batches = []
    for i in range(0, len(jobs), batch_size):
        batches.append(jobs[i : i + batch_size])

    num_batches = len(batches)
    if workers == 0:
        # Process batches sequentially but with same error handling as parallel
        for batch in tqdm.tqdm(batches, total=num_batches):
            f(batch)
    else:
        # Process batches in parallel
        p = multiprocessing.Pool(processes=workers, maxtasksperchild=jobs_per_process)
        outputs = p.imap_unordered(f, batches)
        for _ in tqdm.tqdm(outputs, total=num_batches):
            pass
        p.close()


def apply_on_windows_args(f: Callable[..., None], args: argparse.Namespace) -> None:
    """Call apply_on_windows with arguments passed via command-line interface."""
    dataset = Dataset(UPath(args.root), args.disabled_layers)
    apply_on_windows(
        f=f,
        dataset=dataset,
        group=args.group,
        names=args.window,
        workers=args.workers,
        load_workers=args.load_workers,
        batch_size=args.batch_size,
        jobs_per_process=args.jobs_per_process,
        use_initial_job=args.use_initial_job,
    )


class PrepareHandler:
    """apply_on_windows handler for the rslearn dataset prepare command."""

    def __init__(
        self,
        force: bool,
        retry_max_attempts: int = 0,
        retry_backoff: timedelta = timedelta(minutes=1),
    ) -> None:
        """Initialize a new PrepareHandler.

        Args:
            force: force prepare
            retry_max_attempts: set greater than zero to retry for this many attempts in
                case of error.
            retry_backoff: how long to wait before retrying (see retry).
        """
        self.force = force
        self.dataset: Dataset | None = None
        self.retry_max_attempts = retry_max_attempts
        self.retry_backoff = retry_backoff

    def set_dataset(self, dataset: Dataset) -> None:
        """Captures the dataset from apply_on_windows_args.

        Args:
        dataset: the dataset to prepare.
        """
        self.dataset = dataset

    def __call__(self, windows: list[Window]) -> None:
        """Prepares the windows from apply_on_windows."""
        logger.info(f"Running prepare on {len(windows)} windows")
        if self.dataset is None:
            raise ValueError("dataset not set")
        prepare_dataset_windows(
            self.dataset,
            windows,
            self.force,
            retry_max_attempts=self.retry_max_attempts,
            retry_backoff=self.retry_backoff,
        )


@register_handler("dataset", "prepare")
def dataset_prepare() -> None:
    """Handler for the rslearn dataset prepare command."""
    parser = argparse.ArgumentParser(
        prog="rslearn dataset prepare",
        description="rslearn dataset prepare: lookup items in retrieved data sources",
    )
    parser.add_argument(
        "--force",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Prepare windows even if they were previously prepared",
    )
    parser.add_argument(
        "--disabled-layers",
        type=parse_disabled_layers,
        default="",
        help="List of layers to disable e.g 'layer1,layer2'",
    )
    parser.add_argument(
        "--retry-max-attempts",
        type=int,
        default=0,
        help="Retry for this many attempts",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=int,
        default=0,
        help="Backoff time (seconds) between retries",
    )
    add_apply_on_windows_args(parser)
    args = parser.parse_args(args=sys.argv[3:])

    fn = PrepareHandler(
        args.force,
        retry_max_attempts=args.retry_max_attempts,
        retry_backoff=timedelta(seconds=args.retry_backoff_seconds),
    )
    apply_on_windows_args(fn, args)


def _load_window_layer_datas(
    window: Window,
) -> tuple[Window, dict[str, WindowLayerData]]:
    # Helper for IngestHandler to use with multiprocessing.
    return window, window.load_layer_datas()


class IngestHandler:
    """apply_on_windows handler for the rslearn dataset ingest command."""

    def __init__(
        self,
        ignore_errors: bool = False,
        retry_max_attempts: int = 0,
        retry_backoff: timedelta = timedelta(minutes=1),
    ) -> None:
        """Initialize a new IngestHandler."""
        self.dataset: Dataset | None = None
        self.ignore_errors = ignore_errors
        self.retry_max_attempts = retry_max_attempts
        self.retry_backoff = retry_backoff

    def set_dataset(self, dataset: Dataset) -> None:
        """Captures the dataset from apply_on_windows_args.

        Args:
        dataset: the dataset to ingest.
        """
        self.dataset = dataset

    def __call__(
        self, jobs: list[tuple[str, LayerConfig, Item, list[STGeometry]]]
    ) -> None:
        """Ingest the specified items.

        The items are computed from list of windows via IngestHandler.get_jobs.

        Args:
            jobs: list of (layer_name, item, geometries) tuples to ingest.
        """
        logger.info(f"Running ingest for {len(jobs)} jobs")
        import gc

        if self.dataset is None:
            raise ValueError("dataset not set")
        tile_store = self.dataset.get_tile_store()

        # Group jobs by layer name.
        jobs_by_layer: dict = {}
        configs_by_layer: dict = {}
        for layer_name, layer_cfg, item, geometries in jobs:
            if layer_name not in jobs_by_layer:
                jobs_by_layer[layer_name] = []
            jobs_by_layer[layer_name].append((item, geometries))
            configs_by_layer[layer_name] = layer_cfg

        for layer_name, items_and_geometries in jobs_by_layer.items():
            layer_tile_store = get_tile_store_with_layer(
                tile_store, layer_name, layer_cfg
            )
            layer_cfg = self.dataset.layers[layer_name]
            data_source = data_source_from_config(layer_cfg, self.dataset.path)

            try:
                retry(
                    lambda: data_source.ingest(
                        tile_store=layer_tile_store,
                        items=[item for item, _ in items_and_geometries],
                        geometries=[
                            geometries for _, geometries in items_and_geometries
                        ],
                    ),
                    retry_max_attempts=self.retry_max_attempts,
                    retry_backoff=self.retry_backoff,
                )
            except Exception as e:
                if not self.ignore_errors:
                    raise

                logger.error(
                    "warning: got error while ingesting "
                    + f"{len(items_and_geometries)} items: {e}"
                )

        gc.collect()

    def _load_layer_data_for_windows(
        self, windows: list[Window], workers: int
    ) -> list[tuple[Window, dict[str, WindowLayerData]]]:
        if workers == 0:
            return [(_load_window_layer_datas(window)) for window in windows]
        p = multiprocessing.Pool(workers)
        outputs = p.imap_unordered(_load_window_layer_datas, windows)
        windows_and_layer_datas = []
        for window, layer_datas in tqdm.tqdm(
            outputs, total=len(windows), desc="Loading window layer datas"
        ):
            windows_and_layer_datas.append((window, layer_datas))
        p.close()
        return windows_and_layer_datas

    def get_jobs(
        self, windows: list[Window], workers: int
    ) -> list[tuple[str, LayerConfig, Item, list[STGeometry]]]:
        """Computes ingest jobs from window list.

        Each ingest job is a tuple of the layer name, the item to ingest, and the
        geometries of windows that require that item.

        This makes sure that jobs are grouped by item rather than by window, which
        makes sense because there's no reason to ingest the same item twice.
        """
        if self.dataset is None:
            raise ValueError("dataset not set")
        # TODO: avoid duplicating ingest_dataset_windows...

        # Load layer datas of each window.
        windows_and_layer_datas = self._load_layer_data_for_windows(windows, workers)

        jobs: list[tuple[str, LayerConfig, Item, list[STGeometry]]] = []
        for layer_name, layer_cfg in self.dataset.layers.items():
            if not layer_cfg.data_source:
                continue
            if not layer_cfg.data_source.ingest:
                continue

            data_source = data_source_from_config(layer_cfg, self.dataset.path)

            geometries_by_item: dict = {}
            for window, layer_datas in windows_and_layer_datas:
                if layer_name not in layer_datas:
                    continue
                geometry = window.get_geometry()
                layer_data = layer_datas[layer_name]
                for group in layer_data.serialized_item_groups:
                    for serialized_item in group:
                        item = data_source.deserialize_item(  # type: ignore
                            serialized_item
                        )
                        if item not in geometries_by_item:
                            geometries_by_item[item] = []
                        geometries_by_item[item].append(geometry)

            for item, geometries in geometries_by_item.items():
                jobs.append((layer_name, layer_cfg, item, geometries))

        logger.info(f"computed {len(jobs)} ingest jobs from {len(windows)} windows")
        return jobs


@register_handler("dataset", "ingest")
def dataset_ingest() -> None:
    """Handler for the rslearn dataset ingest command."""
    parser = argparse.ArgumentParser(
        prog="rslearn dataset ingest",
        description="rslearn dataset ingest: ingest items in retrieved data sources",
    )
    parser.add_argument(
        "--disabled-layers",
        type=parse_disabled_layers,
        default="",
        help="List of layers to disable e.g 'layer1,layer2'",
    )
    parser.add_argument(
        "--ignore-errors",
        type=bool,
        default=False,
        help="Ignore ingestion errors in individual jobs",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--retry-max-attempts",
        type=int,
        default=0,
        help="Retry for this many attempts",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=int,
        default=0,
        help="Backoff time (seconds) between retries",
    )
    add_apply_on_windows_args(parser)
    args = parser.parse_args(args=sys.argv[3:])

    fn = IngestHandler(
        ignore_errors=args.ignore_errors,
        retry_max_attempts=args.retry_max_attempts,
        retry_backoff=timedelta(seconds=args.retry_backoff_seconds),
    )
    apply_on_windows_args(fn, args)


class MaterializeHandler:
    """apply_on_windows handler for the rslearn dataset materialize command."""

    def __init__(
        self,
        ignore_errors: bool = False,
        retry_max_attempts: int = 0,
        retry_backoff: timedelta = timedelta(minutes=1),
    ) -> None:
        """Initialize a MaterializeHandler."""
        self.dataset: Dataset | None = None
        self.ignore_errors = ignore_errors
        self.retry_max_attempts = retry_max_attempts
        self.retry_backoff = retry_backoff

    def set_dataset(self, dataset: Dataset) -> None:
        """Captures the dataset from apply_on_windows_args.

        Args:
        dataset: the dataset to prepare.
        """
        self.dataset = dataset

    def __call__(self, windows: list[Window]) -> None:
        """Materializes the windows from apply_on_windows."""
        logger.info(f"Running Materialize with {len(windows)} windows")
        if self.dataset is None:
            raise ValueError("dataset not set")
        try:
            materialize_dataset_windows(
                self.dataset,
                windows,
                retry_max_attempts=self.retry_max_attempts,
                retry_backoff=self.retry_backoff,
            )
        except Exception as e:
            if not self.ignore_errors:
                logger.error(f"Error materializing windows: {e}")
                raise
            logger.warning(f"Ignoring error while materializing windows: {e}")


@register_handler("dataset", "materialize")
def dataset_materialize() -> None:
    """Handler for the rslearn dataset materialize command."""
    parser = argparse.ArgumentParser(
        prog="rslearn dataset materialize",
        description=(
            "rslearn dataset materialize: "
            + "materialize data from retrieved data sources"
        ),
    )
    parser.add_argument(
        "--disabled-layers",
        type=parse_disabled_layers,
        default="",
        help="List of layers to disable e.g 'layer1,layer2'",
    )
    parser.add_argument(
        "--ignore-errors",
        type=bool,
        default=False,
        help="Ignore errors in individual jobs",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--retry-max-attempts",
        type=int,
        default=0,
        help="Retry for this many attempts",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=int,
        default=0,
        help="Backoff time (seconds) between retries",
    )
    add_apply_on_windows_args(parser)
    args = parser.parse_args(args=sys.argv[3:])
    fn = MaterializeHandler(
        ignore_errors=args.ignore_errors,
        retry_max_attempts=args.retry_max_attempts,
        retry_backoff=timedelta(seconds=args.retry_backoff_seconds),
    )
    apply_on_windows_args(fn, args)


class RslearnLightningCLI(LightningCLI):
    """LightningCLI that links data.tasks to model.tasks."""

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Link data.tasks to model.tasks.

        Args:
            parser: the argument parser
        """
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


def model_handler() -> None:
    """Handler for any rslearn model X commands."""
    RslearnLightningCLI(
        model_class=RslearnLightningModule,
        datamodule_class=RslearnDataModule,
        args=sys.argv[2:],
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
    )


@register_handler("model", "fit")
def model_fit() -> None:
    """Handler for rslearn model fit."""
    model_handler()


@register_handler("model", "validate")
def model_validate() -> None:
    """Handler for rslearn model validate."""
    model_handler()


@register_handler("model", "test")
def model_test() -> None:
    """Handler for rslearn model test."""
    model_handler()


@register_handler("model", "predict")
def model_predict() -> None:
    """Handler for rslearn model predict."""
    model_handler()


def main() -> None:
    """CLI entrypoint."""
    try:
        multiprocessing.set_start_method(MULTIPROCESSING_CONTEXT)
    except RuntimeError as e:
        logger.error(
            f"Multiprocessing context already set to {multiprocessing.get_context()}: "
            + f"ignoring {e}"
        )
    except Exception as e:
        logger.error(f"Failed to set multiprocessing context: {e}")
        raise
    finally:
        logger.info(f"Using multiprocessing context: {multiprocessing.get_context()}")
    parser = argparse.ArgumentParser(description="rslearn")
    parser.add_argument(
        "category", help="Command category: dataset, annotate, or model"
    )
    parser.add_argument("command", help="The command to run")
    args = parser.parse_args(args=sys.argv[1:3])

    handler = handler_registry.get((args.category, args.command))
    if handler is None:
        logger.error(f"Unknown command: {args.category} {args.command}")
        sys.exit(1)

    handler()


if __name__ == "__main__":
    main()
