"""Entrypoint for the rslearn command-line interface."""

import argparse
import multiprocessing
import random
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import tqdm
import wandb
from lightning.pytorch.cli import LightningCLI
from rasterio.crs import CRS

from rslearn.const import WGS84_EPSG
from rslearn.data_sources import Item, data_source_from_config
from rslearn.dataset import Dataset, Window
from rslearn.dataset.add_windows import add_windows_from_box, add_windows_from_file
from rslearn.dataset.manage import materialize_dataset_windows, prepare_dataset_windows
from rslearn.tile_stores import PrefixedTileStore
from rslearn.train.data_module import RslearnDataModule
from rslearn.train.lightning_module import RslearnLightningModule
from rslearn.utils import Projection, STGeometry

handler_registry = {}


def register_handler(category, command):
    """Register a new handler for a command."""

    def decorator(f):
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


@register_handler("dataset", "add_windows")
def add_windows():
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

    def parse_projection(crs_str, resolution, x_res, y_res, default_crs=None):
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
        dataset=Dataset(ds_root=args.root),
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
            box=box,
            src_projection=parse_projection(
                args.src_crs, args.src_resolution, args.src_x_res, args.src_y_res
            ),
            **kwargs,
        )

    elif args.fname:
        windows = add_windows_from_file(fname=args.fname, **kwargs)

    else:
        raise Exception("one of box or fname must be specified")

    print(f"created {len(windows)} windows")


def add_apply_on_windows_args(parser: argparse.ArgumentParser):
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
        "--window", type=str, default=None, help="Only prepare this window"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes (default 0 to use main process only)",
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
    window: str | None = None,
    workers: int = 0,
    batch_size: int = 1,
    jobs_per_process: int | None = None,
    use_initial_job: bool = True,
):
    """A helper to apply a function on windows in a dataset.

    Args:
        f: the function to apply on lists of windows.
        dataset: the dataset.
        group: optional, only apply on windows in this group.
        window: optional, only apply on windows with this name.
        workers: the number of parallel workers to use, default 0 (main thread only).
        batch_size: if workers > 0, the maximum number of windows to pass to the
            function. If workers == 0, all windows are always passed.
        jobs_per_process: optional, terminate processes after they have handled this
            many jobs. This is useful if there is a memory leak in a dependency.
        use_initial_job: if workers > 0, by default, an initial job is run on the first
            batch in the main thread before spawning workers. This can handle things
            like building indexes that should not be done in parallel. Set this false
            to disable using the initial job.
    """
    if hasattr(f, "set_dataset"):
        f.set_dataset(dataset)

    print("Loading windows")
    groups = None
    names = None
    if group:
        groups = [group]
    if window:
        names = [window]
    windows = dataset.load_windows(groups=groups, names=names)
    print(f"found {len(windows)} windows")

    if hasattr(f, "get_jobs"):
        jobs = f.get_jobs(windows)
        print(f"got {len(jobs)} jobs")
    else:
        jobs = windows

    if workers == 0:
        f(jobs)
        return

    random.shuffle(jobs)

    if use_initial_job:
        # Apply directly on first window to get any initialization out of the way.
        f([jobs[0]])
        jobs = jobs[1:]

    batches = []
    for i in range(0, len(jobs), batch_size):
        batches.append(jobs[i : i + batch_size])

    p = multiprocessing.Pool(processes=workers, maxtasksperchild=jobs_per_process)
    outputs = p.imap_unordered(f, batches)
    for _ in tqdm.tqdm(outputs, total=len(batches)):
        pass
    p.close()


def apply_on_windows_args(f: Callable[[list[Window]], None], args: argparse.Namespace):
    """Call apply_on_windows with arguments passed via command-line interface."""
    dataset = Dataset(ds_root=args.root)
    apply_on_windows(
        f,
        dataset,
        args.group,
        args.window,
        args.workers,
        args.batch_size,
        args.jobs_per_process,
        args.use_initial_job,
    )


class PrepareHandler:
    """apply_on_windows handler for the rslearn dataset prepare command."""

    def __init__(self, force: bool):
        """Initialize a new PrepareHandler.

        Args:
            force: force prepare
        """
        self.force = force
        self.dataset = None

    def set_dataset(self, dataset: Dataset):
        """Captures the dataset from apply_on_windows_args.

        Args:
        dataset: the dataset to prepare.
        """
        self.dataset = dataset

    def __call__(self, windows: list[Window]):
        """Prepares the windows from apply_on_windows."""
        prepare_dataset_windows(self.dataset, windows, self.force)


@register_handler("dataset", "prepare")
def dataset_prepare():
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
    add_apply_on_windows_args(parser)
    args = parser.parse_args(args=sys.argv[3:])

    fn = PrepareHandler(args.force)
    apply_on_windows_args(fn, args)


class IngestHandler:
    """apply_on_windows handler for the rslearn dataset ingest command."""

    def __init__(self):
        """Initialize a new IngestHandler."""
        self.dataset = None

    def set_dataset(self, dataset: Dataset):
        """Captures the dataset from apply_on_windows_args.

        Args:
        dataset: the dataset to ingest.
        """
        self.dataset = dataset

    def __call__(self, jobs: list[tuple[str, Item, list[STGeometry]]]):
        """Ingest the specified items.

        The items are computed from list of windows via IngestHandler.get_jobs.

        Args:
            jobs: list of (layer_name, item, geometries) tuples to ingest.
        """
        import gc

        tile_store = self.dataset.get_tile_store()

        # Group jobs by layer name.
        jobs_by_layer = {}
        for layer_name, item, geometries in jobs:
            if layer_name not in jobs_by_layer:
                jobs_by_layer[layer_name] = []
            jobs_by_layer[layer_name].append((item, geometries))
        for layer_name, items_and_geometries in jobs_by_layer.items():
            cur_tile_store = PrefixedTileStore(tile_store, (layer_name,))
            layer_cfg = self.dataset.layers[layer_name]
            data_source = data_source_from_config(layer_cfg)

            try:
                data_source.ingest(
                    tile_store=cur_tile_store,
                    items=[item for item, _ in items_and_geometries],
                    geometries=[geometries for _, geometries in items_and_geometries],
                )
            except Exception as e:
                print(
                    "warning: got error while ingesting "
                    + f"{len(items_and_geometries)} items: {e}"
                )

        gc.collect()

    def get_jobs(
        self, windows: list[Window]
    ) -> list[tuple[str, Item, list[STGeometry]]]:
        """Computes ingest jobs from window list.

        Each ingest job is a tuple of the layer name, the item to ingest, and the
        geometries of windows that require that item.

        This makes sure that jobs are grouped by item rather than by window, which
        makes sense because there's no reason to ingest the same item twice.
        """
        # TODO: avoid duplicating ingest_dataset_windows...
        jobs: list[tuple[str, Item, list[STGeometry]]] = []
        for layer_name, layer_cfg in self.dataset.layers.items():
            if not layer_cfg.data_source:
                continue
            if not layer_cfg.data_source.ingest:
                continue

            data_source = data_source_from_config(layer_cfg)

            geometries_by_item = {}
            for window in windows:
                layer_datas = window.load_layer_datas()
                if layer_name not in layer_datas:
                    continue
                geometry = window.get_geometry()
                layer_data = layer_datas[layer_name]
                for group in layer_data.serialized_item_groups:
                    for serialized_item in group:
                        item = data_source.deserialize_item(serialized_item)
                        if item not in geometries_by_item:
                            geometries_by_item[item] = []
                        geometries_by_item[item].append(geometry)

            for item, geometries in geometries_by_item.items():
                jobs.append((layer_name, item, geometries))

        print(f"computed {len(jobs)} ingest jobs from {len(windows)} windows")
        return jobs


@register_handler("dataset", "ingest")
def dataset_ingest():
    """Handler for the rslearn dataset ingest command."""
    parser = argparse.ArgumentParser(
        prog="rslearn dataset ingest",
        description="rslearn dataset ingest: ingest items in retrieved data sources",
    )
    add_apply_on_windows_args(parser)
    args = parser.parse_args(args=sys.argv[3:])

    fn = IngestHandler()
    apply_on_windows_args(fn, args)


class MaterializeHandler:
    """apply_on_windows handler for the rslearn dataset materialize command."""

    def __init__(self):
        """Initialize a MaterializeHandler."""
        self.dataset = None

    def set_dataset(self, dataset: Dataset):
        """Captures the dataset from apply_on_windows_args.

        Args:
        dataset: the dataset to prepare.
        """
        self.dataset = dataset

    def __call__(self, windows: list[Window]):
        """Materializes the windows from apply_on_windows."""
        materialize_dataset_windows(self.dataset, windows)


@register_handler("dataset", "materialize")
def dataset_materialize():
    """Handler for the rslearn dataset materialize command."""
    parser = argparse.ArgumentParser(
        prog="rslearn dataset materialize",
        description=(
            "rslearn dataset materialize: "
            + "materialize data from retrieved data sources"
        ),
    )
    add_apply_on_windows_args(parser)
    args = parser.parse_args(args=sys.argv[3:])

    fn = MaterializeHandler()
    apply_on_windows_args(fn, args)


class RslearnLightningCLI(LightningCLI):
    """LightningCLI that links data.tasks to model.tasks."""

    def add_arguments_to_parser(self, parser) -> None:
        """Link data.tasks to model.tasks.

        Args:
            parser: the argument parser
        """
        parser.link_arguments(
            "data.init_args.task", "model.init_args.task", apply_on="instantiate"
        )
        parser.add_argument(
            "--wandb_run_id",
            default="",
            type=str,
            help="W&B run ID to load checkpoint from",
        )
        parser.add_argument(
            "--wandb_resume",
            default=False,
            type=bool,
            help="Whether to resume from specified wandb_run_id",
        )

    def before_instantiate_classes(self):
        """Called before Lightning class initialization.

        Sets up wandb_run_id / wandb_resume arguments.
        """
        subcommand = self.config.subcommand
        c = self.config[subcommand]

        if c.wandb_run_id:
            api = wandb.Api()
            artifact_id = (
                f"{c.trainer.logger.init_args.project}/model-{c.wandb_run_id}:latest"
            )
            print(f"restoring from artifact {artifact_id} on wandb")
            artifact = api.artifact(artifact_id, type="model")
            artifact_dir = artifact.download()
            c.ckpt_path = str(Path(artifact_dir) / "model.ckpt")

        if c.wandb_resume:
            c.trainer.logger.init_args.id = c.wandb_run_id


def model_handler():
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
def model_fit():
    """Handler for rslearn model fit."""
    model_handler()


@register_handler("model", "validate")
def model_validate():
    """Handler for rslearn model validate."""
    model_handler()


@register_handler("model", "test")
def model_test():
    """Handler for rslearn model test."""
    model_handler()


@register_handler("model", "predict")
def model_predict():
    """Handler for rslearn model predict."""
    model_handler()


def main():
    """CLI entrypoint."""
    multiprocessing.set_start_method("forkserver")
    parser = argparse.ArgumentParser(description="rslearn")
    parser.add_argument(
        "category", help="Command category: dataset, annotate, or model"
    )
    parser.add_argument("command", help="The command to run")
    args = parser.parse_args(args=sys.argv[1:3])

    handler = handler_registry.get((args.category, args.command))
    if handler is None:
        print(f"Unknown command: {args.category} {args.command}", file=sys.stderr)
        sys.exit(1)

    handler()


if __name__ == "__main__":
    main()
