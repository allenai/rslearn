#!/usr/bin/env python

import argparse
import multiprocessing
import random
import sys
from datetime import datetime, timezone
from typing import Callable, Optional

import tqdm
from rasterio.crs import CRS

from rslearn.const import WGS84_EPSG
from rslearn.dataset import (
    Dataset,
    Window,
    ingest_dataset_windows,
    materialize_dataset_windows,
    prepare_dataset_windows,
)
from rslearn.dataset.add_windows import add_windows_from_box, add_windows_from_file
from rslearn.utils import Projection

handler_registry = {}


def register_handler(category, command):
    def decorator(f):
        handler_registry[(category, command)] = f
        return f

    return decorator


def parse_time(time_str: str) -> datetime:
    ts = datetime.fromisoformat(time_str)
    if not ts.tzinfo:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def parse_time_range(
    start: Optional[str], end: Optional[str]
) -> Optional[tuple[datetime, datetime]]:
    if not start or not end:
        return None
    return (parse_time(start), parse_time(end))


@register_handler("dataset", "add_windows")
def add_windows():
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
            + "create windows along a grid of this cell size",
        ),
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help=(
            "Instead of creating windows the size of the input extents, "
            + "create windows of this fixed size centered at each extent's center",
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
        windows = add_windows_from_file(
            fname=args.fname,
            **kwargs,
        )

    else:
        raise Exception("one of box or fname must be specified")

    print(f"created {len(windows)} windows")


def add_apply_on_windows_args(parser):
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
    group: Optional[str] = None,
    window: Optional[str] = None,
    workers: int = 0,
    batch_size: int = 1,
    jobs_per_process: int = None,
    use_initial_job: bool = True,
):
    print("Loading windows")
    groups = None
    names = None
    if group:
        groups = [group]
    if window:
        names = [window]
    windows = dataset.load_windows(groups=groups, names=names)
    print(f"found {len(windows)} windows")

    if workers == 0:
        f(windows)
        return

    random.shuffle(windows)

    if use_initial_job:
        # Apply directly on first window to get any initialization out of the way.
        f([windows[0]])
        windows = windows[1:]

    batches = []
    for i in range(0, len(windows), batch_size):
        batches.append(windows[i : i + batch_size])

    p = multiprocessing.Pool(processes=workers, maxtasksperchild=jobs_per_process)
    outputs = p.imap_unordered(f, batches)
    for _ in tqdm.tqdm(outputs, total=len(batches)):
        pass
    p.close()


def apply_on_windows_args(f: Callable[[list[Window]], None], args: argparse.Namespace):
    dataset = Dataset(ds_root=args.root)
    if hasattr(f, "set_dataset"):
        f.set_dataset(dataset)
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
    def __init__(self, force: bool):
        self.force = force
        self.dataset = None

    def set_dataset(self, dataset: Dataset):
        self.dataset = dataset

    def __call__(self, windows: list[Window]):
        prepare_dataset_windows(self.dataset, windows, self.force)


@register_handler("dataset", "prepare")
def dataset_prepare():
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
    def __init__(self):
        self.dataset = None

    def set_dataset(self, dataset: Dataset):
        self.dataset = dataset

    def __call__(self, windows: list[Window]):
        import gc

        ingest_dataset_windows(self.dataset, windows)
        gc.collect()


@register_handler("dataset", "ingest")
def dataset_ingest():
    parser = argparse.ArgumentParser(
        prog="rslearn dataset ingest",
        description="rslearn dataset ingest: ingest items in retrieved data sources",
    )
    add_apply_on_windows_args(parser)
    args = parser.parse_args(args=sys.argv[3:])

    fn = IngestHandler()
    apply_on_windows_args(fn, args)


class MaterializeHandler:
    def __init__(self):
        self.dataset = None

    def set_dataset(self, dataset: Dataset):
        self.dataset = dataset

    def __call__(self, windows: list[Window]):
        materialize_dataset_windows(self.dataset, windows)


@register_handler("dataset", "materialize")
def dataset_materialize():
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


def main():
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
