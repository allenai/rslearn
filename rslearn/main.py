#!/usr/bin/env python

import argparse
import multiprocessing
import random
import sys
from typing import Callable, Optional

import tqdm

from rslearn.dataset import (
    Dataset,
    Window,
    ingest_dataset_windows,
    materialize_dataset_windows,
    prepare_dataset_windows,
)

handler_registry = {}


def register_handler(category, command):
    def decorator(f):
        handler_registry[(category, command)] = f
        return f

    return decorator


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
