#!/usr/bin/env python

import argparse
import sys

from rslearn.dataset import Dataset, ingest_dataset_windows, prepare_dataset_windows

handler_registry = {}


def register_handler(category, command):
    def decorator(f):
        handler_registry[(category, command)] = f
        return f

    return decorator


@register_handler("dataset", "prepare")
def dataset_prepare():
    parser = argparse.ArgumentParser(
        prog="rslearn dataset prepare",
        description="rslearn dataset prepare: lookup items in retrieved data sources",
    )
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
        "--force",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Prepare windows even if they were previously prepared",
    )
    args = parser.parse_args(args=sys.argv[3:])

    dataset = Dataset(ds_root=args.root)

    print("Loading windows")
    groups = None
    names = None
    if args.group:
        groups = [args.group]
    if args.window:
        names = [args.window]
    windows = dataset.load_windows(groups=groups, names=names)
    print(f"found {len(windows)} windows")

    prepare_dataset_windows(dataset, windows, force=args.force)


@register_handler("dataset", "ingest")
def dataset_ingest():
    parser = argparse.ArgumentParser(
        prog="rslearn dataset ingest",
        description="rslearn dataset ignest: ingest items in retrieved data sources",
    )
    parser.add_argument(
        "--root", type=str, required=True, help="Dataset root directory"
    )
    parser.add_argument(
        "--group", type=str, default=None, help="Only prepare windows in this group"
    )
    parser.add_argument(
        "--window", type=str, default=None, help="Only prepare this window"
    )
    args = parser.parse_args(args=sys.argv[3:])

    dataset = Dataset(ds_root=args.root)

    print("Loading windows")
    groups = None
    names = None
    if args.group:
        groups = [args.group]
    if args.window:
        names = [args.window]
    windows = dataset.load_windows(groups=groups, names=names)
    print(f"found {len(windows)} windows")

    ingest_dataset_windows(dataset, windows)


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
