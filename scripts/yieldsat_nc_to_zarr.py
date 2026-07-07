"""Convert YieldSAT preprocessed NetCDF files to chunked Zarr stores.

The YieldSAT preprocessed NetCDF files are large HDF5/NetCDF4 files with a layout like:

    sample(index, time_step, band)
    target(index)

This script rechunks that layout along the sample/index axis and writes a
directory-style Zarr store. The output path may be local or an fsspec URL such as
``s3://bucket/path.zarr``.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import xarray as xr


def _parse_json_arg(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if value.startswith("@"):
        with open(value[1:]) as f:
            return json.load(f)
    return json.loads(value)


def _extract_zip_member(zip_path: str, member: str, extract_dir: str) -> str:
    os.makedirs(extract_dir, exist_ok=True)
    out_path = os.path.join(extract_dir, member)
    if os.path.exists(out_path):
        return out_path

    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(member, path=extract_dir)
    return out_path


def _get_input_path(args: argparse.Namespace) -> tuple[str, tempfile.TemporaryDirectory[str] | None]:
    if args.input_nc:
        return args.input_nc, None

    if not args.input_zip or not args.zip_member:
        raise ValueError("provide either --input-nc or both --input-zip and --zip-member")

    if args.extract_dir:
        return _extract_zip_member(args.input_zip, args.zip_member, args.extract_dir), None

    tmp = tempfile.TemporaryDirectory(prefix="yieldsat_nc_")
    return _extract_zip_member(args.input_zip, args.zip_member, tmp.name), tmp


def _build_encoding(ds: xr.Dataset, compressor: Any | None) -> dict[str, dict[str, Any]]:
    encoding: dict[str, dict[str, Any]] = {}
    for name, var in ds.variables.items():
        if name in ds.dims:
            continue
        cur: dict[str, Any] = {}
        if compressor is not None and var.dtype.kind in {"f", "i", "u", "b"}:
            cur["compressor"] = compressor
        encoding[name] = cur
    return encoding


def convert(args: argparse.Namespace) -> None:
    input_path, tmp = _get_input_path(args)
    storage_options = _parse_json_arg(args.storage_options)

    try:
        ds = xr.open_dataset(input_path, chunks={})

        if args.drop_global_attrs:
            ds.attrs = {}

        required = {"sample", "target"}
        missing = required - set(ds.variables)
        if missing:
            raise ValueError(f"input dataset is missing required variables: {sorted(missing)}")

        index_chunk = args.index_chunk
        time_chunk = ds.sizes["time_step"] if args.time_chunk is None else args.time_chunk
        band_chunk = ds.sizes["band"] if args.band_chunk is None else args.band_chunk

        chunks: dict[str, int] = {"index": index_chunk}
        if "time_step" in ds.dims:
            chunks["time_step"] = time_chunk
        if "band" in ds.dims:
            chunks["band"] = band_chunk
        ds = ds.chunk(chunks)

        compressor = None
        if args.compressor != "none":
            from numcodecs import Blosc

            compressor = Blosc(
                cname=args.compressor,
                clevel=args.compression_level,
                shuffle=Blosc.SHUFFLE,
            )

        encoding = _build_encoding(ds, compressor)

        ds.to_zarr(
            args.output,
            mode="w",
            consolidated=True,
            storage_options=storage_options,
            encoding=encoding,
            zarr_format=2,
        )
    finally:
        if tmp is not None:
            tmp.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-nc", help="Path to an extracted YieldSAT .nc file.")
    source.add_argument("--input-zip", help="Path to Preprocessed.zip.")
    parser.add_argument(
        "--zip-member",
        help="Member inside --input-zip, e.g. Preprocessed/Germany/merge_s2-soil-dem-weather-coords.nc.",
    )
    parser.add_argument(
        "--extract-dir",
        help="Directory to extract --zip-member into. Recommended for large files.",
    )
    parser.add_argument("--output", required=True, help="Output Zarr path, local or s3://...")
    parser.add_argument(
        "--storage-options",
        help='JSON object, or @path/to/options.json, passed to xarray/fsspec for output. Example: \'{"anon": false}\'.',
    )
    parser.add_argument(
        "--index-chunk",
        type=int,
        default=256,
        help="Chunk size along sample/index axis. 128-1024 is usually reasonable for S3 training.",
    )
    parser.add_argument("--time-chunk", type=int, default=None)
    parser.add_argument("--band-chunk", type=int, default=None)
    parser.add_argument(
        "--compressor",
        default="zstd",
        choices=["zstd", "lz4", "blosclz", "zlib", "none"],
    )
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument(
        "--drop-global-attrs",
        action="store_true",
        help="Drop large global attrs from the NetCDF before writing Zarr.",
    )
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
