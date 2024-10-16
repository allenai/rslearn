"""Inference requests."""

import io
import json
import os
import shutil
import tarfile
from datetime import datetime
from typing import BinaryIO

from upath import UPath

from rslearn.dataset import Dataset, Window
from rslearn.main import (
    IngestHandler,
    MaterializeHandler,
    PrepareHandler,
    RslearnLightningCLI,
    apply_on_windows,
)
from rslearn.train.data_module import RslearnDataModule
from rslearn.train.lightning_module import RslearnLightningModule
from rslearn.utils import PixelBounds, Projection


class Request:
    """An inference request.

    Specifies a spatiotemporal window to apply the model on. And optionally includes
    the data in dataset layers needed to complete the inference.

    Can be serialized to/from bytes (encoded as a tarfile).
    """

    def __init__(
        self,
        projection: Projection,
        bounds: PixelBounds,
        time_range: tuple[datetime, datetime] | None,
        datas: dict[str, str] = {},
        ingest=True,
    ):
        """Creates a new Request.

        Args:
            projection: the window's projection.
            bounds: the window's bounds.
            time_range: the window's time range.
            datas: optional data included with the request. Maps from a target path in
                the window to the local filesystem path.
            ingest: whether to ingest data for this window. If all inputs needed to run
                the model are included in raster_datas and vector_datas, then this
                should be set False.
        """
        self.projection = projection
        self.bounds = bounds
        self.time_range = time_range
        self.datas = datas
        self.ingest = ingest

    def serialize(self, buf: BinaryIO):
        """Serialize the request to the specified buffer.

        Args:
            buf: the buffer to write to.
        """
        with tarfile.open(fileobj=buf, mode="w") as tarf:
            # Write metadata.
            meta_dict = {
                "projection": self.projection.serialize(),
                "bounds": self.bounds,
                "time_range": (
                    self.time_range[0].isoformat(),
                    self.time_range[1].isoformat(),
                ),
                "ingest": self.ingest,
            }
            meta_buf = io.BytesIO()
            meta_buf.write(json.dumps(meta_dict).encode())

            entry = tarfile.TarInfo(
                name="metadata.json",
            )
            entry.size = meta_buf.getbuffer().nbytes
            entry.mode = 0o644
            meta_buf.seek(0)
            tarf.addfile(entry, fileobj=meta_buf)

            # Write rasters.
            for arc_path, local_path in self.datas.items():
                tarf.add(
                    name=local_path,
                    arcpath=arc_path,
                )

    @staticmethod
    def unserialize(buf: BinaryIO, scratch_dir: str) -> "Request":
        """Unserialize this request.

        Args:
            buf: the buffer to read from.
            scratch_dir: directory to write data in dataset layers.
        """
        raise NotImplementedError


def serve(
    ds_cfg_fname: str,
    model_cfg_fname: str,
    request: Request,
    scratch_dir: str,
    workers: int,
):
    """Run prediction for the specified request.

    Args:
        ds_cfg_fname: the dataset configuration.
        model_cfg_fname: the model configuration.
        request: the request to handle.
        scratch_dir: scratch directory to use as temporary dataset root directory.
        workers: number of workers to use.
    """
    shutil.copyfile(ds_cfg_fname, os.path.join(scratch_dir, "config.json"))
    dataset = Dataset(UPath(scratch_dir))
    group = "default"
    window_name = "window"
    window = Window(
        path=UPath(os.path.join(scratch_dir, "windows", group, window_name)),
        group=group,
        name=window_name,
        projection=request.projection,
        bounds=request.bounds,
        time_range=request.time_range,
    )
    window.save()

    if request.ingest:
        apply_on_windows(
            PrepareHandler(force=False),
            dataset,
            workers=workers,
            group=group,
        )
        apply_on_windows(
            IngestHandler(),
            dataset,
            workers=workers,
            use_initial_job=False,
            jobs_per_process=1,
            group=group,
        )
        apply_on_windows(
            MaterializeHandler(),
            dataset,
            workers=workers,
            use_initial_job=False,
            group=group,
        )

    RslearnLightningCLI(
        model_class=RslearnLightningModule,
        datamodule_class=RslearnDataModule,
        args=[
            "model",
            "predict",
            "--config",
            model_cfg_fname,
            "--autoresume=true",
            "--data.init_args.path",
            scratch_dir,
        ],
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
    )
