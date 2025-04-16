"""Test rslearn.train.prediction_writer."""

import json
import pathlib

import torch
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.train.lightning_module import RslearnLightningModule
from rslearn.train.prediction_writer import RslearnWriter
from rslearn.train.tasks.segmentation import SegmentationTask
from rslearn.utils.geometry import Projection


def test_write_raster(tmp_path: pathlib.Path) -> None:
    output_layer_name = "output"
    output_bands = ["value"]

    # Initialize dataset.
    ds_path = UPath(tmp_path)
    ds_config = {
        "layers": {
            output_layer_name: {
                "type": "raster",
                "band_sets": [
                    {
                        "dtype": "uint8",
                        "bands": output_bands,
                    }
                ],
            }
        }
    }
    with (ds_path / "config.json").open("w") as f:
        json.dump(ds_config, f)

    # Create the window.
    window_name = "default"
    window_group = "default"
    window = Window(
        path=Window.get_window_root(ds_path, window_name, window_group),
        group=window_group,
        name=window_name,
        projection=Projection(WGS84_PROJECTION.crs, 0.2, 0.2),
        bounds=(0, 0, 1, 1),
        time_range=None,
    )
    window.save()

    # Initialize prediction writer.
    task = SegmentationTask(num_classes=2)
    pl_module = RslearnLightningModule(
        model=torch.nn.Identity(),
        task=task,
    )
    writer = RslearnWriter(
        path=str(tmp_path),
        output_layer=output_layer_name,
    )

    # Write predictions.
    metadata = {
        "window_name": window.name,
        "group": window.group,
        "bounds": window.bounds,
        "window_bounds": window.bounds,
        "projection": window.projection,
        "patch_idx": 0,
        "num_patches": 1,
    }
    # batch is (inputs, targets, metadatas) but writer only uses the metadatas.
    batch = (None, None, [metadata])
    # output for segmentation task is CHW where C axis contains per-class
    # probabilities.
    output = torch.zeros((2, 5, 5), dtype=torch.float32)
    writer.write_on_batch_end(
        trainer=None,
        pl_module=pl_module,
        prediction=[output],
        batch_indices=[0],
        batch=batch,
        batch_idx=0,
        dataloader_idx=0,
    )

    # Ensure the output is written.
    expected_fname = (
        window.get_raster_dir(output_layer_name, output_bands, 0) / "geotiff.tif"
    )
    assert expected_fname.exists()
    assert window.is_layer_completed(output_layer_name)
