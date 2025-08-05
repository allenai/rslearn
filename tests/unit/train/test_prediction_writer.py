"""Test rslearn.train.prediction_writer."""

import json
import pathlib

import numpy as np
import torch
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.train.lightning_module import RslearnLightningModule
from rslearn.train.prediction_writer import (
    PendingPatchOutput,
    RasterMerger,
    RslearnWriter,
)
from rslearn.train.tasks.segmentation import SegmentationTask
from rslearn.utils.geometry import Projection


class TestRasterMerger:
    """Unit tests for RasterMerger."""

    def test_merge_no_padding(self, tmp_path: pathlib.Path) -> None:
        """Verify patches are merged when no padding is used.

        We make four 3x3 patches to cover a 4x4 window.
        """
        window = Window(
            path=UPath(tmp_path),
            group="fake",
            name="fake",
            projection=WGS84_PROJECTION,
            bounds=(0, 0, 4, 4),
            time_range=None,
        )
        outputs = [
            PendingPatchOutput(
                bounds=(0, 0, 3, 3),
                output=0 * np.ones((1, 3, 3), dtype=np.uint8),
            ),
            PendingPatchOutput(
                bounds=(0, 3, 3, 6),
                output=1 * np.ones((1, 3, 3), dtype=np.uint8),
            ),
            PendingPatchOutput(
                bounds=(3, 0, 6, 3),
                output=2 * np.ones((1, 3, 3), dtype=np.uint8),
            ),
            PendingPatchOutput(
                bounds=(3, 3, 6, 6),
                output=3 * np.ones((1, 3, 3), dtype=np.uint8),
            ),
        ]
        merged = RasterMerger().merge(window, outputs)
        assert merged.shape == (1, 4, 4)
        assert merged.dtype == np.uint8
        # The patches were disjoint, so we just check that those portions of the merged
        # image have the right value.
        assert np.all(merged[0, 0:3, 0:3] == 0)
        assert np.all(merged[0, 3:4, 0:3] == 1)
        assert np.all(merged[0, 0:3, 3:4] == 2)
        assert np.all(merged[0, 3, 3] == 3)

    def test_merge_with_padding(self, tmp_path: pathlib.Path) -> None:
        """Verify merging works with padding."""
        window = Window(
            path=UPath(tmp_path),
            group="fake",
            name="fake",
            projection=WGS84_PROJECTION,
            bounds=(0, 0, 4, 4),
            time_range=None,
        )
        # We make four 3x3 patches:
        # - (0, 0, 3, 3)
        # - (0, 1, 3, 4)
        # - (1, 0, 4, 3)
        # - (1, 1, 4, 4)
        # There are 2 shared pixels between overlapping patches so we set padding=1.
        outputs = [
            PendingPatchOutput(
                bounds=(0, 0, 3, 3),
                output=0 * np.ones((1, 3, 3), dtype=np.int32),
            ),
            PendingPatchOutput(
                bounds=(0, 1, 3, 4),
                output=1 * np.ones((1, 3, 3), dtype=np.int32),
            ),
            PendingPatchOutput(
                bounds=(1, 0, 4, 3),
                output=2 * np.ones((1, 3, 3), dtype=np.int32),
            ),
            PendingPatchOutput(
                bounds=(1, 1, 4, 4),
                output=3 * np.ones((1, 3, 3), dtype=np.int32),
            ),
        ]
        merged = RasterMerger(padding=1).merge(window, outputs)
        assert merged.shape == (1, 4, 4)
        assert merged.dtype == np.int32
        # The top-left should use the first patch.
        assert np.all(merged[0, 0:2, 0:2] == 0)
        # The bottom-left should use the second patch.
        assert np.all(merged[0, 2:4, 0:2] == 1)
        # The top-right should use the third patch.
        assert np.all(merged[0, 0:2, 2:4] == 2)
        # And finally the bottom-right should use the fourth patch.
        assert np.all(merged[0, 2:4, 2:4] == 3)


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
        "time_range": window.time_range,
        "patch_idx": 0,
        "num_patches": 1,
    }
    # batch is (inputs, targets, metadatas) but writer only uses the metadatas.
    batch = ([None], [None], [metadata])
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
