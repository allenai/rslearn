"""Tests for RslearnLightningModule."""

import pathlib

import torch

from rslearn.const import WGS84_PROJECTION
from rslearn.models.component import FeatureExtractor, FeatureMaps
from rslearn.models.singletask import SingleTaskModel
from rslearn.train.lightning_module import RslearnLightningModule
from rslearn.train.model_context import (
    ModelContext,
    RasterImage,
    SampleMetadata,
)
from rslearn.train.tasks.segmentation import SegmentationHead, SegmentationTask


class FakeEncoder(FeatureExtractor):
    """A simple encoder that passes through the input."""

    def forward(self, context: ModelContext) -> FeatureMaps:
        """Stack the input images and return as is."""
        batch_images = [
            inp["image"].single_ts_to_chw_tensor() for inp in context.inputs
        ]
        return FeatureMaps([torch.stack(batch_images)])


def test_test_step_produces_visualization(tmp_path: pathlib.Path) -> None:
    """Verify test_step produces visualization images when visualize_dir is set."""
    # Input image will have this many channels and logits will too.
    num_classes = 3
    image_size = 8

    # Create task and model
    task = SegmentationTask(
        num_classes=num_classes,
        image_bands=(0, 1, 2),
    )
    model = SingleTaskModel(
        encoder=[FakeEncoder()],
        decoder=[SegmentationHead()],
    )

    # Create visualize directory
    visualize_dir = tmp_path / "visualize"
    visualize_dir.mkdir()

    # Create lightning module with visualize_dir
    lightning_module = RslearnLightningModule(
        model=model,
        task=task,
        visualize_dir=str(visualize_dir),
    )

    # Create input and target dicts for the batch.
    inputs = [
        {
            "image": RasterImage(
                torch.randint(
                    0,
                    256,
                    (num_classes, 1, image_size, image_size),
                    dtype=torch.float32,
                )
            )
        }
    ]
    targets = [
        {
            "classes": torch.randint(0, num_classes, (image_size, image_size)).long(),
            "valid": torch.ones((image_size, image_size), dtype=torch.float32),
        }
    ]

    # Create SampleMetadata
    metadatas = [
        SampleMetadata(
            window_group="test_group",
            window_name="test_window",
            window_bounds=(0, 0, image_size, image_size),
            patch_bounds=(0, 0, image_size, image_size),
            patch_idx=0,
            num_patches_in_window=1,
            time_range=None,
            projection=WGS84_PROJECTION,
            dataset_source=None,
        )
    ]

    # Run test_step
    lightning_module.test_step((inputs, targets, metadatas), batch_idx=0)

    # Verify that visualization files are created
    expected_files = [
        "test_window_0_0_image.png",
        "test_window_0_0_gt.png",
        "test_window_0_0_pred.png",
    ]
    for filename in expected_files:
        filepath = visualize_dir / filename
        assert filepath.exists(), f"Expected visualization file {filename} not found"
