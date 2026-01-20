"""Tests for rslearn.train.tasks.task."""

import torch

from rslearn.train.model_context import RasterImage
from rslearn.train.tasks.task import BasicTask


def test_basic_task_visualize_with_raster_image() -> None:
    """Verify BasicTask.visualize returns uint8 HWC tensor given RasterImage."""
    task = BasicTask(image_bands=(0, 1, 2))

    # Create a RasterImage with CTHW format
    image_tensor = torch.randint(0, 256, (3, 1, 8, 8), dtype=torch.float32)
    raster_image = RasterImage(image=image_tensor)

    input_dict = {"image": raster_image}

    # Call visualize
    result = task.visualize(input_dict, None, None)

    # Verify the result
    assert "image" in result
    assert result["image"].shape == (8, 8, 3)  # HWC format
    assert result["image"].dtype.name == "uint8"
