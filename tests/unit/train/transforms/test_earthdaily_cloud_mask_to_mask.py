"""Unit tests for rslearn.train.transforms.earthdaily."""

import pytest
import torch

from rslearn.train.model_context import RasterImage
from rslearn.train.transforms.earthdaily import EarthDailyCloudMaskToMask
from rslearn.train.transforms.mask import Mask


def test_earthdaily_cloud_mask_to_mask() -> None:
    cloud_mask_to_mask = EarthDailyCloudMaskToMask()

    input_dict = {
        "cloud_mask": RasterImage(
            torch.tensor([[[[0, 1, 2], [3, 4, 1]]]], dtype=torch.uint8)
        ),
    }

    input_dict, _ = cloud_mask_to_mask(input_dict, {})

    assert torch.all(
        input_dict["mask"].image
        == torch.tensor([[[[0, 1, 0], [0, 0, 1]]]], dtype=torch.int32)
    )


def test_earthdaily_cloud_mask_to_mask_and_apply_mask() -> None:
    cloud_mask_to_mask = EarthDailyCloudMaskToMask()
    apply_mask = Mask(selectors=["image"], mask_selector="mask", mask_value=0)

    input_dict = {
        "image": RasterImage(torch.full((2, 2, 2, 2), 5.0)),
        "cloud_mask": RasterImage(
            torch.tensor([[[[1, 2], [3, 4]]]], dtype=torch.uint8)
        ),
    }

    input_dict, _ = cloud_mask_to_mask(input_dict, {})
    input_dict, _ = apply_mask(input_dict, {})

    expected = torch.tensor(
        [
            [
                [[5, 0], [0, 0]],
                [[5, 0], [0, 0]],
            ],
            [
                [[5, 0], [0, 0]],
                [[5, 0], [0, 0]],
            ],
        ],
        dtype=torch.float32,
    )
    assert torch.all(input_dict["image"].image == expected)


def test_earthdaily_cloud_mask_to_mask_custom_clear_values() -> None:
    cloud_mask_to_mask = EarthDailyCloudMaskToMask(clear_values=[1, 4])

    input_dict = {
        "cloud_mask": RasterImage(
            torch.tensor([[[[0, 1, 2], [3, 4, 1]]]], dtype=torch.uint8)
        ),
    }

    input_dict, _ = cloud_mask_to_mask(input_dict, {})

    assert torch.all(
        input_dict["mask"].image
        == torch.tensor([[[[0, 1, 0], [0, 1, 1]]]], dtype=torch.int32)
    )


def test_earthdaily_cloud_mask_to_mask_requires_single_band() -> None:
    cloud_mask_to_mask = EarthDailyCloudMaskToMask()

    input_dict = {
        "cloud_mask": RasterImage(torch.zeros((2, 1, 2, 2), dtype=torch.uint8)),
    }

    with pytest.raises(ValueError, match="one band"):
        cloud_mask_to_mask(input_dict, {})
