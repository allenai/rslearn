"""Unit tests for rslearn.train.transforms.sentinel2."""

import torch

from rslearn.train.model_context import RasterImage
from rslearn.train.transforms.mask import Mask
from rslearn.train.transforms.sentinel2 import Sentinel2SCLToMask


def test_sentinel2_scl_to_mask_and_apply_mask() -> None:
    scl_to_mask = Sentinel2SCLToMask(exclude_scl_values=[3, 8])
    apply_mask = Mask(selectors=["image"], mask_selector="mask", mask_value=0)

    input_dict = {
        "image": RasterImage(torch.full((1, 1, 2, 2), 5.0)),
        "scl": RasterImage(torch.tensor([[[[0, 8], [3, 1]]]], dtype=torch.int32)),
    }

    input_dict, _ = scl_to_mask(input_dict, {})
    assert torch.all(
        input_dict["mask"].image == torch.tensor([[[[1, 0], [0, 1]]]], dtype=torch.int32)
    )

    input_dict, _ = apply_mask(input_dict, {})
    assert torch.all(input_dict["image"].image == torch.tensor([[[[5, 0], [0, 5]]]]))

