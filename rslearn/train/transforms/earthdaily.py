"""Transforms related to EarthDaily data."""

from typing import Any

import torch

from rslearn.train.model_context import RasterImage

from .transform import Transform, read_selector, selector_exists, write_selector


class EarthDailyCloudMaskToMask(Transform):
    """Convert an EarthDaily EDA cloud-mask raster into a binary mask raster.

    The output mask is 1 for clear pixels and 0 for pixels that should be masked out.
    EarthDaily EDA cloud-mask values are:

    - 0: nodata
    - 1: clear
    - 2: cloud
    - 3: cloud shadow
    - 4: thin cloud
    """

    DEFAULT_CLEAR_VALUES = [1]

    def __init__(
        self,
        cloud_mask_selector: str = "cloud_mask",
        output_selector: str = "mask",
        clear_values: list[int] | None = None,
        skip_missing: bool = False,
    ) -> None:
        """Initialize a new EarthDailyCloudMaskToMask.

        Args:
            cloud_mask_selector: selector for the EarthDaily EDA cloud-mask image.
            output_selector: selector to write the binary mask to.
            clear_values: cloud-mask values to treat as clear. Defaults to [1].
            skip_missing: if True, skip when cloud_mask_selector is missing.
        """
        super().__init__(skip_missing=skip_missing)
        self.cloud_mask_selector = cloud_mask_selector
        self.output_selector = output_selector
        self.clear_values = (
            list(self.DEFAULT_CLEAR_VALUES)
            if clear_values is None
            else list(clear_values)
        )

    def _to_mask(self, cloud_mask: RasterImage) -> RasterImage:
        cloud_mask_tensor = cloud_mask.image
        if cloud_mask_tensor.shape[0] != 1:
            raise ValueError("expected EarthDaily cloud-mask image to have one band")

        clear = torch.zeros_like(cloud_mask_tensor, dtype=torch.bool)
        for value in self.clear_values:
            clear |= cloud_mask_tensor == value
        mask = clear.to(dtype=torch.int32)
        return RasterImage(mask, cloud_mask.timestamps)

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Write a binary mask derived from EarthDaily EDA cloud-mask values."""
        if self.skip_missing and not selector_exists(
            input_dict, target_dict, self.cloud_mask_selector
        ):
            return input_dict, target_dict

        cloud_mask = read_selector(input_dict, target_dict, self.cloud_mask_selector)
        mask = self._to_mask(cloud_mask)
        write_selector(input_dict, target_dict, self.output_selector, mask)
        return input_dict, target_dict
