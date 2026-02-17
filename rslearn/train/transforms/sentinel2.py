"""Transforms related to Sentinel-2 data."""

from typing import Any

import torch

from rslearn.train.model_context import RasterImage

from .transform import Transform, read_selector, selector_exists, write_selector


class Sentinel2SCLToMask(Transform):
    """Convert a Sentinel-2 SCL raster into a binary mask raster.

    The output mask is 1 where pixels are considered valid and 0 where pixels should
    be masked out (e.g., clouds). This is intended to be used together with the
    `rslearn.train.transforms.mask.Mask` transform.
    """

    DEFAULT_EXCLUDE_SCL_VALUES = [3, 8, 9, 10]

    def __init__(
        self,
        scl_selector: str = "scl",
        output_selector: str = "mask",
        exclude_scl_values: list[int] | None = None,
        skip_missing: bool = False,
    ) -> None:
        """Initialize a new Sentinel2SCLToMask.

        Args:
            scl_selector: selector for the SCL image (typically a single band).
            output_selector: selector to write the binary mask to.
            exclude_scl_values: SCL values to treat as invalid (defaults to common
                cloud/cloud-shadow/cirrus values).
            skip_missing: if True, skip when scl_selector is missing.
        """
        super().__init__()
        self.scl_selector = scl_selector
        self.output_selector = output_selector
        self.exclude_scl_values = (
            list(self.DEFAULT_EXCLUDE_SCL_VALUES)
            if exclude_scl_values is None
            else list(exclude_scl_values)
        )
        self.skip_missing = skip_missing

    def _to_mask(self, scl: RasterImage) -> RasterImage:
        scl_tensor = scl.image
        if scl_tensor.shape[0] != 1:
            raise ValueError("expected SCL image to have exactly one band")

        invalid = torch.zeros_like(scl_tensor, dtype=torch.bool)
        for v in self.exclude_scl_values:
            invalid |= scl_tensor == v
        mask = (~invalid).to(dtype=torch.int32)
        return RasterImage(mask, scl.timestamps)

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Write a binary mask derived from SCL into the input/target dicts."""
        if self.skip_missing and not selector_exists(input_dict, target_dict, self.scl_selector):
            return input_dict, target_dict

        scl = read_selector(input_dict, target_dict, self.scl_selector)
        mask = self._to_mask(scl)
        write_selector(input_dict, target_dict, self.output_selector, mask)
        return input_dict, target_dict
