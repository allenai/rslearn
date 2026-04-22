"""Adaptive pooling transform."""

from typing import Any, Literal

import torch

from rslearn.train.model_context import RasterImage

from .transform import Transform

POOLING_MODES = {
    "max": torch.nn.functional.adaptive_max_pool2d,
    "mean": torch.nn.functional.adaptive_avg_pool2d,
}


class AdaptivePooling(Transform):
    """Pools inputs to a target size using adaptive spatial pooling.

    Supports max pooling and mean pooling over each spatial region. Mean pooling
    returns floating-point outputs for non-floating inputs so the averages are
    preserved.
    """

    def __init__(
        self,
        target_size: tuple[int, int],
        selectors: list[str] | None = None,
        pooling: Literal["max", "mean"] = "max",
        skip_missing: bool = False,
    ):
        """Initialize an adaptive pooling transform.

        Args:
            target_size: the (height, width) to pool to.
            selectors: items to transform.
            pooling: the adaptive pooling mode to use. Must be "max" or "mean".
            skip_missing: if True, skip selectors that don't exist in the input/target
                dicts.
        """
        super().__init__(skip_missing=skip_missing)
        if pooling not in POOLING_MODES:
            raise ValueError(f"Unsupported pooling mode {pooling!r}")

        self.target_size = target_size
        self.selectors = [] if selectors is None else selectors
        self.pooling = pooling

    def apply_pooling(self, image: RasterImage) -> RasterImage:
        """Apply adaptive spatial pooling on the specified image.

        Args:
            image: the image to transform (CTHW tensor).

        Returns:
            the pooled image.
        """
        # image.image is [C, T, H, W]. Merge C and T so we can pool over
        # (H, W), then restore the original leading dims.
        c, t, h, w = image.image.shape
        merged = image.image.reshape(c * t, 1, h, w)
        pooled = POOLING_MODES[self.pooling](merged.float(), self.target_size)
        if self.pooling == "max" or image.image.is_floating_point():
            pooled = pooled.to(image.image.dtype)
        image.image = pooled.reshape(c, t, *self.target_size)
        return image

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply transform over the inputs and targets.

        Args:
            input_dict: the input
            target_dict: the target

        Returns:
            transformed (input_dicts, target_dicts) tuple
        """
        self.apply_fn(self.apply_pooling, input_dict, target_dict, self.selectors)
        return input_dict, target_dict
