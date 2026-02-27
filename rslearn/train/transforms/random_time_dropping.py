"""Random temporal dropping augmentation transform."""

from typing import Any

import torch

from rslearn.train.model_context import RasterImage

from .transform import Transform


class RandomTimeDropping(Transform):
    """Randomly drop entire timesteps from a CTHW tensor.

    Dropped timesteps are removed from both ``image.image`` (shrinking T)
    and ``image.timestamps``.  ``image.expected_timestamps`` is preserved
    so the model can infer which temporal positions are absent.
    """

    def __init__(
        self,
        selectors: list[str] = ["image"],
        drop_ratio: float = 0.2,
        min_keep: int = 1,
        skip_missing: bool = False,
    ):
        """Initialize RandomTimeDropping.

        Args:
            selectors: inputs to augment.
            drop_ratio: expected fraction of timesteps to drop.
            min_keep: minimum number of timesteps to keep (at least 1).
            skip_missing: skip missing selectors.
        """
        super().__init__()
        self.selectors = selectors
        self.drop_ratio = drop_ratio
        self.min_keep = max(min_keep, 1)
        self.skip_missing = skip_missing

    def apply_image(self, image: RasterImage) -> RasterImage:
        """Drop random timesteps from the image."""
        C, T, H, W = image.shape
        if T <= self.min_keep:
            return image

        # Decide which timesteps to keep.
        keep = torch.rand(T) >= self.drop_ratio

        # Guarantee we keep at least min_keep timesteps.
        if keep.sum() < self.min_keep:
            # Randomly pick min_keep indices to force-keep.
            indices = torch.randperm(T)[: self.min_keep]
            keep[:] = False
            keep[indices] = True

        # If nothing was dropped, return unchanged.
        if keep.all():
            return image

        image.image = image.image[:, keep, :, :]

        if image.timestamps is not None:
            image.timestamps = [
                ts for ts, k in zip(image.timestamps, keep.tolist()) if k
            ]

        return image

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply transform."""
        self.apply_fn(self.apply_image, input_dict, target_dict, self.selectors)
        return input_dict, target_dict
