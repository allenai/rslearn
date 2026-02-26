"""Time-series augmentation transforms: masking, shift, noise, and mixup."""

import random
from typing import Any

import torch

from rslearn.train.model_context import RasterImage

from .transform import Transform


class RandomTimeMasking(Transform):
    """Randomly mask entire timesteps in a CTHW tensor.

    After normalization, 0 equals the mean, so masking to 0 makes missing data
    indistinguishable from real mean-valued data.  Set ``mask_value`` to a
    sentinel outside the normal range (e.g. -9999) and/or enable
    ``append_mask_channel`` to add a binary (1=valid, 0=masked) channel so
    the downstream encoder can tell masked from real timesteps.
    """

    def __init__(
        self,
        selectors: list[str] = ["image"],
        mask_ratio: float = 0.2,
        mask_value: float = 0.0,
        append_mask_channel: bool = False,
        skip_missing: bool = False,
    ):
        """Initialize RandomTimeMasking.

        Args:
            selectors: inputs to augment.
            mask_ratio: expected fraction of timesteps to mask.
            mask_value: fill value for masked timesteps.
            append_mask_channel: if True, prepend a binary channel (1=valid,
                0=masked) so the encoder can learn to ignore masked steps.
            skip_missing: skip missing selectors.
        """
        super().__init__()
        self.selectors = selectors
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        self.append_mask_channel = append_mask_channel
        self.skip_missing = skip_missing

    def apply_image(self, image: RasterImage) -> RasterImage:
        """Mask random timesteps."""
        C, T, H, W = image.shape
        if T <= 1:
            if self.append_mask_channel:
                valid = torch.ones(
                    1, T, H, W, dtype=image.image.dtype, device=image.image.device
                )
                image.image = torch.cat([valid, image.image], dim=0)
            return image
        mask = torch.rand(T) < self.mask_ratio
        # Ensure at least one timestep survives.
        if mask.all():
            mask[torch.randint(T, (1,))] = False
        image.image[:, mask] = self.mask_value
        if self.append_mask_channel:
            valid = (~mask).float().view(1, T, 1, 1).expand(1, T, H, W)
            valid = valid.to(dtype=image.image.dtype, device=image.image.device)
            image.image = torch.cat([valid, image.image], dim=0)
        return image

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply transform."""
        self.apply_fn(self.apply_image, input_dict, target_dict, self.selectors)
        return input_dict, target_dict


class TemporalShift(Transform):
    """Non-circular time shift with padding (edge or zero)."""

    def __init__(
        self,
        selectors: list[str] = ["image"],
        max_shift: int = 2,
        pad_mode: str = "edge",  # "edge" or "zero"
        skip_missing: bool = False,
    ):
        """Initialize TemporalShift.

        Args:
            selectors: inputs to augment.
            max_shift: maximum shift in either direction (inclusive).
            pad_mode: ``"edge"`` repeats the boundary timestep,
                ``"zero"`` fills with zeros.
            skip_missing: skip missing selectors.
        """
        super().__init__()
        assert pad_mode in ("edge", "zero")
        self.selectors = selectors
        self.max_shift = max_shift
        self.pad_mode = pad_mode
        self.skip_missing = skip_missing

    def apply_image(self, image: RasterImage, shift: int) -> RasterImage:
        """Shift along T with edge/zero padding (no circular wrap)."""
        x = image.image
        C, T, H, W = x.shape
        if shift == 0 or T <= 1:
            return image

        k = min(abs(shift), T - 1)  # never wipe all timesteps
        if k == 0:
            return image

        if self.pad_mode == "edge":
            pad_val = x[:, :1] if shift > 0 else x[:, -1:]
            pad = pad_val.expand(-1, k, -1, -1)
        else:  # "zero"
            pad = torch.zeros((C, k, H, W), device=x.device, dtype=x.dtype)

        if shift > 0:
            # shift right: drop oldest k, pad at front
            image.image = torch.cat([pad, x[:, :-k]], dim=1)
            if image.timestamps is not None:
                image.timestamps = [image.timestamps[0]] * k + image.timestamps[:-k]
        else:
            # shift left: drop newest k, pad at end
            image.image = torch.cat([x[:, k:], pad], dim=1)
            if image.timestamps is not None:
                image.timestamps = image.timestamps[k:] + [image.timestamps[-1]] * k

        return image

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply transform."""
        shift = random.randint(-self.max_shift, self.max_shift)
        self.apply_fn(
            self.apply_image, input_dict, target_dict, self.selectors, shift=shift
        )
        return input_dict, target_dict


class GaussianNoise(Transform):
    """Add Gaussian noise to inputs."""

    def __init__(
        self,
        selectors: list[str] = ["image"],
        std: float = 0.01,
        skip_missing: bool = False,
    ):
        """Initialize GaussianNoise.

        Args:
            selectors: inputs to augment.
            std: standard deviation of the noise.
            skip_missing: skip missing selectors.
        """
        super().__init__()
        self.selectors = selectors
        self.std = std
        self.skip_missing = skip_missing

    def apply_image(self, image: RasterImage) -> RasterImage:
        """Add noise."""
        image.image = image.image + torch.randn_like(image.image) * self.std
        return image

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply transform."""
        self.apply_fn(self.apply_image, input_dict, target_dict, self.selectors)
        return input_dict, target_dict
