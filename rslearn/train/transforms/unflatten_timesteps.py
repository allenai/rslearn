"""Reshape a flattened channel-time dimension back into separate C and T axes."""

from typing import Any

from rslearn.train.model_context import RasterImage

from .transform import Transform


class UnflattenTimesteps(Transform):
    """Unflatten a RasterImage whose time steps are stacked in the channel dim.

    Many data pipelines store multi-variable time-series as a single raster
    with ``C_total = num_variables * num_timesteps`` bands and ``T = 1``.
    This transform reshapes the image from ``[C*T_new, 1, H, W]`` to
    ``[C, T_new, H, W]`` so that downstream transforms (e.g. Normalize) and
    encoders see the proper channel count.
    """

    def __init__(
        self,
        num_channels: int,
        selectors: list[str] = ["image"],
        skip_missing: bool = False,
    ) -> None:
        """Initialize a new UnflattenTimesteps.

        Args:
            num_channels: the number of variables (channels) per timestep.
                The total band count must be divisible by this value; the
                inferred number of timesteps is ``total_bands // num_channels``.
            selectors: image items to transform.
            skip_missing: if True, skip selectors that don't exist in the
                input/target dicts.
        """
        super().__init__()
        self.num_channels = num_channels
        self.selectors = selectors
        self.skip_missing = skip_missing

    def apply_image(self, image: RasterImage) -> RasterImage:
        """Reshape the image from [C*T, T_orig, H, W] to [C, T*T_orig, H, W].

        Args:
            image: the image to reshape.

        Returns:
            a new RasterImage with unflattened channel and time dimensions.
        """
        c_total, t_orig, h, w = image.image.shape
        if c_total % self.num_channels != 0:
            raise ValueError(
                f"Total channels ({c_total}) is not divisible by "
                f"num_channels ({self.num_channels})."
            )
        t_new = (c_total // self.num_channels) * t_orig
        image.image = image.image.reshape(self.num_channels, t_new, h, w)
        return image

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply the reshape over the inputs and targets.

        Args:
            input_dict: the input dict.
            target_dict: the target dict.

        Returns:
            (input_dict, target_dict) with reshaped images.
        """
        self.apply_fn(self.apply_image, input_dict, target_dict, self.selectors)
        return input_dict, target_dict
