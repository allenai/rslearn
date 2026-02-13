"""Deterministic crop-to-bounds transform."""

from typing import Any

from rslearn.train.model_context import RasterImage

from .transform import Transform


class CropTo(Transform):
    """Crop inputs to specific pixel bounds.

    Unlike Crop (random) or Pad (symmetric center/topleft), this crops to exact
    user-configured bounds with no randomness.
    """

    def __init__(
        self,
        bounds: tuple[int, int, int, int],
        image_selectors: list[str] = ["image"],
    ):
        """Initialize a new CropTo.

        Args:
            bounds: (col1, row1, col2, row2) pixel bounds to crop to.
            image_selectors: image items to transform. Supports both input and
                target selectors (e.g. ["sentinel2", "target/targets"]).
        """
        super().__init__()
        self.bounds = bounds
        self.image_selectors = image_selectors

    def apply_image(self, image: RasterImage) -> RasterImage:
        """Crop the image to the configured bounds.

        Args:
            image: the image to crop.

        Returns:
            the cropped image.
        """
        col1, row1, col2, row2 = self.bounds
        image.image = image.image[..., row1:row2, col1:col2]
        return image

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply the crop over inputs and targets.

        Args:
            input_dict: the input
            target_dict: the target

        Returns:
            transformed (input_dict, target_dict) tuple
        """
        self.apply_fn(self.apply_image, input_dict, target_dict, self.image_selectors)
        return input_dict, target_dict
