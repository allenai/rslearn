"""Mask transform."""

import torch

from rslearn.train.model_context import RasterImage
from rslearn.train.transforms.transform import Transform, read_selector


class Mask(Transform):
    """Apply a mask to one or more images.

    This uses one (mask) image input to mask another (target) image input. The value of
    the target image is set to the mask value everywhere where the mask image is 0.
    """

    def __init__(
        self,
        selectors: list[str] = ["image"],
        mask_selector: str = "mask",
        mask_value: int = 0,
    ):
        """Initialize a new Mask.

        Args:
            selectors: images to mask.
            mask_selector: the selector for the mask image to apply.
            mask_value: set each image in selectors to this value where the image
                corresponding to the mask_selector is 0.
        """
        super().__init__()
        self.selectors = selectors
        self.mask_selector = mask_selector
        self.mask_value = mask_value

    def apply_image(
        self, image: torch.Tensor | RasterImage, mask: torch.Tensor | RasterImage
    ) -> torch.Tensor | RasterImage:
        """Apply the mask on the image.

        Args:
            image: the image
            mask: the mask

        Returns:
            masked image
        """
        # Tile the mask to have same number of bands as the image.
        if isinstance(mask, RasterImage):
            mask = mask.image

        if image.shape[0] != mask.shape[0]:
            if mask.shape[0] != 1:
                raise ValueError(
                    "expected mask to either have same bands as image, or one band"
                )
            mask = mask.repeat(image.shape[0], 1, 1)

        if isinstance(image, torch.Tensor):
            image[mask == 0] = self.mask_value
        else:
            image.image[mask == 0] = self.mask_value
        return image

    def forward(self, input_dict: dict, target_dict: dict) -> tuple[dict, dict]:
        """Apply mask.

        Args:
            input_dict: the input
            target_dict: the target

        Returns:
            normalized (input_dicts, target_dicts) tuple
        """
        mask = read_selector(input_dict, target_dict, self.mask_selector)
        self.apply_fn(
            self.apply_image, input_dict, target_dict, self.selectors, mask=mask
        )
        return input_dict, target_dict
