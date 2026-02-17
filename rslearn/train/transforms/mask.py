"""Mask transform."""

from rslearn.train.model_context import RasterImage
from rslearn.train.transforms.transform import Transform, read_selector, selector_exists


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
        skip_missing: bool = False,
    ):
        """Initialize a new Mask.

        Args:
            selectors: images to mask.
            mask_selector: the selector for the mask image to apply.
            mask_value: set each image in selectors to this value where the image
                corresponding to the mask_selector is 0.
            skip_missing: if True, skip selectors that don't exist in the input/target
                dicts. Useful when working with optional inputs.
        """
        super().__init__()
        self.selectors = selectors
        self.mask_selector = mask_selector
        self.mask_value = mask_value
        self.skip_missing = skip_missing

    def apply_image(self, image: RasterImage, mask: RasterImage) -> RasterImage:
        """Apply the mask on the image.

        Args:
            image: the image
            mask: the mask

        Returns:
            masked image
        """
        # Extract the mask tensor (CTHW format)
        mask_tensor = mask.image

        # Tile the mask to have same number of bands (C dimension) as the image.
        if image.shape[0] != mask_tensor.shape[0]:
            if mask_tensor.shape[0] != 1:
                raise ValueError(
                    "expected mask to either have same bands as image, or one band"
                )
            # Repeat along C dimension, keep T, H, W the same
            mask_tensor = mask_tensor.repeat(image.shape[0], 1, 1, 1)

        # Tile the mask to have same number of timesteps (T dimension) as the image.
        if image.shape[1] != mask_tensor.shape[1]:
            if mask_tensor.shape[1] != 1:
                raise ValueError(
                    "expected mask to either have same timesteps as image, or one"
                    " timestep"
                )
            mask_tensor = mask_tensor.repeat(1, image.shape[1], 1, 1)

        image.image[mask_tensor == 0] = self.mask_value
        return image

    def forward(self, input_dict: dict, target_dict: dict) -> tuple[dict, dict]:
        """Apply mask.

        Args:
            input_dict: the input
            target_dict: the target

        Returns:
            normalized (input_dicts, target_dicts) tuple
        """
        # Check if mask exists when skip_missing is enabled
        if self.skip_missing and not selector_exists(
            input_dict, target_dict, self.mask_selector
        ):
            return input_dict, target_dict

        mask = read_selector(input_dict, target_dict, self.mask_selector)
        self.apply_fn(
            self.apply_image, input_dict, target_dict, self.selectors, mask=mask
        )
        return input_dict, target_dict
