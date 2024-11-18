"""Crop transform."""

from typing import Any

import torch
import torchvision

from .transform import Transform


class Crop(Transform):
    """Crop inputs down to a smaller size."""

    def __init__(
        self,
        crop_size: int | tuple[int, int],
        image_selectors: list[str] = ["image"],
        box_selectors: list[str] = [],
    ):
        """Initialize a new Crop.

        Result will be (input - mean) / std.

        Args:
            crop_size: the size to crop to, or a min/max range of crop sizes
            image_selectors: image items to transform.
            box_selectors: boxes items to transform.
        """
        super().__init__()
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size + 1)
        else:
            self.crop_size = crop_size

        self.image_selectors = image_selectors
        self.box_selectors = box_selectors

    def sample_state(self, image_shape: tuple[int, int]) -> dict[str, Any]:
        """Randomly decide how to transform the input.

        Args:
            image_shape: the (height, width) of the images to transform. In case images
                are at different resolutions, it should correspond to the lowest
                resolution image.

        Returns:
            dict of sampled choices
        """
        crop_size = torch.randint(
            low=self.crop_size[0],
            high=self.crop_size[1],
            size=(),
        )
        assert image_shape[0] >= crop_size and image_shape[1] >= crop_size
        remove_from_left = torch.randint(
            low=0,
            high=image_shape[1] - crop_size,
            size=(),
        )
        remove_from_top = torch.randint(
            low=0,
            high=image_shape[0] - crop_size,
            size=(),
        )
        return {
            "image_shape": image_shape,
            "crop_size": crop_size,
            "remove_from_left": remove_from_left,
            "remove_from_top": remove_from_top,
        }

    def apply_image(self, image: torch.Tensor, state: dict[str, Any]) -> torch.Tensor:
        """Apply the sampled state on the specified image.

        Args:
            image: the image to transform.
            state: the sampled state.
        """
        image_shape = state["image_shape"]
        crop_size = state["crop_size"] * image.shape[-1] // image_shape[1]
        remove_from_left = state["remove_from_left"] * image.shape[-1] // image_shape[1]
        remove_from_top = state["remove_from_top"] * image.shape[-2] // image_shape[0]
        return torchvision.transforms.functional.crop(
            image,
            top=remove_from_top,
            left=remove_from_left,
            height=crop_size,
            width=crop_size,
        )

    def apply_boxes(self, boxes: Any, state: dict[str, bool]) -> torch.Tensor:
        """Apply the sampled state on the specified image.

        Args:
            boxes: the boxes to transform.
            state: the sampled state.
        """
        raise NotImplementedError

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
        smallest_image_shape = None
        for selector in self.image_selectors:
            image = self.read_selector(input_dict, target_dict, selector)
            if (
                smallest_image_shape is None
                or image.shape[-1] < smallest_image_shape[1]
            ):
                smallest_image_shape = image.shape[-2:]

        if smallest_image_shape is None:
            raise ValueError("No image found to crop")
        state = self.sample_state(smallest_image_shape)

        self.apply_fn(
            self.apply_image, input_dict, target_dict, self.image_selectors, state=state
        )
        self.apply_fn(
            self.apply_boxes, input_dict, target_dict, self.box_selectors, state=state
        )
        return input_dict, target_dict
