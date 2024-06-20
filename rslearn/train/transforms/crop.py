"""Crop transform."""

from typing import Any, Union

import torch
import torchvision


class Crop(torch.nn.Module):
    """Crop inputs down to a smaller size."""

    def __init__(
        self,
        crop_size: Union[int, tuple[int, int]],
        input_images: list[str] = ["image"],
        target_images: list[str] = [],
        input_boxes: list[str] = [],
        target_boxes: list[str] = [],
    ):
        """Initialize a new Crop.

        Result will be (input - mean) / std.

        Args:
            crop_size: the size to crop to, or a min/max range of crop sizes
            input_images: image inputs to operate on (default "image")
            target_images: image targets to operate on (default none)
            input_boxes: box inputs to operate on (default none)
            target_boxes: box targets to operate on (default none)
        """
        super().__init__()
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size + 1)
        else:
            self.crop_size = crop_size

        self.input_images = input_images
        self.target_images = target_images
        self.input_boxes = input_boxes
        self.target_boxes = target_boxes
        self.generator = torch.Generator()

    def sample_state(self) -> dict[str, Any]:
        """Randomly decide how to transform the input.

        Returns:
            dict of sampled choices
        """
        return {
            "crop_size": torch.randint(
                low=self.crop_size[0],
                high=self.crop_size[1],
                generator=self.generator,
                size=(),
            )
        }

    def apply_state(
        self,
        state: dict[str, bool],
        d: dict[str, Any],
        image_keys: list[str],
        box_keys: list[str],
    ) -> None:
        """Apply the sampled state on the specified dict.

        Args:
            state: the sampled state.
            d: the dict to transform.
            image_keys: image keys in the dict to transform.
            box_keys: box keys in the dict to transform.
        """
        crop_size = state["crop_size"]
        for k in image_keys:
            assert d[k].shape[-1] >= crop_size and d[k].shape[-2] >= crop_size
            remove_from_left = torch.randint(
                low=0,
                high=d[k].shape[-1] - crop_size,
                generator=self.generator,
                size=(),
            )
            remove_from_top = torch.randint(
                low=0,
                high=d[k].shape[-2] - crop_size,
                generator=self.generator,
                size=(),
            )
            d[k] = torchvision.transforms.functional.crop(
                d[k],
                top=remove_from_top,
                left=remove_from_left,
                height=crop_size,
                width=crop_size,
            )

    def forward(self, input_dict, target_dict):
        """Apply transform over the inputs and targets.

        Args:
            input_dict: the input
            target_dict: the target

        Returns:
            transformed (input_dicts, target_dicts) tuple
        """
        state = self.sample_state()
        self.apply_state(state, input_dict, self.input_images, self.input_boxes)
        self.apply_state(state, target_dict, self.target_images, self.target_boxes)
        return input_dict, target_dict
