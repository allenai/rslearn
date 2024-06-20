"""Flip transform."""

from typing import Any

import torch


class Flip(torch.nn.Module):
    """Flip inputs horizontally and/or vertically."""

    def __init__(
        self,
        horizontal: bool = True,
        vertical: bool = True,
        input_images: list[str] = ["image"],
        target_images: list[str] = [],
        input_boxes: list[str] = [],
        target_boxes: list[str] = [],
    ):
        """Initialize a new Flip.

        Args:
            horizontal: whether to randomly flip horizontally
            vertical: whether to randomly flip vertically
            input_images: image inputs to operate on (default "image")
            target_images: image targets to operate on (default none)
            input_boxes: box inputs to operate on (default none)
            target_boxes: box targets to operate on (default none)
        """
        super().__init__()
        self.horizontal = horizontal
        self.vertical = vertical
        self.input_images = input_images
        self.target_images = target_images
        self.input_boxes = input_boxes
        self.target_boxes = target_boxes
        self.generator = torch.Generator()

    def sample_state(self) -> dict[str, bool]:
        """Randomly decide how to transform the input.

        Returns:
            dict of sampled choices
        """
        horizontal = False
        if self.horizontal:
            horizontal = (
                torch.randint(low=0, high=2, generator=self.generator, size=()) == 0
            )
        vertical = False
        if self.vertical:
            vertical = (
                torch.randint(low=0, high=2, generator=self.generator, size=()) == 0
            )
        return {
            "horizontal": horizontal,
            "vertical": vertical,
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
        for k in image_keys:
            if state["horizontal"]:
                d[k] = torch.flip(d[k], dims=[-1])
            if state["vertical"]:
                d[k] = torch.flip(d[k], dims=[-2])

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
