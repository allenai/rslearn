"""Flip transform."""

from typing import Any

import torch

from .transform import Transform


class Flip(Transform):
    """Flip inputs horizontally and/or vertically."""

    def __init__(
        self,
        horizontal: bool = True,
        vertical: bool = True,
        image_selectors: list[str] = ["image"],
        box_selectors: list[str] = [],
    ):
        """Initialize a new Flip.

        Args:
            horizontal: whether to randomly flip horizontally
            vertical: whether to randomly flip vertically
            image_selectors: image items to transform.
            box_selectors: boxes items to transform.
        """
        super().__init__()
        self.horizontal = horizontal
        self.vertical = vertical
        self.image_selectors = image_selectors
        self.box_selectors = box_selectors
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

    def apply_image(self, image: torch.Tensor, state: dict[str, bool]) -> torch.Tensor:
        """Apply the sampled state on the specified image.

        Args:
            image: the image to transform.
            state: the sampled state.
        """
        if state["horizontal"]:
            image = torch.flip(image, dims=[-1])
        if state["vertical"]:
            image = torch.flip(image, dims=[-2])
        return image

    def apply_boxes(self, boxes: Any, state: dict[str, bool]) -> torch.Tensor:
        """Apply the sampled state on the specified image.

        Args:
            boxes: the boxes to transform.
            state: the sampled state.
        """
        raise NotImplementedError

    def forward(self, input_dict, target_dict):
        """Apply transform over the inputs and targets.

        Args:
            input_dict: the input
            target_dict: the target

        Returns:
            transformed (input_dicts, target_dicts) tuple
        """
        state = self.sample_state()
        self.apply_fn(
            self.apply_image, input_dict, target_dict, self.image_selectors, state=state
        )
        self.apply_fn(
            self.apply_boxes, input_dict, target_dict, self.box_selectors, state=state
        )
        return input_dict, target_dict
