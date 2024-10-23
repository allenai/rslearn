"""Concatenate bands across multiple image inputs."""

from typing import Any

import torch

from .transform import Transform


class Concatenate(Transform):
    """Concatenate bands across multiple image inputs."""

    def __init__(
        self,
        selections: dict[str, list[int]],
        output_selector: str,
    ):
        """Initialize a new Concatenate.

        Args:
            selections: map from selector to list of band indices in that input to
                retain, or empty list to use all bands.
            output_selector: the output selector under which to save the concatenate image.
        """
        super().__init__()
        self.selections = selections
        self.output_selector = output_selector

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply concatenation over the inputs and targets.

        Args:
            input_dict: the input
            target_dict: the target

        Returns:
            normalized (input_dicts, target_dicts) tuple
        """
        images = []
        for selector, wanted_bands in self.selections.items():
            image = self.read_selector(input_dict, target_dict, selector)
            if wanted_bands:
                image = image[wanted_bands, :, :]
            images.append(image)
        result = torch.concatenate(images, dim=0)
        self.write_selector(input_dict, target_dict, self.output_selector, result)
        return input_dict, target_dict
