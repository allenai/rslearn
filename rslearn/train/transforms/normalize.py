"""Normalization transforms."""

from typing import Union

import torch


class Normalize(torch.nn.Module):
    """Normalize one or more input images with mean and standard deviation."""

    def __init__(
        self,
        mean: Union[float, list[float]],
        std: Union[float, list[float]],
        input_keys: list[str] = ["image"],
        target_keys: list[str] = [],
    ):
        """Initialize a new Normalize.

        Result will be (input - mean) / std.

        Args:
            mean: a single value or one mean per channel
            std: a single value or one std per channel
            input_keys: which inputs to operate on (default "image")
            target_keys: which targets to operate on (default none)
        """
        super().__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.input_keys = input_keys
        self.target_keys = target_keys

    def forward(self, input_dict, target_dict):
        """Apply normalization over the inputs and targets.

        Args:
            input_dict: the input
            target_dict: the target

        Returns:
            normalized (input_dicts, target_dicts) tuple
        """
        for key_list, d in [
            (self.input_keys, input_dict),
            (self.target_keys, target_dict),
        ]:
            for k in key_list:
                d[k] = (d[k] - self.mean) / self.std

        return input_dict, target_dict
