"""Normalization transforms."""

from typing import Optional, Union

import torch


class Normalize(torch.nn.Module):
    """Normalize one or more input images with mean and standard deviation."""

    def __init__(
        self,
        mean: Union[float, list[float]],
        std: Union[float, list[float]],
        valid_range: Optional[
            Union[tuple[float, float], tuple[list[float], list[float]]]
        ] = None,
        input_keys: list[str] = ["image"],
        target_keys: list[str] = [],
    ):
        """Initialize a new Normalize.

        Result will be (input - mean) / std.

        Args:
            mean: a single value or one mean per channel
            std: a single value or one std per channel
            valid_range: optionally clip to a minimum and maximum value
            input_keys: which inputs to operate on (default "image")
            target_keys: which targets to operate on (default none)
        """
        super().__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

        if valid_range:
            self.valid_min = torch.tensor(valid_range[0])
            self.valid_max = torch.tensor(valid_range[1])
        else:
            self.valid_min = None
            self.valid_max = None

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

                if self.valid_min is not None:
                    d[k] = torch.clamp(d[k], min=self.valid_min, max=self.valid_max)

        return input_dict, target_dict
