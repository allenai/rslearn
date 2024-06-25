"""Normalization transforms."""

import torch


class Concatenate(torch.nn.Module):
    """Concatenate bands across multiple image inputs."""

    def __init__(
        self,
        selections: dict[str, list[int]],
        output_key: str,
        apply_on_inputs: bool = True,
        apply_on_targets: bool = False,
    ):
        """Initialize a new Concatenate.

        Args:
            selections: map from input key to list of band indices in that input to
                retain, or empty list to use all bands.
            output_key: the output key under which to save the concatenate image.
            apply_on_inputs: whether to apply the concatenation on input dict.
            apply_on_targets: whether to apply the concatenation on target dict.
        """
        super().__init__()
        self.selections = selections
        self.output_key = output_key
        self.apply_on_inputs = apply_on_inputs
        self.apply_on_targets = apply_on_targets

    def forward(self, input_dict, target_dict):
        """Apply concatenation over the inputs and targets.

        Args:
            input_dict: the input
            target_dict: the target

        Returns:
            normalized (input_dicts, target_dicts) tuple
        """
        dicts = []
        if self.apply_on_inputs:
            dicts.append(input_dict)
        if self.apply_on_targets:
            dicts.append(target_dict)

        for d in dicts:
            images = []
            for k, wanted_bands in self.selections.items():
                image = d[k]
                if wanted_bands:
                    image = image[wanted_bands, :, :]
                images.append(image)
            d[self.output_key] = torch.concatenate(images, dim=0)

        return input_dict, target_dict
