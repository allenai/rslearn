"""The SelectBands transform."""

from typing import Any

from .transform import Transform, read_selector, write_selector


class SelectBands(Transform):
    """Select a subset of bands from an image."""

    def __init__(
        self,
        band_indices: list[int],
        input_selector: str = "image",
        output_selector: str = "image",
    ):
        """Initialize a new SelectBands.

        Args:
            band_indices: the bands to select from the channel dimension.
            input_selector: the selector to read the input image.
            output_selector: the output selector under which to save the output image.
        """
        super().__init__()

        self.input_selector = input_selector
        self.output_selector = output_selector
        self.band_indices = band_indices

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply band selection over the inputs and targets.

        Args:
            input_dict: the input
            target_dict: the target

        Returns:
            (input_dicts, target_dicts) tuple with selected bands
        """
        image = read_selector(input_dict, target_dict, self.input_selector)
        image.image = image.image[self.band_indices]
        write_selector(input_dict, target_dict, self.output_selector, image)
        return input_dict, target_dict
