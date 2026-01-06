"""The SelectBands transform."""

from typing import Any

from rslearn.train.model_context import RasterImage

from .transform import Transform, read_selector, write_selector


class SelectBands(Transform):
    """Select a subset of bands from an image."""

    def __init__(
        self,
        band_indices: list[int],
        input_selector: str = "image",
        output_selector: str = "image",
        num_bands_per_timestep: int | None = None,
    ):
        """Initialize a new Concatenate.

        Args:
            band_indices: the bands to select.
            input_selector: the selector to read the input image.
            output_selector: the output selector under which to save the output image.
            num_bands_per_timestep: the number of bands per image, to distinguish
                between stacked images in an image time series. If set, then the
                band_indices are selected for each image in the time series.
        """
        super().__init__()
        self.input_selector = input_selector
        self.output_selector = output_selector
        self.band_indices = band_indices
        self.num_bands_per_timestep = num_bands_per_timestep

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
        image = read_selector(input_dict, target_dict, self.input_selector)
        num_bands_per_timestep = (
            self.num_bands_per_timestep
            if self.num_bands_per_timestep is not None
            else image.shape[0]
        )
        if isinstance(image, RasterImage):
            assert num_bands_per_timestep == image.shape[0], (
                "Expect a seperate dimension for timesteps in RasterImages."
            )

        if image.shape[0] % num_bands_per_timestep != 0:
            raise ValueError(
                f"channel dimension {image.shape[0]} is not multiple of bands per timestep {num_bands_per_timestep}"
            )

        # Copy the band indices for each timestep in the input.
        wanted_bands: list[int] = []
        for start_channel_idx in range(0, image.shape[0], num_bands_per_timestep):
            wanted_bands.extend(
                [(start_channel_idx + band_idx) for band_idx in self.band_indices]
            )

        if isinstance(image, RasterImage):
            image.image = image.image[wanted_bands]
        else:
            image = image[wanted_bands]
        write_selector(input_dict, target_dict, self.output_selector, image)
        return input_dict, target_dict
