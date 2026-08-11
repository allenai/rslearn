"""Transforms deriving terrain products from an elevation band."""

from typing import Any

import torch

from rslearn.train.model_context import RasterImage

from .transform import Transform, read_selector, selector_exists, write_selector

ELEVATION_BAND = "elevation"
SLOPE_BAND = "slope"
ASPECT_BAND = "aspect"
SUPPORTED_BANDS = (ELEVATION_BAND, SLOPE_BAND, ASPECT_BAND)

# Slopes below this many degrees are treated as flat, where aspect is undefined.
FLAT_SLOPE_EPSILON = 1e-6


class ElevationToSlopeAspect(Transform):
    """Derive slope and aspect from a single-band elevation image.

    The input must be a single-channel elevation raster in meters. Because the
    transform runs after materialization, the image is already in the window's
    projection, so the pixel spacing is uniform and must be provided in meters via
    ``pixel_size_m`` (a ``RasterImage`` does not carry projection metadata). Pixels
    are assumed to be square, which holds for rslearn windows.

    Nodata is assumed to be represented as NaN in the input elevation. NaN pixels
    produce NaN slope and aspect, and since the gradients read neighboring pixels,
    each NaN also invalidates the slope and aspect of its four orthogonally adjacent
    pixels.

    Gradients use central differences in the interior and one-sided differences on
    the image border, so slope and aspect on the outermost pixel ring are less
    accurate than the interior.
    """

    def __init__(
        self,
        pixel_size_m: float,
        input_selector: str = "elevation",
        output_selector: str | None = None,
        bands: list[str] = list(SUPPORTED_BANDS),
        flat_aspect_value: float = -1.0,
        skip_missing: bool = False,
    ):
        """Initialize a new ElevationToSlopeAspect.

        Args:
            pixel_size_m: the size of a pixel in meters in the window's projection.
            input_selector: the selector containing the elevation image.
            output_selector: the selector to write the result to. Defaults to
                ``input_selector``.
            bands: the bands to emit, in order. Choose from "elevation", "slope",
                and "aspect".
            flat_aspect_value: the aspect value to use for flat pixels, where the
                compass direction of steepest descent is undefined.
            skip_missing: if True, skip the transform when the input selector does
                not exist in the input/target dicts.
        """
        super().__init__(skip_missing=skip_missing)
        if pixel_size_m <= 0:
            raise ValueError("pixel_size_m must be positive")
        if not bands:
            raise ValueError("expected at least one band")
        for band in bands:
            if band not in SUPPORTED_BANDS:
                raise ValueError(
                    f"unsupported band '{band}'; choose from {SUPPORTED_BANDS}"
                )

        self.pixel_size_m = pixel_size_m
        self.input_selector = input_selector
        self.output_selector = (
            output_selector if output_selector is not None else input_selector
        )
        self.bands = list(bands)
        self.flat_aspect_value = flat_aspect_value

    def compute(self, elevation: torch.Tensor) -> torch.Tensor:
        """Compute the configured terrain bands from an elevation tensor.

        Args:
            elevation: THW tensor of elevations in meters. Rows run north to south,
                following the standard raster convention.

        Returns:
            a CTHW tensor with one channel per configured band.
        """
        dem = elevation.to(torch.float32)
        band_arrays: dict[str, torch.Tensor] = {ELEVATION_BAND: dem}

        if SLOPE_BAND in self.bands or ASPECT_BAND in self.bands:
            # torch.gradient returns derivatives per unit spacing, so passing the
            # metric pixel size directly yields elevation change per meter.
            dz_drow = torch.gradient(dem, spacing=self.pixel_size_m, dim=-2)[0]
            dz_dcol = torch.gradient(dem, spacing=self.pixel_size_m, dim=-1)[0]

            # Rows increase southward, so negate to get the northward derivative.
            dz_north = -dz_drow
            dz_east = dz_dcol

            slope = torch.rad2deg(torch.atan(torch.hypot(dz_east, dz_north)))
            band_arrays[SLOPE_BAND] = slope

            if ASPECT_BAND in self.bands:
                # atan2(east, north) is the clockwise bearing of steepest ascent;
                # add 180 degrees for the direction of steepest descent.
                aspect = (torch.rad2deg(torch.atan2(dz_east, dz_north)) + 180.0) % 360.0
                aspect = torch.where(
                    slope < FLAT_SLOPE_EPSILON,
                    torch.full_like(aspect, self.flat_aspect_value),
                    aspect,
                )
                band_arrays[ASPECT_BAND] = aspect

        # Central differences do not read the center pixel, so a pixel with unknown
        # elevation would otherwise get a slope derived purely from its neighbors.
        nan_mask = torch.isnan(dem)
        for band in (SLOPE_BAND, ASPECT_BAND):
            if band in band_arrays:
                band_arrays[band] = torch.where(
                    nan_mask,
                    torch.full_like(band_arrays[band], float("nan")),
                    band_arrays[band],
                )

        return torch.stack([band_arrays[band] for band in self.bands], dim=0)

    def apply_image(self, image: RasterImage) -> RasterImage:
        """Derive terrain bands from the given elevation image.

        Args:
            image: a single-channel CTHW elevation image.

        Returns:
            a CTHW image with one channel per configured band.
        """
        if image.image.shape[0] != 1:
            raise ValueError(
                "expected a single-channel elevation image, got "
                f"{image.image.shape[0]} channels"
            )
        return RasterImage(
            self.compute(image.image[0]),
            timestamps=image.timestamps,
        )

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply the transform over the inputs and targets.

        Args:
            input_dict: the input
            target_dict: the target

        Returns:
            (input_dict, target_dict) where output_selector contains the terrain
            bands.
        """
        if self.skip_missing and not selector_exists(
            input_dict, target_dict, self.input_selector
        ):
            return input_dict, target_dict

        image = read_selector(input_dict, target_dict, self.input_selector)
        write_selector(
            input_dict, target_dict, self.output_selector, self.apply_image(image)
        )
        return input_dict, target_dict
