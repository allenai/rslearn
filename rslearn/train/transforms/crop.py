"""Crop transform."""

from typing import Any

import torch
import torchvision

from rslearn.train.model_context import RasterImage

from .transform import Transform, read_selector, selector_exists


class Crop(Transform):
    """Crop inputs down to a smaller size.

    Supports two modes:
    - Random crop (default): specify crop_size to randomly crop to that size.
    - Deterministic crop: specify crop_size and offset=(left, top) to crop at a fixed
      position with no randomness.
    """

    def __init__(
        self,
        crop_size: int | tuple[int, int] = 0,
        image_selectors: list[str] = ["image"],
        box_selectors: list[str] = [],
        skip_missing: bool = False,
        offset: tuple[int, int] | None = None,
    ):
        """Initialize a new Crop.

        Performs random cropping (if offset=None) or deterministic cropping (if offset
        is set and crop_size is one integer). If input images have different shapes,
        crop_size and offset are applied based on the smallest image.

        Args:
            crop_size: the size to crop to, or a min/max range of crop sizes.
            image_selectors: image items to transform.
            box_selectors: boxes items to transform.
            skip_missing: if True, skip selectors that don't exist in the input/target
                dicts. Useful when working with optional inputs.
            offset: optional (left, top) pixel offset for deterministic cropping. When
                set, the crop is placed at this fixed position instead of randomly.
        """
        super().__init__()
        self.offset = offset

        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size + 1)
        else:
            self.crop_size = crop_size

        self.image_selectors = image_selectors
        self.box_selectors = box_selectors
        self.skip_missing = skip_missing

    def sample_state(self, image_shape: tuple[int, int]) -> dict[str, Any]:
        """Decide how to crop the input.

        When offset is set, returns a deterministic state using crop_size at the given
        offset. Otherwise randomly samples a square crop.

        Args:
            image_shape: the (height, width) of the images to transform. In case images
                are at different resolutions, it should correspond to the lowest
                resolution image.

        Returns:
            dict of sampled choices
        """
        crop_size = torch.randint(
            low=self.crop_size[0],
            high=self.crop_size[1],
            size=(),
        )
        assert image_shape[0] >= crop_size and image_shape[1] >= crop_size

        if self.offset is not None:
            remove_from_left = self.offset[0]
            remove_from_top = self.offset[1]
            if remove_from_left + crop_size > image_shape[1]:
                raise ValueError(
                    f"offset[0]={remove_from_left} + crop_size={crop_size} "
                    f"exceeds image width={image_shape[1]}"
                )
            if remove_from_top + crop_size > image_shape[0]:
                raise ValueError(
                    f"offset[1]={remove_from_top} + crop_size={crop_size} "
                    f"exceeds image height={image_shape[0]}"
                )
        else:
            remove_from_left = torch.randint(
                low=0,
                high=image_shape[1] - crop_size,
                size=(),
            )
            remove_from_top = torch.randint(
                low=0,
                high=image_shape[0] - crop_size,
                size=(),
            )

        return {
            "image_shape": image_shape,
            "crop_size": crop_size,
            "remove_from_left": remove_from_left,
            "remove_from_top": remove_from_top,
        }

    def apply_image(self, image: RasterImage, state: dict[str, Any]) -> RasterImage:
        """Apply the sampled state on the specified image.

        Args:
            image: the image to transform.
            state: the sampled state.
        """
        # Convert from coordinates based on the smallest image shape to coordinates in
        # this particular imaeg.
        image_shape = state["image_shape"]
        crop_size = state["crop_size"] * image.shape[-1] // image_shape[1]
        remove_from_left = state["remove_from_left"] * image.shape[-1] // image_shape[1]
        remove_from_top = state["remove_from_top"] * image.shape[-2] // image_shape[0]
        # Apply the cropping operation.
        image.image = torchvision.transforms.functional.crop(
            image.image,
            top=remove_from_top,
            left=remove_from_left,
            height=crop_size,
            width=crop_size,
        )
        return image

    def apply_boxes(self, boxes: torch.Tensor, state: dict[str, Any]) -> torch.Tensor:
        """Apply the crop on the specified boxes.

        Offsets box coordinates by the crop origin and clips them to the crop region.
        Boxes are expected to be (N, 5) tensors with columns (x1, y1, x2, y2, class).

        Args:
            boxes: the boxes to transform.
            state: the sampled state.

        Returns:
            the transformed boxes.
        """
        remove_from_left = state["remove_from_left"]
        remove_from_top = state["remove_from_top"]
        crop_size = state["crop_size"]

        boxes = boxes.clone()
        boxes[:, 0] -= remove_from_left
        boxes[:, 2] -= remove_from_left
        boxes[:, 1] -= remove_from_top
        boxes[:, 3] -= remove_from_top

        boxes[:, 0].clamp_(min=0, max=crop_size)
        boxes[:, 2].clamp_(min=0, max=crop_size)
        boxes[:, 1].clamp_(min=0, max=crop_size)
        boxes[:, 3].clamp_(min=0, max=crop_size)

        # Remove boxes that have zero area after clipping.
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        return boxes[valid]

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply transform over the inputs and targets.

        Args:
            input_dict: the input
            target_dict: the target

        Returns:
            transformed (input_dicts, target_dicts) tuple
        """
        smallest_image_shape = None
        for selector in self.image_selectors:
            if self.skip_missing and not selector_exists(
                input_dict, target_dict, selector
            ):
                continue
            image = read_selector(input_dict, target_dict, selector)
            if (
                smallest_image_shape is None
                or image.shape[-1] < smallest_image_shape[1]
            ):
                smallest_image_shape = image.shape[-2:]

        if smallest_image_shape is None:
            if self.skip_missing:
                # All selectors were missing, nothing to crop
                return input_dict, target_dict
            raise ValueError("No image found to crop")
        state = self.sample_state(smallest_image_shape)

        self.apply_fn(
            self.apply_image, input_dict, target_dict, self.image_selectors, state=state
        )
        self.apply_fn(
            self.apply_boxes, input_dict, target_dict, self.box_selectors, state=state
        )
        return input_dict, target_dict
