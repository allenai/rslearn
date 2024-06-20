"""Pad transform."""

from typing import Any, Union

import torch
import torchvision


class Pad(torch.nn.Module):
    """Pad (or crop) inputs to a fixed size."""

    def __init__(
        self,
        size: Union[int, tuple[int, int]],
        mode: str = "topleft",
        input_images: list[str] = ["image"],
        target_images: list[str] = [],
        input_boxes: list[str] = [],
        target_boxes: list[str] = [],
    ):
        """Initialize a new Crop.

        Result will be (input - mean) / std.

        Args:
            size: the size to pad to, or a min/max range of pad sizes. If the image is
                larger than this size, then it is cropped instead.
            mode: "center" (default) to apply padding equally on all sides, or
                "topleft" to only apply it on the bottom and right.
            input_images: image inputs to operate on (default "image")
            target_images: image targets to operate on (default none)
            input_boxes: box inputs to operate on (default none)
            target_boxes: box targets to operate on (default none)
        """
        super().__init__()
        if isinstance(size, int):
            self.size = (size, size + 1)
        else:
            self.size = size

        self.mode = mode
        self.input_images = input_images
        self.target_images = target_images
        self.input_boxes = input_boxes
        self.target_boxes = target_boxes
        self.generator = torch.Generator()

    def sample_state(self) -> dict[str, Any]:
        """Randomly decide how to transform the input.

        Returns:
            dict of sampled choices
        """
        return {
            "size": torch.randint(
                low=self.size[0], high=self.size[1], generator=self.generator, size=()
            )
        }

    def apply_state(
        self,
        state: dict[str, bool],
        d: dict[str, Any],
        image_keys: list[str],
        box_keys: list[str],
    ):
        """Apply the transform on one dict.

        Args:
            state: the state from sample_state
            d: the dict to apply the transform on.
            image_keys: image keys in the dict to transform.
            box_keys: bounding box keys in the dict to transform.
        """
        size = state["size"]
        for k in image_keys:
            horizontal_extra = size - d[k].shape[-1]
            vertical_extra = size - d[k].shape[-2]

            def apply_padding(
                im: torch.Tensor, horizontal: bool, before: int, after: int
            ) -> torch.Tensor:
                # Before/after must either be both non-negative or both negative.
                # >=0 indicates padding while <0 indicates cropping.
                assert (before < 0 and after < 0) or (before >= 0 and after >= 0)
                if before > 0:
                    # Padding.
                    if horizontal:
                        padding_tuple = (before, after)
                    else:
                        padding_tuple = (before, after, 0, 0)
                    return torch.nn.functional.pad(im, padding_tuple)
                else:
                    # Cropping.
                    if horizontal:
                        return torchvision.transforms.functional.crop(
                            im,
                            top=0,
                            left=-before,
                            height=im.shape[-2],
                            width=im.shape[-1] + before + after,
                        )
                    else:
                        return torchvision.transforms.functional.crop(
                            im,
                            top=-before,
                            left=0,
                            height=im.shape[-2] + before + after,
                            width=im.shape[-1],
                        )

            if self.mode == "topleft":
                horizontal_pad = (0, horizontal_extra)
                vertical_pad = (0, vertical_extra)

            elif self.mode == "center":
                horizontal_half = horizontal_extra // 2
                vertical_half = vertical_extra // 2
                horizontal_pad = (horizontal_half, horizontal_extra - horizontal_half)
                vertical_pad = (vertical_half, vertical_extra - vertical_half)

            d[k] = apply_padding(d[k], True, horizontal_pad[0], horizontal_pad[1])
            d[k] = apply_padding(d[k], False, vertical_pad[0], vertical_pad[1])

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
