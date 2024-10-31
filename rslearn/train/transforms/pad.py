"""Pad transform."""

from typing import Any

import torch
import torchvision

from .transform import Transform


class Pad(Transform):
    """Pad (or crop) inputs to a fixed size."""

    def __init__(
        self,
        size: int | tuple[int, int],
        mode: str = "topleft",
        image_selectors: list[str] = ["image"],
        box_selectors: list[str] = [],
    ):
        """Initialize a new Crop.

        Result will be (input - mean) / std.

        Args:
            size: the size to pad to, or a min/max range of pad sizes. If the image is
                larger than this size, then it is cropped instead.
            mode: "center" (default) to apply padding equally on all sides, or
                "topleft" to only apply it on the bottom and right.
            image_selectors: image items to transform.
            box_selectors: boxes items to transform.
        """
        super().__init__()
        if isinstance(size, int):
            self.size = (size, size + 1)
        else:
            self.size = size

        self.mode = mode
        self.image_selectors = image_selectors
        self.box_selectors = box_selectors

    def sample_state(self) -> dict[str, Any]:
        """Randomly decide how to transform the input.

        Returns:
            dict of sampled choices
        """
        return {"size": torch.randint(low=self.size[0], high=self.size[1], size=())}

    def apply_image(self, image: torch.Tensor, state: dict[str, bool]) -> torch.Tensor:
        """Apply the sampled state on the specified image.

        Args:
            image: the image to transform.
            state: the sampled state.
        """
        size = state["size"]
        horizontal_extra = size - image.shape[-1]
        vertical_extra = size - image.shape[-2]

        def apply_padding(
            im: torch.Tensor, horizontal: bool, before: int, after: int
        ) -> torch.Tensor:
            # Before/after must either be both non-negative or both negative.
            # >=0 indicates padding while <0 indicates cropping.
            assert (before < 0 and after < 0) or (before >= 0 and after >= 0)
            if before > 0:
                # Padding.
                if horizontal:
                    padding_tuple: tuple = (before, after)
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

        image = apply_padding(image, True, horizontal_pad[0], horizontal_pad[1])
        image = apply_padding(image, False, vertical_pad[0], vertical_pad[1])
        return image

    def apply_boxes(self, boxes: Any, state: dict[str, bool]) -> torch.Tensor:
        """Apply the sampled state on the specified image.

        Args:
            boxes: the boxes to transform.
            state: the sampled state.
        """
        raise NotImplementedError

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
        state = self.sample_state()
        self.apply_fn(
            self.apply_image, input_dict, target_dict, self.image_selectors, state=state
        )
        self.apply_fn(
            self.apply_boxes, input_dict, target_dict, self.box_selectors, state=state
        )
        return input_dict, target_dict
