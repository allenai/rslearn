"""Unit tests for rslearn.train.transforms.crop.Crop."""

from typing import Any

import torch

from rslearn.train.model_context import RasterImage
from rslearn.train.transforms.crop import Crop


def _make_input(
    h: int = 16, w: int = 16, channels: int = 3, selector: str = "image"
) -> dict[str, Any]:
    """Create an input dict with a single RasterImage filled with arange values."""
    image = torch.arange(channels * h * w, dtype=torch.float32).reshape(
        1, channels, h, w
    )
    return {selector: RasterImage(image)}


class TestRandomCrop:
    """Tests for the default random crop mode."""

    def test_output_shape(self) -> None:
        """Random crop with exact crop size produces the expected size."""
        crop = Crop(crop_size=8, image_selectors=["image"])
        input_dict = _make_input(16, 16)
        out, _ = crop(input_dict, {})
        assert out["image"].image.shape == (1, 3, 8, 8)

    def test_crop_size_range(self) -> None:
        """When crop_size is a (min, max) range, output size is within that range."""
        crop = Crop(crop_size=(4, 9), image_selectors=["image"])

        for _ in range(10):
            input_dict = _make_input(16, 16)
            out, _ = crop(input_dict, {})
            h, w = out["image"].image.shape[-2:]
            assert h == w
            assert 4 <= h < 9


class TestDeterministicCrop:
    """Tests for the deterministic crop mode (offset parameter)."""

    def test_offset_produces_correct_region(self) -> None:
        """Crop with offset extracts the exact expected region."""
        # Use a known image so we can verify pixel values.
        image = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
        crop = Crop(crop_size=4, offset=(2, 3), image_selectors=["image"])
        input_dict: dict[str, Any] = {"image": RasterImage(image)}
        out, _ = crop(input_dict, {})
        # Expected: rows 3..7, cols 2..6 of the 8x8 image.
        expected = image[:, :, 3:7, 2:6]
        assert torch.equal(out["image"].image, expected)

    def test_multi_resolution(self) -> None:
        """Deterministic crop scales coordinates correctly across resolutions."""
        # low=8x8, high=16x16. crop_size=4, offset=(2,1) in low-res coords.
        # low should get rows 1:5, cols 2:6 (4x4), since the crop_size and offset are
        # based on the smallest shape input.
        # high should get rows 2:10, cols 4:12 (8x8) — doubled coordinates.
        low = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
        high = torch.arange(256, dtype=torch.float32).reshape(1, 1, 16, 16)
        crop = Crop(crop_size=4, offset=(2, 1), image_selectors=["low", "high"])
        input_dict: dict[str, Any] = {
            "low": RasterImage(low),
            "high": RasterImage(high),
        }
        out, _ = crop(input_dict, {})
        assert torch.equal(out["low"].image, low[:, :, 1:5, 2:6])
        assert torch.equal(out["high"].image, high[:, :, 2:10, 4:12])


class TestSkipMissing:
    """Tests for skip_missing behavior."""

    def test_skip_missing_true(self) -> None:
        """With skip_missing=True, missing selectors are silently skipped."""
        crop = Crop(
            crop_size=4,
            image_selectors=["image", "missing"],
            skip_missing=True,
        )
        input_dict = _make_input(8, 8)
        out, _ = crop(input_dict, {})
        assert out["image"].image.shape == (1, 3, 4, 4)

    def test_skip_missing_false(self) -> None:
        """Without skip_missing, a missing selector raises KeyError."""
        crop = Crop(
            crop_size=4,
            image_selectors=["image", "missing"],
            skip_missing=False,
        )
        input_dict = _make_input(8, 8)
        try:
            crop(input_dict, {})
            assert False, "Expected KeyError to be raised"
        except KeyError:
            pass


class TestApplyBoxes:
    """Tests for box cropping behavior."""

    def test_boxes_mixed_inside_outside(self) -> None:
        """Only boxes with nonzero area after clipping are kept."""
        # Crop to (0, 0, 4, 4).
        crop = Crop(
            crop_size=4,
            offset=(0, 0),
            image_selectors=["image"],
            box_selectors=["target/boxes"],
        )
        image = torch.zeros(1, 1, 8, 8)
        boxes = torch.tensor(
            [
                [1, 1, 3, 3, 0],  # inside -> kept
                [5, 5, 7, 7, 1],  # outside -> removed
                [0, 0, 6, 6, 2],  # partially inside -> clipped to (0, 0, 4, 4)
            ],
            dtype=torch.float32,
        )
        input_dict: dict[str, Any] = {"image": RasterImage(image)}
        target_dict: dict[str, Any] = {"boxes": boxes}
        _, out_tgt = crop(input_dict, target_dict)
        result = out_tgt["boxes"]
        assert torch.equal(
            result,
            torch.tensor(
                [[1, 1, 3, 3, 0], [0, 0, 4, 4, 2]],
                dtype=torch.float32,
            ),
        )
