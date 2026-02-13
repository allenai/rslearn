"""Unit tests for rslearn.train.transforms.transform base module."""

import torch

from rslearn.train.model_context import RasterImage
from rslearn.train.transforms.flip import Flip
from rslearn.train.transforms.normalize import Normalize
from rslearn.train.transforms.transform import selector_exists


class TestSelectorExists:
    """Tests for the selector_exists helper function."""

    def test_simple_key_exists(self) -> None:
        """Test that selector_exists returns True for existing simple keys."""
        input_dict = {"image": RasterImage(torch.zeros((1, 1, 4, 4)))}
        assert selector_exists(input_dict, {}, "image") is True

    def test_simple_key_missing(self) -> None:
        """Test that selector_exists returns False for missing keys."""
        input_dict = {"image": RasterImage(torch.zeros((1, 1, 4, 4)))}
        assert selector_exists(input_dict, {}, "missing_key") is False

    def test_nested_key_exists(self) -> None:
        """Test that selector_exists works with nested selectors."""
        input_dict = {
            "modality": {
                "sentinel2": RasterImage(torch.zeros((1, 1, 4, 4))),
            }
        }
        assert selector_exists(input_dict, {}, "modality/sentinel2") is True

    def test_nested_key_missing(self) -> None:
        """Test that selector_exists returns False for missing nested keys."""
        input_dict = {
            "modality": {
                "sentinel2": RasterImage(torch.zeros((1, 1, 4, 4))),
            }
        }
        assert selector_exists(input_dict, {}, "modality/sentinel1") is False

    def test_input_prefix(self) -> None:
        """Test that selector_exists works with 'input/' prefix."""
        input_dict = {"image": RasterImage(torch.zeros((1, 1, 4, 4)))}
        assert selector_exists(input_dict, {}, "input/image") is True
        assert selector_exists(input_dict, {}, "input/missing") is False

    def test_target_prefix(self) -> None:
        """Test that selector_exists works with 'target/' prefix."""
        target_dict = {"label": torch.zeros((1, 4, 4))}
        assert selector_exists({}, target_dict, "target/label") is True
        assert selector_exists({}, target_dict, "target/missing") is False


class TestSkipMissing:
    """Tests for transforms with skip_missing=True."""

    def test_flip_skips_missing_selector(self) -> None:
        """Test that Flip with skip_missing=True skips missing selectors."""
        flip = Flip(
            horizontal=True,
            vertical=True,
            image_selectors=["sentinel2", "sentinel1"],  # sentinel1 is missing
            skip_missing=True,
        )
        input_dict = {
            "sentinel2": RasterImage(torch.ones((1, 1, 4, 4))),
            # sentinel1 is intentionally missing
        }
        # Should not raise KeyError
        output_dict, _ = flip(input_dict, {})
        # sentinel2 should still be processed
        assert "sentinel2" in output_dict

    def test_flip_raises_without_skip_missing(self) -> None:
        """Test that Flip without skip_missing raises KeyError for missing selectors."""
        flip = Flip(
            horizontal=True,
            vertical=True,
            image_selectors=["sentinel2", "sentinel1"],  # sentinel1 is missing
            skip_missing=False,
        )
        input_dict = {
            "sentinel2": RasterImage(torch.ones((1, 1, 4, 4))),
            # sentinel1 is intentionally missing
        }
        # Should raise KeyError
        try:
            flip(input_dict, {})
            assert False, "Expected KeyError to be raised"
        except KeyError:
            pass

    def test_normalize_skips_missing_selector(self) -> None:
        """Test that Normalize with skip_missing=True skips missing selectors."""
        normalize = Normalize(
            mean=0.5,
            std=0.5,
            selectors=["sentinel2", "sentinel1"],  # sentinel1 is missing
            skip_missing=True,
        )
        input_dict = {
            "sentinel2": RasterImage(torch.ones((1, 1, 4, 4))),
            # sentinel1 is intentionally missing
        }
        # Should not raise KeyError
        output_dict, _ = normalize(input_dict, {})
        # sentinel2 should be normalized
        assert "sentinel2" in output_dict
        # Check that normalization was applied: (1 - 0.5) / 0.5 = 1.0
        assert output_dict["sentinel2"].image[0, 0, 0, 0] == 1.0
