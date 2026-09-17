"""Unit tests for rslearn.train.transforms.concatenate."""

from datetime import UTC, datetime, timedelta

import pytest
import torch

from rslearn.train.model_context import RasterImage
from rslearn.train.transforms.concatenate import Concatenate, ConcatenateDim


def _single_timestep_image(value: float, month: int) -> RasterImage:
    """Build a (1, 1, 2, 2) RasterImage with one timestamp in the given month."""
    image = torch.full((1, 1, 2, 2), value, dtype=torch.float32)
    start = datetime(2024, month, 1, tzinfo=UTC)
    return RasterImage(image, timestamps=[(start, start + timedelta(days=30))])


def test_concatenate_time() -> None:
    """Test Mask with default arguments where image should be set 0."""
    concat = Concatenate(
        selections={"image1": [0], "image2": [0]},
        concatenate_dim=ConcatenateDim.TIME,
        output_selector="image3",
    )
    input_image = torch.ones((2, 4, 2, 2), dtype=torch.float32)
    input_image[0] = 0
    input_dict = {
        "image1": RasterImage(input_image.clone()),
        "image2": RasterImage(input_image.clone()),
    }
    input_dict, _ = concat(input_dict, {})
    assert input_dict["image3"].shape == (1, 8, 2, 2)
    assert (input_dict["image3"].image == 0).all()


def test_concatenate_channels() -> None:
    """Test Mask with default arguments where image should be set 0."""
    concat = Concatenate(
        selections={"image1": [0], "image2": [0]},
        concatenate_dim=ConcatenateDim.CHANNEL,
        output_selector="image3",
    )
    input_image = torch.ones((2, 4, 2, 2), dtype=torch.float32)
    input_image[0] = 0
    input_dict = {
        "image1": RasterImage(input_image.clone()),
        "image2": RasterImage(input_image.clone()),
    }
    input_dict, _ = concat(input_dict, {})
    assert input_dict["image3"].shape == (2, 4, 2, 2)
    assert (input_dict["image3"].image == 0).all()


def test_concatenate_time_extends_timestamps() -> None:
    """TIME concatenation should build a length-T timestamp list across inputs."""
    concat = Concatenate(
        selections={"mo01": [], "mo02": [], "mo03": []},
        concatenate_dim=ConcatenateDim.TIME,
        output_selector="image",
    )
    input_dict = {
        "mo01": _single_timestep_image(1.0, 1),
        "mo02": _single_timestep_image(2.0, 2),
        "mo03": _single_timestep_image(3.0, 3),
    }
    input_dict, _ = concat(input_dict, {})
    result = input_dict["image"]
    assert result.shape == (1, 3, 2, 2)
    assert result.timestamps is not None
    assert len(result.timestamps) == 3
    assert [ts[0].month for ts in result.timestamps] == [1, 2, 3]


def test_concatenate_skip_missing() -> None:
    """skip_missing should drop absent selectors and keep timestamps consistent."""
    concat = Concatenate(
        selections={"mo01": [], "mo02": [], "mo03": []},
        concatenate_dim=ConcatenateDim.TIME,
        output_selector="image",
        skip_missing=True,
    )
    # mo02 is missing from the input dict.
    input_dict = {
        "mo01": _single_timestep_image(1.0, 1),
        "mo03": _single_timestep_image(3.0, 3),
    }
    input_dict, _ = concat(input_dict, {})
    result = input_dict["image"]
    assert result.shape == (1, 2, 2, 2)
    assert result.timestamps is not None
    assert [ts[0].month for ts in result.timestamps] == [1, 3]


def test_concatenate_missing_without_skip_raises() -> None:
    """Without skip_missing, an absent selector raises."""
    concat = Concatenate(
        selections={"mo01": [], "mo02": []},
        concatenate_dim=ConcatenateDim.TIME,
        output_selector="image",
    )
    input_dict = {"mo01": _single_timestep_image(1.0, 1)}
    with pytest.raises(KeyError):
        concat(input_dict, {})
