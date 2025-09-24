"""Tests for adapters callback."""

from typing import Any

import pytest
import torch.nn as nn

from rslearn.train.callbacks.adapters import ActivateLayers


class DummyAdapter(nn.Module):
    """Dummy adapter module."""

    def __init__(self) -> None:
        """Initialize the dummy adapter module."""
        super().__init__()
        self.active = False  # default state


class ToyModel(nn.Module):
    """Toy model for testing."""

    def __init__(self) -> None:
        """Initialize the toy model."""
        super().__init__()
        # Names intentionally include substrings we'll select on
        self.encoder = nn.ModuleDict(
            {
                "adapterA": DummyAdapter(),
            }
        )
        self.decoder = nn.ModuleDict(
            {
                "block": nn.ModuleDict(
                    {
                        "adapterB": DummyAdapter(),
                    }
                )
            }
        )
        self.other = nn.Linear(2, 2)  # should be unaffected


class FakeTrainer:
    """Fake trainer for testing."""

    def __init__(self, current_epoch: int) -> None:
        """Initialize the fake trainer."""
        self.current_epoch = current_epoch


@pytest.fixture
def model() -> ToyModel:
    """Fixture for the toy model."""
    return ToyModel()


@pytest.fixture
def selectors() -> list[dict[str, Any]]:
    """Fixture for the selectors."""
    return [
        {"name": "adapterA", "at_epoch": 1},
        {"name": "adapterB", "at_epoch": 2},
    ]


def _run(callback: ActivateLayers, epoch: int, model: ToyModel) -> None:
    """Run the callback.

    Args:
        callback: The callback to run.
        epoch: The epoch to run the callback at.
        model: The model to run the callback on.
    """
    trainer = FakeTrainer(current_epoch=epoch)
    # match the signature used in the provided callback
    callback.on_train_epoch_start(trainer, model)


def test_activation_progression(
    model: ToyModel, selectors: list[dict[str, Any]]
) -> None:
    """Test the activation progression.

    Args:
        model: The model to run the callback on.
        selectors: The selectors to run the callback with.
    """
    cb = ActivateLayers(selectors)

    # Before any activation epoch
    _run(cb, epoch=0, model=model)
    assert not model.encoder["adapterA"].active
    assert not model.decoder["block"]["adapterB"].active

    # First selector should flip at epoch >= 1
    _run(cb, epoch=1, model=model)
    assert model.encoder["adapterA"].active
    assert not model.decoder["block"]["adapterB"].active

    # Second selector should flip at epoch >= 2
    _run(cb, epoch=2, model=model)
    assert model.encoder["adapterA"].active
    assert model.decoder["block"]["adapterB"].active

    # Non-selected modules remain untouched
    assert not hasattr(model.other, "active")
