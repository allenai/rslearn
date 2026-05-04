"""Tests for LayerDecayAdamW and SimpleFreeze."""

from unittest.mock import MagicMock

import pytest
import torch.nn as nn

from rslearn.models.olmoearth_pretrain.optimizer import (
    ENCODER_PREFIX,
    LayerDecayAdamW,
    SimpleFreeze,
)

# ----------------------------- Helpers ------------------------------------ #


class FakeEncoder(nn.Module):
    """Mimics the OlmoEarth encoder structure with blocks + patch_embeddings + norm."""

    def __init__(self, num_blocks: int = 4, dim: int = 8) -> None:
        super().__init__()
        self.patch_embeddings = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(dim)


class FakeOlmoEarth(nn.Module):
    def __init__(self, num_blocks: int = 4) -> None:
        super().__init__()
        self.model = FakeEncoder(num_blocks=num_blocks)


class FakeMultiTaskModel(nn.Module):
    def __init__(self, num_blocks: int = 4) -> None:
        super().__init__()
        self.encoder = nn.ModuleList([FakeOlmoEarth(num_blocks=num_blocks)])
        self.decoders = nn.ModuleDict({"head": nn.Linear(8, 2)})


class FakeLightningModule(nn.Module):
    """Minimal stand-in for RslearnLightningModule."""

    def __init__(self, num_blocks: int = 4) -> None:
        super().__init__()
        self.model = FakeMultiTaskModel(num_blocks=num_blocks)


# ----------------------------- LayerDecayAdamW ---------------------------- #


class TestLayerDecayAdamW:
    def test_creates_correct_param_groups(self) -> None:
        num_blocks = 4
        decay = 0.5
        base_lr = 0.01
        lm = FakeLightningModule(num_blocks=num_blocks)

        factory = LayerDecayAdamW(
            lr=base_lr,
            layer_decay_rate=decay,
            num_layers=num_blocks,
            encoder_prefix=ENCODER_PREFIX,
        )
        optimizer = factory.build(lm)

        all_param_ids = {id(p) for p in lm.parameters()}
        tracked_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
        assert all_param_ids == tracked_ids, "All params must be in the optimizer"

        lrs = [g["lr"] for g in optimizer.param_groups]
        assert lrs[0] < lrs[-1], "Early layers should have lower LR than decoder"
        assert lrs[0] == pytest.approx(base_lr * decay**num_blocks, rel=1e-6)
        assert lrs[-1] == pytest.approx(base_lr, rel=1e-6)

    def test_includes_frozen_params(self) -> None:
        lm = FakeLightningModule(num_blocks=4)
        for p in lm.model.encoder[0].model.blocks[0].parameters():
            p.requires_grad = False

        factory = LayerDecayAdamW(
            lr=0.01,
            layer_decay_rate=0.5,
            num_layers=4,
            encoder_prefix=ENCODER_PREFIX,
        )
        optimizer = factory.build(lm)

        all_param_ids = {id(p) for p in lm.parameters()}
        tracked_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
        assert all_param_ids == tracked_ids, (
            "Frozen params must still be in the optimizer for SimpleFreeze compatibility"
        )


# ----------------------------- SimpleFreeze ------------------------------- #


class TestSimpleFreeze:
    def _make_trainer(self, epoch: int) -> MagicMock:
        trainer = MagicMock()
        trainer.current_epoch = epoch
        return trainer

    def test_freezes_before_unfreeze_epoch(self) -> None:
        lm = FakeLightningModule(num_blocks=4)
        cb = SimpleFreeze(module_selector=["model", "encoder"], unfreeze_at_epoch=5)

        cb.on_train_epoch_start(self._make_trainer(0), lm)

        encoder = lm.model.encoder
        assert all(not p.requires_grad for p in encoder.parameters())

    def test_unfreezes_at_epoch(self) -> None:
        lm = FakeLightningModule(num_blocks=4)
        cb = SimpleFreeze(module_selector=["model", "encoder"], unfreeze_at_epoch=5)

        cb.on_train_epoch_start(self._make_trainer(4), lm)
        assert all(not p.requires_grad for p in lm.model.encoder.parameters())

        cb.on_train_epoch_start(self._make_trainer(5), lm)
        assert all(p.requires_grad for p in lm.model.encoder.parameters())

    def test_stays_unfrozen_after_epoch(self) -> None:
        lm = FakeLightningModule(num_blocks=4)
        cb = SimpleFreeze(module_selector=["model", "encoder"], unfreeze_at_epoch=5)

        cb.on_train_epoch_start(self._make_trainer(10), lm)
        assert all(p.requires_grad for p in lm.model.encoder.parameters())

    def test_does_not_affect_other_modules(self) -> None:
        lm = FakeLightningModule(num_blocks=4)
        cb = SimpleFreeze(module_selector=["model", "encoder"], unfreeze_at_epoch=5)

        cb.on_train_epoch_start(self._make_trainer(0), lm)

        assert all(p.requires_grad for p in lm.model.decoders.parameters()), (
            "Decoder params should remain trainable"
        )

    def test_composes_with_layer_decay_adamw(self) -> None:
        """Verify that SimpleFreeze + LayerDecayAdamW work together."""
        num_blocks = 4
        lm = FakeLightningModule(num_blocks=num_blocks)
        cb = SimpleFreeze(module_selector=["model", "encoder"], unfreeze_at_epoch=2)

        factory = LayerDecayAdamW(
            lr=0.01,
            layer_decay_rate=0.5,
            num_layers=num_blocks,
            encoder_prefix=ENCODER_PREFIX,
        )

        cb.on_train_epoch_start(self._make_trainer(0), lm)
        optimizer = factory.build(lm)

        all_param_ids = {id(p) for p in lm.parameters()}
        tracked_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
        assert all_param_ids == tracked_ids

        encoder = lm.model.encoder
        assert all(not p.requires_grad for p in encoder.parameters())

        num_groups_before = len(optimizer.param_groups)
        cb.on_train_epoch_start(self._make_trainer(2), lm)
        assert all(p.requires_grad for p in encoder.parameters())
        assert len(optimizer.param_groups) == num_groups_before, (
            "Param groups should not change on unfreeze"
        )
