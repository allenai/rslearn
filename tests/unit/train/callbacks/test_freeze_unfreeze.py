"""Tests for the MultiStageFineTuning callback."""

import pytest
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from rslearn.train.callbacks.freeze_unfreeze import FTStage, MultiStageFineTuning

# ---------------------------- Fixtures & helpers ---------------------------- #


class ToyBackbone(nn.Module):
    """Tiny backbone with an encoder and a faux LoRA adapter layer."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(8, 8),  # backbone.encoder.0
            nn.ReLU(),
            nn.Linear(8, 8),  # backbone.encoder.2
        )
        self.lora_adapter = nn.Linear(8, 8)  # backbone.lora_adapter


class ToyHead(nn.Module):
    """Simple decoder head."""

    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(8, 4),  # head.decoder.0
            nn.ReLU(),
            nn.Linear(4, 2),  # head.decoder.2
        )


class ToyModel(nn.Module):
    """Composite model = backbone + head."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = ToyBackbone()
        self.head = ToyHead()


@pytest.fixture()
def model() -> ToyModel:
    """Provide a fresh toy model for each test.

    Returns:
        A new `ToyModel` instance.
    """
    return ToyModel()


@pytest.fixture()
def scheduler_and_pl(model: ToyModel) -> tuple[nn.Module, SGD, ReduceLROnPlateau]:
    """Create a dummy Lightning-like module, optimizer, and scheduler.

    Starts the optimizer with **only head parameters** to exercise param-group
    additions when other parts become trainable later. Also seeds the scheduler's
    `min_lrs` with a non-zero value so LR scaling can be asserted precisely.

    Args:
        model: The toy model from the fixture.

    Returns:
        Tuple of:
            - a dummy `pl_module` exposing `schedulers={"scheduler": ReduceLROnPlateau(...)}`
            - the `SGD` optimizer tracking only head parameters
            - the `ReduceLROnPlateau` scheduler bound to that optimizer
    """

    class PL(nn.Module):
        pass

    pl = PL()
    head_params = list(model.head.parameters())
    opt = SGD(head_params, lr=0.1)
    sch = ReduceLROnPlateau(opt)
    # Make min_lrs non-zero to observe multiplicative scaling
    sch.min_lrs = [opt.param_groups[0]["lr"] * 0.2]  # 0.02 initially
    pl.schedulers = {"scheduler": sch}  # type: ignore[attr-defined]
    return pl, opt, sch


# --------------------------------- Tests ----------------------------------- #


def test_duplicate_epoch_raises() -> None:
    """Ensure configuring two stages at the same epoch raises ValueError."""
    with pytest.raises(ValueError):
        MultiStageFineTuning(
            [
                {"at_epoch": 0, "freeze_selectors": [], "unfreeze_selectors": []},
                {"at_epoch": 0, "freeze_selectors": [], "unfreeze_selectors": []},
            ]
        )


def test_unfreeze_overrides_freeze(
    model: ToyModel,
    scheduler_and_pl: tuple[nn.Module, SGD, ReduceLROnPlateau],
) -> None:
    """Unfreeze selectors must take precedence over freeze selectors.

    - Freeze `"backbone"` broadly.
    - Explicitly unfreeze `"backbone.encoder"` and verify it remains trainable.
    - Other `"backbone.*"` modules should be frozen.
    """
    pl, opt, _ = scheduler_and_pl

    cb = MultiStageFineTuning(
        [
            FTStage(
                at_epoch=0,
                freeze_selectors=["backbone"],
                unfreeze_selectors=["backbone.encoder"],
                unfreeze_lr_factor=1.0,
            ),
        ]
    )

    cb.finetune_function(pl_module=model, current_epoch=0, optimizer=opt)

    for name, mod in model.named_modules():
        if not name:
            continue
        if name.startswith("backbone.encoder"):
            # encoder should be trainable
            assert all(
                p.requires_grad for p in mod.parameters()
            ), f"{name} should be trainable"
        elif name == "backbone":
            # parent container: check only its *own* params, not recursive
            own_params = list(mod.parameters(recurse=False))
            if own_params:  # some containers have none
                assert all(
                    not p.requires_grad for p in own_params
                ), f"{name} (own params) should be frozen"
        elif name.startswith("backbone."):
            # specific non-encoder children should be frozen
            assert all(
                not p.requires_grad for p in mod.parameters()
            ), f"{name} should be frozen"


def test_multistage_lora_plus_head_then_all_with_lr_scaling(
    model: ToyModel,
    scheduler_and_pl: tuple[nn.Module, SGD, ReduceLROnPlateau],
) -> None:
    """Exercise a 3-stage plan with LR scaling: head-only → (scale head; LoRA+head) → full model.

    Verifies:
      1) Stage 0 (epoch 0): backbone frozen, head trainable; no new param groups since head
         was already tracked by optimizer.
      2) Stage 1 (epoch 3): scale existing groups by ×0.5 (head LR halves, scheduler.min_lrs halves),
         keep backbone frozen but unfreeze `lora_adapter` (and keep head trainable); new param groups
         added for newly trainable modules at LR = (scaled_base_lr)/10. Also verify scheduler.min_lrs
         is extended to match param group count, with appended entries equal to the (scaled) first min_lr.
      3) Stage 2 (epoch 6): unfreeze everything and ensure all params are tracked by optimizer.
    """
    pl, opt, sch = scheduler_and_pl
    init_group_param_ids: set[int] = {
        id(p) for g in opt.param_groups for p in g["params"]
    }
    init_head_lr = opt.param_groups[0]["lr"]  # 0.1
    init_min_lr0 = sch.min_lrs[0]  # 0.02

    stages = [
        {
            "at_epoch": 0,
            "freeze_selectors": ["backbone", "encoder"],
            "unfreeze_selectors": ["head", "decoder"],
            "unfreeze_lr_factor": 1.0,
        },
        {
            "at_epoch": 3,
            "freeze_selectors": ["backbone", "encoder"],
            "unfreeze_selectors": ["lora_adapter", "head", "decoder"],
            "unfreeze_lr_factor": 10.0,  # new groups at (current base)/10
            "scale_existing_groups": 0.5,
        },  # scale head and any existing groups by ×0.5
        {
            "at_epoch": 6,
            "freeze_selectors": [],
            "unfreeze_selectors": ["backbone", "head"],
            "unfreeze_lr_factor": 1.0,
        },
    ]
    cb = MultiStageFineTuning(stages)

    # Stage 0
    cb.finetune_function(pl_module=model, current_epoch=0, optimizer=opt)
    for name, mod in model.named_modules():
        if not name:
            continue
        if name.startswith("head"):
            assert all(p.requires_grad for p in mod.parameters())
        if name.startswith("backbone"):
            assert all(not p.requires_grad for p in mod.parameters())

    assert len(opt.param_groups) == 1
    assert {
        id(p) for g in opt.param_groups for p in g["params"]
    } == init_group_param_ids

    # Stage 1
    prev_ngroups = len(opt.param_groups)
    cb.finetune_function(pl_module=model, current_epoch=3, optimizer=opt)

    # Existing group (head) LR should be halved (callback scales optimizer groups)
    scaled_head_lr = opt.param_groups[0]["lr"]
    assert scaled_head_lr == pytest.approx(init_head_lr * 0.5, rel=1e-6)
    assert sch.min_lrs[0] == pytest.approx(init_min_lr0, rel=1e-6)
    assert all(
        m == sch.min_lrs[0] for m in sch.min_lrs
    ), "All min_lrs should equal the first value"

    # Module trainability checks (unchanged)
    for name, mod in model.named_modules():
        if not name:
            continue
        if name.startswith("backbone.lora_adapter") or name.startswith("head"):
            assert all(
                p.requires_grad for p in mod.parameters()
            ), f"{name} should be trainable"
        elif name.startswith("backbone") and name != "backbone":
            assert all(
                not p.requires_grad for p in mod.parameters()
            ), f"{name} should be frozen"

    # New param groups added for newly trainable modules at LR = scaled_head_lr / 10
    assert (
        len(opt.param_groups) > prev_ngroups
    ), "Expected new param groups for newly trainable modules"
    new_group_lrs = []
    for g in opt.param_groups:
        if any(id(p) not in init_group_param_ids for p in g["params"]):
            new_group_lrs.append(g["lr"])
    assert new_group_lrs, "Expected at least one new param group for unfreezed modules"
    for lr in new_group_lrs:
        assert lr == pytest.approx(scaled_head_lr / 10.0, rel=1e-6)

    # Stage 2
    cb.finetune_function(pl_module=model, current_epoch=6, optimizer=opt)
    assert all(
        p.requires_grad for p in model.parameters()
    ), "All params should be trainable now"

    all_param_ids = {id(p) for p in model.parameters()}
    tracked_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert (
        all_param_ids == tracked_ids
    ), "All params should be tracked by optimizer after final stage"


def test_empty_selectors_are_ignored(
    model: ToyModel,
    scheduler_and_pl: tuple[nn.Module, SGD, ReduceLROnPlateau],
) -> None:
    """Empty strings in selectors are ignored and cause no broad effect.

    Confirms that providing empty strings does not freeze or unfreeze everything.
    """
    pl, opt, _ = scheduler_and_pl
    cb = MultiStageFineTuning(
        [
            {"at_epoch": 0, "freeze_selectors": [""], "unfreeze_selectors": [""]},
        ]
    )

    cb.finetune_function(pl_module=model, current_epoch=0, optimizer=opt)

    # Head params (already in optimizer) remain trainable.
    assert all(p.requires_grad for p in model.head.parameters())
    # Sanity: not all params were flipped off due to empty selectors.
    assert any(p.requires_grad for p in model.parameters())


def test_reapplying_same_epoch_has_no_effect(
    model: ToyModel,
    scheduler_and_pl: tuple[nn.Module, SGD, ReduceLROnPlateau],
) -> None:
    """Re-invoking the same stage again should not change parameter states.

    The callback records applied epochs; calling with the same epoch a second time
    should be a no-op for trainability.
    """
    pl, opt, _ = scheduler_and_pl
    cb = MultiStageFineTuning(
        [
            {
                "at_epoch": 0,
                "freeze_selectors": ["backbone"],
                "unfreeze_selectors": ["head"],
            },
        ]
    )

    cb.finetune_function(pl_module=model, current_epoch=0, optimizer=opt)
    state_after_first = [p.requires_grad for p in model.parameters()]

    cb.finetune_function(pl_module=model, current_epoch=0, optimizer=opt)
    state_after_second = [p.requires_grad for p in model.parameters()]

    assert state_after_first == state_after_second
