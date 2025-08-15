"""FreezeUnfreeze callback."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import BaseFinetuning
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer

from rslearn.log_utils import get_logger

logger = get_logger(__name__)


def _existing_param_ids(optimizer: Optimizer) -> set[int]:
    """Collect ids of all parameters already tracked by the optimizer.

    Args:
        optimizer: The optimizer to inspect.

    Returns:
        A set of parameter ids already tracked by the optimizer.
    """
    return {id(p) for g in optimizer.param_groups for p in g["params"]}


def _iter_module_params(modules: list[torch.nn.Module]) -> list[torch.nn.Parameter]:
    """Flatten parameters from a list of modules (no duplicates, trainable first).

    Args:
        modules: A list of modules to inspect.

    Returns:
        A list of parameters from the modules, in order of appearance.
    """
    seen: set[int] = set()
    ordered: list[torch.nn.Parameter] = []
    for m in modules:
        for p in m.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                ordered.append(p)
    return ordered


class FreezeUnfreeze(BaseFinetuning):
    """Freezes a module and optionally unfreezes it after a number of epochs."""

    def __init__(
        self,
        module_selector: list[str | int],
        unfreeze_at_epoch: int | None = None,
        unfreeze_lr_factor: float = 1,
    ) -> None:
        """Creates a new FreezeUnfreeze.

        Args:
            module_selector: list of keys to access the target module to freeze. For
                example, the selector for backbone.encoder is ["backbone", "encoder"].
            unfreeze_at_epoch: optionally unfreeze the target module after this many
                epochs.
            unfreeze_lr_factor: if unfreezing, how much lower to set the learning rate
                of this module compared to the default learning rate after unfreezing,
                e.g. 10 to set it 10x lower. Default is 1 to use same learning rate.
        """
        super().__init__()
        self.module_selector = module_selector
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.unfreeze_lr_factor = unfreeze_lr_factor
        if unfreeze_at_epoch == 0:
            raise ValueError("unfreeze_at_epoch cannot be 0")

    def _get_target_module(self, pl_module: LightningModule) -> torch.nn.Module:
        target_module = pl_module
        for k in self.module_selector:
            if isinstance(k, int):
                target_module = target_module[k]
            else:
                target_module = getattr(target_module, k)
        return target_module

    def freeze_before_training(self, pl_module: LightningModule) -> None:
        """Freeze the model at the beginning of training.

        Args:
            pl_module: the LightningModule.
        """
        logger.info(f"freezing model at {self.module_selector}")
        self.freeze(self._get_target_module(pl_module))

    def finetune_function(
        self, pl_module: LightningModule, current_epoch: int, optimizer: Optimizer
    ) -> None:
        """Check whether we should unfreeze the model on each epoch.

        Args:
            pl_module: the LightningModule.
            current_epoch: the current epoch number.
            optimizer: the optimizer
        """
        if self.unfreeze_at_epoch is None:
            return
        elif current_epoch == self.unfreeze_at_epoch:
            logger.info(
                f"unfreezing model at {self.module_selector} since we are on epoch {current_epoch}"
            )
            self.unfreeze_and_add_param_group(
                modules=self._get_target_module(pl_module),
                optimizer=optimizer,
                initial_denom_lr=self.unfreeze_lr_factor,
            )
            if "scheduler" in pl_module.schedulers:
                scheduler = pl_module.schedulers["scheduler"]
                if isinstance(scheduler, ReduceLROnPlateau):
                    while len(scheduler.min_lrs) < len(optimizer.param_groups):
                        logger.info(
                            "appending to ReduceLROnPlateau scheduler min_lrs for unfreeze"
                        )
                        scheduler.min_lrs.append(scheduler.min_lrs[0])
        elif current_epoch > self.unfreeze_at_epoch:
            # always do this because overhead is minimal, and it allows restoring
            # from a checkpoint (resuming a run) without messing up unfreezing
            BaseFinetuning.make_trainable(self._get_target_module(pl_module))


@dataclass
class FTStage:
    """Specification for a single fine-tuning stage.

    Each stage is activated when the trainer reaches a specific epoch (`at_epoch`).
    Within that stage, modules whose **qualified name** (from `named_modules()`)
    matches any substring in `freeze_selectors` will be frozen, except those whose
    name matches any substring in `unfreeze_selectors`, which are forced trainable.

    Args:
        at_epoch: Epoch index at which to apply this stage (0-based).
        freeze_selectors: Substrings; any module name containing any of these will
            be frozen in this stage (unless also matched by `unfreeze_selectors`).
        unfreeze_selectors: Substrings; any module name containing any of these
            will be **unfrozen** (trainable) in this stage, overriding freezes.
        unfreeze_lr_factor: When parameters become trainable and are **not yet**
            part of the optimizer, a new param group is added with learning rate
            `base_lr / unfreeze_lr_factor`. Use 1.0 to keep the base learning rate.
        scale_existing_groups: If provided and not 1.0, multiply the learning rate
            of **all existing optimizer param groups** by this factor at the moment
            this stage is applied. Use this to calm down previously-trainable
            parts (e.g., the head) when unfreezing deeper layers.
            Set to ``None`` to leave existing groups unchanged.
    """

    at_epoch: int
    freeze_selectors: Sequence[str]
    unfreeze_selectors: Sequence[str]
    unfreeze_lr_factor: float = 1.0
    scale_existing_groups: float | None = None


class MultiStageFineTuning(BaseFinetuning):
    """Multi-stage fine-tuning with flexible name-based selection.

    Behavior per stage:
      1) Start from a **fully trainable** baseline.
      2) Optionally **scale existing optimizer groups** via `scale_existing_groups`.
      3) **Freeze** modules matching any `freeze_selectors`.
      4) **Unfreeze** modules matching any `unfreeze_selectors` (overrides step 3).
      5) For newly trainable parameters **not yet** in the optimizer, add a new
         param group using `unfreeze_lr_factor` (lr = base_lr / factor).

    Stages are applied exactly once at their `at_epoch`. The plan is recomputed
    from scratch at each stage to keep behavior predictable on resume.
    """

    def __init__(self, stages: Sequence[dict[str, Any] | FTStage]) -> None:
        """Multi-stage fine-tuning with flexible name-based selection.

        Args:
            stages: A sequence of stage specifications. Each element may be either an
                `FTStage` instance or a `dict` with keys:
                `{"at_epoch", "freeze_selectors", "unfreeze_selectors",
                  "unfreeze_lr_factor", "scale_existing_groups"}`.

        Raises:
            ValueError: If two stages specify the same `at_epoch`.
        """
        super().__init__()
        self.stages: list[FTStage] = []
        for s in stages:
            if isinstance(s, FTStage):
                self.stages.append(s)
            else:
                self.stages.append(
                    FTStage(
                        at_epoch=int(s.get("at_epoch", 0)),
                        freeze_selectors=list(s.get("freeze_selectors", [])),
                        unfreeze_selectors=list(s.get("unfreeze_selectors", [])),
                        unfreeze_lr_factor=float(s.get("unfreeze_lr_factor", 1.0)),
                        scale_existing_groups=(
                            None
                            if s.get("scale_existing_groups", None) is None
                            else float(s["scale_existing_groups"])
                        ),
                    )
                )

        # Validate uniqueness of epochs and sort stages.
        seen: set[int] = set()
        for st in self.stages:
            if st.at_epoch in seen:
                raise ValueError(f"Duplicate at_epoch in stages: {st.at_epoch}")
            if st.scale_existing_groups is not None and st.scale_existing_groups <= 0.0:
                raise ValueError("scale_existing_groups, if set, must be > 0.")
            seen.add(st.at_epoch)
        self.stages.sort(key=lambda x: x.at_epoch)

        self._applied_epochs: set[int] = set()

    @staticmethod
    def _names_matching(names: Iterable[str], selectors: Sequence[str]) -> set[str]:
        """Return the subset of `names` that contains any of the given selectors.

        Matching is done via simple substring checks (`sel in name`).

        Args:
            names: Iterable of qualified module names (e.g., from `named_modules()`).
            selectors: Substrings to match against each name. Empty strings are ignored.

        Returns:
            A set of names from `names` that match at least one selector.
        """
        if not selectors:
            return set()
        sels: list[str] = [s for s in selectors if s]
        out: set[str] = set()
        for n in names:
            if any(sel in n for sel in sels):
                out.add(n)
        return out

    @staticmethod
    def _modules_by_names(
        root: torch.nn.Module, wanted: set[str]
    ) -> list[torch.nn.Module]:
        """Map qualified names to module objects.

        Args:
            root: The root module (e.g., your LightningModule).
            wanted: Qualified names of submodules to retrieve.

        Returns:
            A list of modules corresponding to the given names that exist under `root`.
        """
        if not wanted:
            return []
        name_to_module: dict[str, torch.nn.Module] = dict(root.named_modules())
        return [name_to_module[n] for n in wanted if n in name_to_module]

    def _apply_stage(
        self, pl_module: LightningModule, optimizer: Optimizer, stage: FTStage
    ) -> None:
        """Apply a single fine-tuning stage to `pl_module` and `optimizer`.

        Order of operations:
          1) Make everything trainable (baseline).
          2) If `scale_existing_groups` is set, multiply LR of **existing** optimizer
             groups by this factor (and update ReduceLROnPlateau `min_lrs` if present).
          3) Freeze modules matched by `freeze_selectors` minus `unfreeze_selectors`.
          4) Ensure modules matched by `unfreeze_selectors` are trainable.
          5) Add new optimizer param groups for newly-trainable modules with LR
             scaled by `unfreeze_lr_factor`.

        Args:
            pl_module: The LightningModule being trained.
            optimizer: The optimizer currently used by the trainer.
            stage: The stage specification to apply at the current epoch.

        Returns:
            None.
        """
        model: torch.nn.Module = pl_module
        all_names: list[str] = [n for n, _ in model.named_modules()]

        freeze_names: set[str] = self._names_matching(all_names, stage.freeze_selectors)
        unfreeze_names: set[str] = self._names_matching(
            all_names, stage.unfreeze_selectors
        )

        # 1) Baseline: everything trainable.
        BaseFinetuning.make_trainable(model)

        # 2) Optionally scale existing optimizer groups (e.g., calm down the head).
        if (
            stage.scale_existing_groups is not None
            and stage.scale_existing_groups != 1.0
        ):
            factor: float = stage.scale_existing_groups
            for g in optimizer.param_groups:
                old_lr = float(g.get("lr", 0.0))
                g["lr"] = old_lr * factor
            # Keep ReduceLROnPlateau bounds consistent if present.
            if hasattr(pl_module, "schedulers") and "scheduler" in getattr(
                pl_module, "schedulers", {}
            ):
                scheduler = pl_module.schedulers["scheduler"]
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.min_lrs = [float(m) * factor for m in scheduler.min_lrs]

        # 3) Freeze matched, except those explicitly unfreezed.
        to_freeze: set[str] = freeze_names - unfreeze_names
        freeze_modules: list[torch.nn.Module] = self._modules_by_names(model, to_freeze)
        if freeze_modules:
            logger.info(
                f"[FT stage @ epoch {stage.at_epoch}] Freezing {len(freeze_modules)} modules "
                f"(matched: {sorted(list(to_freeze))[:5]}{'...' if len(to_freeze) > 5 else ''})"
            )
            for m in freeze_modules:
                self.freeze(m)

        # 4) Ensure explicitly unfreezed modules are trainable.
        unfreeze_modules: list[torch.nn.Module] = self._modules_by_names(
            model, unfreeze_names
        )
        if unfreeze_modules:
            logger.info(
                f"[FT stage @ epoch {stage.at_epoch}] Unfreezing {len(unfreeze_modules)} modules "
                f"(matched: {sorted(list(unfreeze_names))[:5]}{'...' if len(unfreeze_names) > 5 else ''})"
            )
            for m in unfreeze_modules:
                BaseFinetuning.make_trainable(m)

            # 5) Add *newly-trainable* params only (no duplicates)
            denom: float = (
                stage.unfreeze_lr_factor if stage.unfreeze_lr_factor != 1.0 else 1.0
            )
            all_params = _iter_module_params(unfreeze_modules)
            already = _existing_param_ids(optimizer)
            new_params = [
                p for p in all_params if p.requires_grad and id(p) not in already
            ]

            if new_params:
                # Use current "base" lr (after any scale_existing_groups) as the reference
                base_lr = float(optimizer.param_groups[0].get("lr", 0.0))
                group_lr = base_lr / denom if denom != 0 else base_lr
                optimizer.add_param_group({"params": new_params, "lr": group_lr})

                # Extend ReduceLROnPlateau.min_lrs to match param group count
                if hasattr(pl_module, "schedulers") and "scheduler" in getattr(
                    pl_module, "schedulers", {}
                ):
                    scheduler = pl_module.schedulers["scheduler"]
                    if isinstance(scheduler, ReduceLROnPlateau):
                        while len(scheduler.min_lrs) < len(optimizer.param_groups):
                            logger.info(
                                "Extending ReduceLROnPlateau.min_lrs for new param group"
                            )
                            scheduler.min_lrs.append(scheduler.min_lrs[0])

        # Summary logging.
        trainable, frozen = 0, 0
        for p in model.parameters():
            if p.requires_grad:
                trainable += p.numel()
            else:
                frozen += p.numel()
        logger.info(
            f"[FT stage @ epoch {stage.at_epoch}] Trainable params: {trainable:,} | Frozen params: {frozen:,}"
        )

    def freeze_before_training(self, pl_module: LightningModule) -> None:
        """Hook: Called by Lightning before the first training epoch.

        If a stage is scheduled at epoch 0, we defer its application to the first
        call of `finetune_function` (when the optimizer is available). Otherwise,
        we simply log that training begins with a fully trainable model.

        Args:
            pl_module: The LightningModule being trained.
        """
        if any(st.at_epoch == 0 for st in self.stages):
            logger.info(
                "Stage scheduled for epoch 0 will be applied at the first finetune_function "
                "call when the optimizer is available."
            )
        else:
            logger.info("No stage at epoch 0; starting fully trainable by default.")

    def finetune_function(
        self, pl_module: LightningModule, current_epoch: int, optimizer: Optimizer
    ) -> None:
        """Hook: Called by Lightning at each epoch to adjust trainability.

        Applies any stage whose `at_epoch` equals `current_epoch` and that has not
        yet been applied in this run. Recomputes freeze/unfreeze decisions from
        scratch for that stage.

        Args:
            pl_module: The LightningModule being trained.
            current_epoch: The current epoch index (0-based).
            optimizer: The optimizer currently used by the trainer.
        """
        for st in self.stages:
            if st.at_epoch == current_epoch and st.at_epoch not in self._applied_epochs:
                logger.info(
                    f"Applying multi-stage fine-tuning plan at epoch {current_epoch}"
                )
                self._apply_stage(pl_module, optimizer, st)
                self._applied_epochs.add(st.at_epoch)
