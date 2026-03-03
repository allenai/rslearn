"""Optimizers for rslearn."""

from dataclasses import asdict, dataclass, field

import lightning as L
import torch.optim
from torch.optim import Optimizer


class OptimizerFactory:
    """A factory class that initializes the optimizer given the LightningModule."""

    def build(self, lm: L.LightningModule) -> Optimizer:
        """Build the optimizer configured by this factory class."""
        raise NotImplementedError


@dataclass
class AdamW(OptimizerFactory):
    """Factory for AdamW optimzier."""

    lr: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float | None = None
    weight_decay: float | None = None

    def build(self, lm: L.LightningModule) -> Optimizer:
        """Build the AdamW optimizer."""
        params = [p for p in lm.parameters() if p.requires_grad]
        kwargs = {k: v for k, v in asdict(self).items() if v is not None}
        return torch.optim.AdamW(params, **kwargs)


@dataclass
class GroupAdamW(OptimizerFactory):
    """AdamW with per-prefix parameter groups.

    Each entry in ``groups`` is a dict with keys ``prefix``, ``lr``, and
    optionally ``weight_decay``.  Parameters whose name starts with ``prefix``
    are placed in that group (first match wins).  Remaining trainable
    parameters go into a default group.
    """

    groups: list[dict] = field(default_factory=list)
    default_lr: float = 1e-3
    default_weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    def build(self, lm: L.LightningModule) -> Optimizer:
        """Build AdamW with per-prefix parameter groups."""
        param_groups: list[dict] = []
        assigned: set[int] = set()
        for g in self.groups:
            params = [
                p
                for n, p in lm.named_parameters()
                if p.requires_grad
                and id(p) not in assigned
                and n.startswith(g["prefix"])
            ]
            assigned.update(id(p) for p in params)
            if params:
                param_groups.append(
                    {
                        "params": params,
                        "lr": g["lr"],
                        "weight_decay": g.get(
                            "weight_decay", self.default_weight_decay
                        ),
                    }
                )
        rest = [p for p in lm.parameters() if p.requires_grad and id(p) not in assigned]
        if rest:
            param_groups.append(
                {
                    "params": rest,
                    "lr": self.default_lr,
                    "weight_decay": self.default_weight_decay,
                }
            )
        return torch.optim.AdamW(param_groups, betas=self.betas, eps=self.eps)
