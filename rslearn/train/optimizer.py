"""Optimizers for rslearn."""

from dataclasses import asdict, dataclass

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
