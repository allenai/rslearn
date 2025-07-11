"""Learning rate schedulers for rslearn."""

from dataclasses import asdict, dataclass

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau


class SchedulerFactory:
    """A factory class that initializes an LR scheduler given the optimizer."""

    def build(self, optimizer: Optimizer) -> LRScheduler:
        """Build the learning rate scheduler configured by this factory class."""
        raise NotImplementedError


@dataclass
class PlateauScheduler(SchedulerFactory):
    """Plateau learning rate scheduler."""

    mode: str | None = None
    factor: float | None = None
    patience: int | None = None
    threshold: float | None = None
    threshold_mode: str | None = None
    cooldown: int | None = None
    min_lr: float | None = None
    eps: float | None = None

    def build(self, optimizer: Optimizer) -> LRScheduler:
        """Build the ReduceLROnPlateau scheduler."""
        kwargs = {k: v for k, v in asdict(self).items() if v is not None}
        return ReduceLROnPlateau(optimizer, **kwargs)
