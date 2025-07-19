"""Learning rate schedulers for rslearn."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    ReduceLROnPlateau,
)

from rslearn.log_utils import get_logger

logger = get_logger(__name__)


class SchedulerFactory(ABC):
    """A factory class that initializes an LR scheduler given the optimizer."""

    def get_kwargs(self) -> dict:
        """Get the keyword arguments for the scheduler."""
        return {k: v for k, v in asdict(self).items() if v is not None}  # type: ignore

    @abstractmethod
    def build(self, optimizer: Optimizer) -> LRScheduler:
        """Build the learning rate scheduler configured by this factory class."""
        logger.info(
            f"Using scheduler {self.__class__.__name__} with kwargs {self.get_kwargs()}"
        )


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
        super().build(optimizer)
        return ReduceLROnPlateau(optimizer, **self.get_kwargs())


@dataclass
class CosineAnnealingScheduler(SchedulerFactory):
    """Cosine annealing learning rate scheduler."""

    T_max: int
    eta_min: float | None = None

    def build(self, optimizer: Optimizer) -> LRScheduler:
        """Build the CosineAnnealingLR scheduler."""
        super().build(optimizer)
        return CosineAnnealingLR(optimizer, **self.get_kwargs())
