"""Learning rate schedulers for rslearn."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LRScheduler,
    MultiStepLR,
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
class MultiStepScheduler(SchedulerFactory):
    """Step learning rate scheduler."""

    milestones: list[int]
    gamma: float | None = None
    last_epoch: int | None = None

    def build(self, optimizer: Optimizer) -> LRScheduler:
        """Build the ReduceLROnPlateau scheduler."""
        super().build(optimizer)
        return MultiStepLR(optimizer, **self.get_kwargs())


@dataclass
class CosineAnnealingScheduler(SchedulerFactory):
    """Cosine annealing learning rate scheduler."""

    T_max: int
    eta_min: float | None = None
    eta_min_factor: float | None = None

    def build(self, optimizer: Optimizer) -> LRScheduler:
        """Build the CosineAnnealingLR scheduler."""
        super().build(optimizer)
        kwargs = self.get_kwargs()
        if self.eta_min is not None and self.eta_min_factor is not None:
            raise ValueError("Cannot specify both eta_min and eta_min_factor.")
        if self.eta_min_factor is not None:
            # Derive eta_min as a factor of the optimizer's base learning rate.
            base_lr = optimizer.defaults["lr"]
            kwargs["eta_min"] = base_lr * self.eta_min_factor
            kwargs.pop("eta_min_factor", None)
            logger.info(
                f"CosineAnnealingScheduler: eta_min set to {kwargs['eta_min']} "
                f"(base_lr={base_lr} * factor={self.eta_min_factor})"
            )
        else:
            kwargs.pop("eta_min_factor", None)
        return CosineAnnealingLR(optimizer, **kwargs)


@dataclass
class CosineAnnealingWarmRestartsScheduler(SchedulerFactory):
    """Cosine annealing with warm restarts learning rate scheduler."""

    T_0: int
    T_mult: int = 1
    eta_min: float = 0.0

    def build(self, optimizer: Optimizer) -> LRScheduler:
        """Build the CosineAnnealingWarmRestarts scheduler."""
        super().build(optimizer)
        return CosineAnnealingWarmRestarts(optimizer, **self.get_kwargs())
