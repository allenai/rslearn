"""Gradient logging and surgery callbacks."""

from typing import Any

import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer import Trainer
from torch.nn import Module
from torch.optim import Optimizer


class MiniPCGrad(Callback):
    """PCGrad from https://arxiv.org/abs/2001.06782.

    This is roughly equivalent to PCGrad but uses gradient accumulation to factorize
    projections, so we can keep gradients orthogonal in O(1) memory instead of O(n).
    This is still quite slow, requiring an extra copy of parameter gradients in memory.
    """

    def __init__(
        self,
        selectors: list[str],
        deselectors: list[str] | None = None,
        only_monitor: bool = False,
    ) -> None:
        """Initialize the callback.

        Args:
            selectors: Prefixes for selecting which parameters to operate on.
            deselectors: Prefixes for deselecting which parameters to operate on. Applied after selectors.
            only_monitor: If true, only log gradients, don't clip them.
        """
        self.selectors = selectors
        self.deselectors = deselectors or []
        self.only_monitor = only_monitor
        self.prev_grads: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: Module, batch: Any, batch_idx: int
    ) -> None:
        """Save the dataset source each batch."""
        self.dataset_source = batch[0][0]["dataset_source"]
        self.batch_size = len(batch[0])

    def on_before_optimizer_step(
        self, trainer: Trainer, pl_module: Module, optimizer: Optimizer
    ) -> None:
        """Reset the previous gradients."""
        self.prev_grads = {}

    def on_after_backward(self, trainer: Trainer, pl_module: Module) -> None:
        """Called after every loss.backward(), even under gradient accumulation.

        Receives the accumulated gradients (i.e., accumulated + micro batch gradient).

        Args:
            trainer: The trainer object.
            pl_module: The module object.
        """
        prev_grad_norms = []
        micro_grad_norms = []
        angles = []

        eps = 1e-12  # numerical stability

        for name, param in pl_module.named_parameters():
            if param.grad is None:
                continue
            elif all(selector not in name for selector in self.selectors) or any(
                deselector in name for deselector in self.deselectors
            ):
                continue

            try:
                prev_grad, prev_grad_norm = self.prev_grads[name]
            except KeyError:
                prev_grad = torch.zeros_like(param.grad, device=param.device)
                prev_grad_norm = torch.tensor(0.0, device=param.device)

            with torch.no_grad():
                # current accumulated grad = prev_grad + micro_grad
                micro_grad = param.grad - prev_grad
                micro_grad_norm = micro_grad.norm()

                micro_grad_norms.append(micro_grad_norm)
                prev_grad_norms.append(prev_grad_norm)

                # cosine of angle between micro and prev
                denom = (micro_grad_norm * prev_grad_norm).clamp_min(eps)
                if prev_grad_norm > 0 and micro_grad_norm > 0:
                    dot = torch.dot(micro_grad.flatten(), prev_grad.flatten())
                    cos_theta = dot / denom
                    angles.append(cos_theta)

                    if not self.only_monitor and dot < 0:
                        # Remove the component of micro_grad along prev_grad
                        proj_coeff = dot / (prev_grad_norm**2 + eps)
                        micro_projection = micro_grad - proj_coeff * prev_grad
                        # keep accumulated gradient as (prev + projected micro)
                        param.grad = prev_grad + micro_projection

                # store the latest accumulated gradient and its norm
                self.prev_grads[name] = (param.grad.clone(), param.grad.norm())

        log_prev_grad_norms = (
            torch.stack(prev_grad_norms).norm()
            if prev_grad_norms
            else torch.tensor(0.0)
        )
        log_micro_grad_norms = (
            torch.stack(micro_grad_norms).norm()
            if micro_grad_norms
            else torch.tensor(0.0)
        )
        log_angles = torch.stack(angles).mean() if angles else torch.tensor(0.0)

        info = {
            f"grads/{self.dataset_source}_prev_grad_norms": log_prev_grad_norms,
            f"grads/{self.dataset_source}_micro_grad_norms": log_micro_grad_norms,
            f"grads/{self.dataset_source}_angles": log_angles,
        }
        self.log_dict(info, on_step=True, on_epoch=False, batch_size=self.batch_size)
