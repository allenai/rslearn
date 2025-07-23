"""Gradient logging and surgery callbacks."""

from typing import Any

import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer import Trainer
from torch.nn import Module
from torch.optim import Optimizer


class MiniPCGrad(Callback):
    """PCGrad from https://arxiv.org/abs/2001.06782.

    This should be equivalent to PCGrad, but uses gradient accumulation to factorize
    projections, so we can keep gradients orthogonal in O(1) memory instead of O(n).

    Still quite slow, requiring an extra copy of parameter gradients in memory.
    """

    def __init__(self, selector: str, only_monitor: bool = False) -> None:
        """Initialize the callback.

        Args:
            selector: Prefix for selecting which parameters to operate on.
            only_monitor: If true, only log gradients, don't clip them.
        """
        self.selector = selector
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
        for name, param in pl_module.named_parameters():
            if param.grad is None or self.selector not in name:
                continue

            try:
                prev_grad, prev_grad_norm = self.prev_grads[name]
            except KeyError:
                prev_grad = torch.zeros_like(param.grad).to(param.device)
                prev_grad_norm = torch.tensor(0.0).to(param.device)

            with torch.no_grad():
                micro_grad = param.grad - prev_grad
                micro_grad_norm = micro_grad.norm()

                micro_grad_norms.append(micro_grad_norm)
                prev_grad_norms.append(prev_grad_norm)

                norm_prod = micro_grad_norm * prev_grad_norm
                if norm_prod != 0:
                    angle = (
                        torch.dot(micro_grad.flatten(), prev_grad.flatten()) / norm_prod
                    )
                    angles.append(angle)

                    if not self.only_monitor and angle < 0:
                        # Project the micro grad onto the prev grad's normal plane, and then vice versea
                        micro_projection = (
                            micro_grad - norm_prod / (prev_grad_norm**2) * prev_grad
                        )
                        prev_projection = (
                            prev_grad - norm_prod / (micro_grad_norm**2) * micro_grad
                        )

                        # Since gradient accumulation does not divide by the batch size until
                        # the optimizer step, we can just sum the projected gradients here
                        param.grad = micro_projection + prev_projection

                self.prev_grads[name] = (param.grad.clone(), param.grad.norm())

        info = {
            f"grads/{self.dataset_source}_prev_grad_norms": torch.stack(
                prev_grad_norms
            ).norm(),
            f"grads/{self.dataset_source}_micro_grad_norms": torch.stack(
                micro_grad_norms
            ).norm(),
        }
        if len(angles) > 0:
            info[f"grads/{self.dataset_source}_angles"] = torch.stack(angles).mean()
        self.log_dict(info, on_step=True, on_epoch=False, batch_size=self.batch_size)
