"""Parameter-efficient finetuning callbacks."""

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import BaseFinetuning
from torch.optim.optimizer import Optimizer


class SplitProjection(torch.nn.Module):
    """Split projection weights into trainable and frozen parts.

    This module is used to split the projection weights into trainable and frozen parts.
    The trainable part is used to compute the output, and the frozen part is used to
    compute the output without gradients.
    """

    def __init__(self, dim: int, r: int = 8) -> None:
        """Initialize the SplitProjection module.

        Args:
            dim: the dimension of the input and output
            r: the number of trainable parameters
        """
        super().__init__()
        self.dim = dim
        self.r = r

        # Register indices as buffers so they move to the correct device automatically
        indices = torch.randperm(dim)
        self.register_buffer("trainable_inds", indices[:r])
        self.register_buffer("frozen_inds", indices[r:])

        # Create parameter modules directly
        self.trainable_w = torch.nn.Parameter(torch.empty(dim, r), requires_grad=True)
        self.frozen_w = torch.nn.Parameter(
            torch.empty(dim, dim - r), requires_grad=False
        )
        self.trainable_b = torch.nn.Parameter(torch.empty(r), requires_grad=True)
        self.frozen_b = torch.nn.Parameter(torch.empty(dim - r), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SplitProjection module.

        Args:
            x: the input tensor

        Returns:
            the output tensor
        """
        trainable_out = F.linear(x, self.trainable_w, self.trainable_b)
        frozen_out = F.linear(x, self.frozen_w, self.frozen_b)

        output = torch.zeros(x.shape, device=x.device, dtype=trainable_out.dtype)
        output[..., self.trainable_inds] = trainable_out  # type: ignore
        output[..., self.frozen_inds] = frozen_out  # type: ignore

        return output


class APLA(BaseFinetuning):
    """APLA (https://arxiv.org/pdf/2503.11335v2) finetuning callback."""

    def __init__(self, r: int = 8) -> None:
        """Initialize the APLA finetuning callback.

        Args:
            r: the number of trainable parameters
        """
        super().__init__()
        self.r = r

    def freeze_before_training(self, pl_module: LightningModule) -> None:
        """Freeze the model before training.

        Args:
            pl_module: the LightningModule
        """
        print("splitting projection weights by monkeypatching")
        model = pl_module.model
        self.freeze(model.encoder[0])
        n_trainable = 0
        for layer in model.encoder[0].model.blocks:
            if hasattr(layer, "attn"):
                alpa_proj = SplitProjection(layer.attn.proj.weight.shape[0], r=self.r)
                proj_weight = layer.attn.proj.weight.data.clone()
                proj_bias = layer.attn.proj.bias.data.clone()

                alpa_proj.trainable_w.data = proj_weight[alpa_proj.trainable_inds, :]
                alpa_proj.frozen_w.data = proj_weight[alpa_proj.frozen_inds, :]

                alpa_proj.trainable_b.data = proj_bias[alpa_proj.trainable_inds]
                alpa_proj.frozen_b.data = proj_bias[alpa_proj.frozen_inds]

                alpa_proj.trainable_w.requires_grad = True
                alpa_proj.trainable_b.requires_grad = True
                n_trainable += (
                    alpa_proj.trainable_w.numel() + alpa_proj.trainable_b.numel()
                )

                layer.attn.proj = alpa_proj

        print(f"n_trainable: {n_trainable / int(1e6)}M")

    def finetune_function(
        self, pl_module: LightningModule, current_epoch: int, optimizer: Optimizer
    ) -> None:
        """Do nothing here.

        Args:
            pl_module: the LightningModule
            current_epoch: the current epoch
            optimizer: the optimizer
        """
        # Maybe worth unfreezing down the line?
        pass
