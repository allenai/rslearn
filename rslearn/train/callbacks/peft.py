"""Parameter-efficient finetuning callbacks."""

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import BaseFinetuning
from torch.optim.optimizer import Optimizer
from helios.nn.attention import Attention


class SplitProjection(torch.nn.Module):

    def __init__(self, dim, r=8):
        super().__init__()
        self.dim = dim
        self.r = r

        # Register indices as buffers so they move to the correct device automatically
        indices = torch.randperm(dim)
        self.register_buffer('trainable_inds', indices[:r])
        self.register_buffer('frozen_inds', indices[r:])

        # Create parameter modules directly
        self.trainable_w = torch.nn.Parameter(torch.empty(dim, r), requires_grad=True)
        self.frozen_w = torch.nn.Parameter(torch.empty(dim, dim - r), requires_grad=False)
        self.trainable_b = torch.nn.Parameter(torch.empty(r), requires_grad=True)
        self.frozen_b = torch.nn.Parameter(torch.empty(dim - r), requires_grad=False)

    def forward(self, x):
        trainable_out = F.linear(x, self.trainable_w, self.trainable_b)
        frozen_out = F.linear(x, self.frozen_w, self.frozen_b)
        
        output = torch.zeros(x.shape, device=x.device, dtype=trainable_out.dtype)
        output[..., self.trainable_inds] = trainable_out
        output[..., self.frozen_inds] = frozen_out
        
        return output


class APLA(BaseFinetuning):
    """APLA (https://arxiv.org/pdf/2503.11335v2) finetuning callback."""

    def __init__(self, r: int = 8) -> None:
        super().__init__()
        self.r = r

    def freeze_before_training(self, pl_module: LightningModule) -> None:
        print(f"splitting projection weights by monkeypatching")
        model = pl_module.model
        self.freeze(model.encoder[0])
        n_trainable = 0
        for layer in model.encoder[0].model.blocks:
            if hasattr(layer, 'attn'):
                alpa_proj = SplitProjection(layer.attn.proj.weight.shape[0], r=self.r)
                proj_weight = layer.attn.proj.weight.data.clone()
                proj_bias = layer.attn.proj.bias.data.clone()

                alpa_proj.trainable_w.data = proj_weight[alpa_proj.trainable_inds, :]
                alpa_proj.frozen_w.data = proj_weight[alpa_proj.frozen_inds, :]

                alpa_proj.trainable_b.data = proj_bias[alpa_proj.trainable_inds]
                alpa_proj.frozen_b.data = proj_bias[alpa_proj.frozen_inds]

                alpa_proj.trainable_w.requires_grad = True
                alpa_proj.trainable_b.requires_grad = True
                n_trainable += alpa_proj.trainable_w.numel() + alpa_proj.trainable_b.numel()

                layer.attn.proj = alpa_proj

        print(f"n_trainable: {n_trainable / int(1e6)}M")

    def finetune_function(
        self, pl_module: LightningModule, current_epoch: int, optimizer: Optimizer
    ) -> None:
        # Maybe worth unfreezing down the line?
        pass