"""Trunk module for decoder."""

from typing import Any

import torch

from rslearn.models.moe.soft import SoftMoE
from rslearn.models.task_embedding import BaseTaskEmbedding


class TrunkTransformer(torch.nn.Module):
    """Transformer for decoder trunk."""

    def __init__(
        self,
        dim: int,
        n_layers: int,
        n_heads: int,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        use_moe: bool = False,
        task_moe: bool = False,
        num_experts: int = 16,
        num_slots: int = 256,
    ):
        """Standard ViT-style transformer, with optional soft MoE.

        Since the point of the MoE layers is to deal with task-specific and task-shared
        features (and not to route specific tokens), it's probably best to use max_seq_len
        as the number of slots, and have at least one expert per task (probably more).

        Args:
            dim: dimension of the input and output
            n_layers: number of transformer blocks
            n_heads: number of attention heads
            mlp_dim: dimension of the MLP
            dropout: dropout rate
            use_moe: whether to use soft MoE
            task_moe: if specified, compute dispatch weights given the task embedding
                only, and not the token
            num_experts: number of experts in soft MoE
            num_slots: number of slots in soft MoE
        """
        super().__init__()
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_slots = num_slots
        self.task_moe = task_moe
        self.norm = torch.nn.LayerNorm(dim)
        self.layers = torch.nn.ModuleList([])
        for _ in range(n_layers):
            mha = torch.nn.MultiheadAttention(
                dim, n_heads, dropout=dropout, batch_first=True
            )
            if use_moe:
                ffn = SoftMoE(
                    dim=dim,
                    num_experts=num_experts,
                    num_slots=num_slots,
                    dropout=dropout,
                )
            else:
                ffn = torch.nn.Sequential(
                    torch.nn.LayerNorm(dim),
                    torch.nn.Linear(dim, mlp_dim),
                    torch.nn.GELU(),
                    torch.nn.Linear(mlp_dim, dim),
                )
            drop = torch.nn.Dropout(dropout)
            self.layers.append(torch.nn.ModuleList([mha, ffn, drop]))

    def forward(
        self,
        x: torch.Tensor,
        task_embedding: torch.Tensor,
        return_dispatch_weights: bool = False,
        return_combine_weights: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Forward pass.

        Args:
            x: input tensor of shape (batch_size, seq_len, dim)
            task_embedding: task embedding tensor of shape (batch_size, dim)
            return_dispatch_weights: whether to return the dispatch weights
            return_combine_weights: whether to return the combine weights

        Returns:
            (
                output tensor of shape (batch_size, seq_len, dim),
                list of dispatch weights of shape (num_experts,)
                list of combine weights of shape (num_experts,)
            )
        """
        dispatch_weights = []
        combine_weights = []
        kwargs = {}
        if return_dispatch_weights:
            kwargs["return_dispatch_weights"] = True
        if return_combine_weights:
            kwargs["return_combine_weights"] = True
        if self.task_moe:
            kwargs["weight_key"] = task_embedding

        for mha, ffn, drop in self.layers:
            x = mha(x, x, x)[0] + x
            if kwargs:
                x_ffn, dw, cw = ffn(x, **kwargs)
                dispatch_weights.append(dw)
                combine_weights.append(cw)
            else:
                x_ffn = ffn(x)
            x = drop(x_ffn + x)
        x = self.norm(x)

        if return_dispatch_weights:
            for i, weight in enumerate(dispatch_weights):
                # each weight is [batch, seq_len, num_experts, num_slots]
                # compute the average weight per token across slot/batch/seq len
                # NOTE: this is probably about the same across all tokens, assuming all tokens
                # get looked at by at least a couple experts
                dispatch_weights[i] = weight.mean(dim=(0, 2, 3))
        else:
            dispatch_weights = []

        if return_combine_weights:
            for i, weight in enumerate(combine_weights):
                # each weight is [batch, seq_len, num_experts * num_slots]
                # compute the average weight per expert (slot group) across batch/seq
                weight = weight.unflatten(
                    dim=-1, sizes=(self.num_experts, self.num_slots)
                )
                weight = weight.sum(
                    dim=-1
                )  # [batch, seq_len, num_experts], last dim softmaxed
                combine_weights[i] = weight.mean(dim=(0, 1))
        else:
            combine_weights = []

        return x, dispatch_weights, combine_weights


class DecoderTrunk(torch.nn.Module):
    """Trunk module for decoder.

    A stack of ViT blocks, with optional task-specific embeddings and
    optional soft Mixture of Experts within the transformer.
    """

    def __init__(
        self,
        task_embedding: BaseTaskEmbedding,
        **kwargs: Any,
    ):
        """Initialize the DecoderTrunk module.

        Args:
            task_embedding: Task-specific embedding module.
            **kwargs: Additional arguments passed to TrunkTransformer.
        """
        super().__init__()
        self.task_embedding = task_embedding
        self.transformer = TrunkTransformer(**kwargs)

    def register_tasks(self, task_names: list[str]) -> None:
        """Register tasks.

        Args:
            task_names: list of task names
        """
        self.task_embedding.register_tasks(task_names)

    def forward(
        self,
        features: list[torch.tensor],
        inputs: list[dict[str, Any]],
        return_dispatch_weights: bool = False,
        return_combine_weights: bool = False,
    ) -> tuple[list[torch.tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Forward pass.

        Args:
            features: The encoder features, a 1-list of B x C x H x W features.
            inputs: The original inputs to the encoder.
            return_dispatch_weights: whether to return the dispatch weights
            return_combine_weights: whether to return the combine weights

        Returns:
            (
                The encoder features with the trunk applied. Same shape as features.
                optional list of dispatch weights, each of shape (num_experts,)
                optional list of combine weights, each of shape (num_experts,)
            )
        """
        embeds = self.task_embedding.compute_embeds(features, inputs)
        features = self.task_embedding(features, inputs, embeds=embeds)
        x = torch.einsum("bchw->bhwc", features[0])
        x = torch.flatten(x, start_dim=1, end_dim=2)  # B x T x C, T = HW
        out, dispatch_weights, combine_weights = self.transformer(
            x,
            task_embedding=embeds,
            return_dispatch_weights=return_dispatch_weights,
            return_combine_weights=return_combine_weights,
        )  # B x T x C
        out = torch.einsum("btc->bct", out)  # B x C x T
        out = out.view(*features[0].shape)  # B x C x H x W
        return [out], dispatch_weights, combine_weights
