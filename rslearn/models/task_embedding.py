"""Task embedding modules."""

import math
from typing import Any

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Simple sinusoidal positional encoding for the task embedding. From torch docs."""

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 1024):
        """Initialize the positional encoding module.

        Args:
            d_model: The dimension of the model.
            dropout: The dropout rate.
            max_len: The maximum length of the sequence.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding to the input tensor.

        Args:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class BaseTaskEmbedding(torch.nn.Module):
    """Base class for task embedding modules."""

    def __init__(self, encoder_embedding_size: int) -> None:
        """Initialize the base task embedding module.

        Args:
            encoder_embedding_size: The size of the encoder embedding.
        """
        super().__init__()
        self.encoder_embedding_size = encoder_embedding_size

    def register_tasks(self, task_names: list[str]) -> None:
        """Register the tasks.

        This must happen post-init so that we can dynamically determine
        the tasks to use, so it doesn't have to be specified in the config.

        Args:
            task_names: The names of the tasks.
        """
        raise NotImplementedError

    def compute_embeds(
        self,
        features: list[torch.tensor],
        inputs: list[dict[str, Any]],
    ) -> torch.Tensor:
        """Compute the task-specific embeddings.

        Args:
            features: The encoder features.
            inputs: The inputs to the model.

        Returns:
            The task-specific embeddings.
        """
        raise NotImplementedError


class TaskChannelEmbedding(BaseTaskEmbedding):
    """Registers task-specific 'tokens', i.e. embeddings.

    Each embedding is learned per-channel and copied over the full spatial dimensions.
    Optionally, add a spatial sinusoidal positional embedding to the task embedding.
    """

    def __init__(
        self,
        encoder_embedding_size: int,
        default_idx: int = 0,
        add_spatial_embed: bool = False,
    ) -> None:
        """Initialize the task channel embedding module.

        Args:
            encoder_embedding_size: The size of the encoder embedding.
            default_idx: The index of the default task, useful if loading a merged model.
            add_spatial_embed: if true, add a spatial sinusoidal positional embedding to the task embedding
        """
        super().__init__(encoder_embedding_size)
        self.default_idx = default_idx
        self.add_spatial_embed = add_spatial_embed
        if add_spatial_embed:
            self.pos_embed = PositionalEncoding(encoder_embedding_size)

    def register_tasks(self, task_names: list[str]) -> None:
        """Register the tasks.

        This must happen post-init so that we can dynamically determine
        the tasks to use, so it doesn't have to be specified in the config.

        Args:
            task_names: The names of the tasks.
        """
        self.embed = torch.nn.Embedding(len(task_names), self.encoder_embedding_size)
        self.target_to_embed_idx = {name: i for i, name in enumerate(task_names)}

    def compute_embeds(
        self,
        features: list[torch.tensor],
        inputs: list[dict[str, Any]],
    ) -> torch.Tensor:
        """Compute the task-specific embeddings.

        Args:
            inputs: The inputs to the model.
            features: computed encoder features
            device: The device to compute the embeddings on.

        Returns:
            The task-specific embeddings, shape (B, T, C), T = HW
            The embeddings are repeated over the spatial dimensions, and optionally
            a sinusoidal positional embedding is added.
        """
        try:
            idx = [self.target_to_embed_idx[inp["dataset_source"]] for inp in inputs]
        except KeyError:
            idx = [self.default_idx] * len(inputs)
        embeds = self.embed(torch.tensor(idx).to(features[0].device))
        seq_len = features[0].shape[-1] * features[0].shape[-2]  # T = HW
        embeds = embeds.unsqueeze(0).repeat(seq_len, 1, 1)  # T x B x C
        if self.add_spatial_embed:
            embeds = self.pos_embed(embeds)
        embeds = torch.einsum("tbc->btc", embeds)  # B x T x C
        return embeds

    def forward(
        self,
        features: list[torch.tensor],
        inputs: list[dict[str, Any]],
        embeds: torch.Tensor | None = None,
    ) -> list[torch.tensor]:
        """Compute and apply task-specific embeddings to encoder features.

        Optionally, add a spatial sinusoidal positional embedding to the task embedding.
        Otherwise, the task embedding is repeated over the spatial dimensions.

        Args:
            features: The encoder features, a 1-list of B x C x H x W features.
            inputs: The inputs to the model.
            embeds: Already-computed task embeddings, if provided, skip the computation.

        Returns:
            The encoder features with the task-specific embeddings added.
        """
        height, width = features[0].shape[-2:]
        if embeds is None:
            embeds = self.compute_embeds(features, inputs)  # B x HW x C
        embeds = embeds.unflatten(dim=1, sizes=(height, width))  # B x H x W x C
        features[0] += torch.einsum("bhwc->bchw", embeds)  # B x C x H x W
        return features


class TaskMHAEmbedding(TaskChannelEmbedding):
    """Multi-headed cross-attention over the spatial dimensions.

    The task embedding is the query and the features are the key and value.
    We copy the task embedding over the spatial dimensions, and optionally
    add a sinusoidal positional embedding before the MHA layer.
    """

    def __init__(
        self,
        encoder_embedding_size: int,
        num_heads: int,
        default_idx: int = 0,
        add_spatial_embed: bool = True,
    ) -> None:
        """Initialize the task MHA embedding module.

        Args:
            encoder_embedding_size: The size of the encoder embedding.
            num_heads: The number of attention heads.
            default_idx: The index of the default task, useful if loading a merged model.
            add_spatial_embed: if true, add a spatial sinusoidal positional embedding to the task embedding
        """
        super().__init__(encoder_embedding_size, default_idx, add_spatial_embed)
        self.mha = torch.nn.MultiheadAttention(
            encoder_embedding_size, num_heads, batch_first=True
        )

    def register_tasks(self, task_names: list[str]) -> None:
        """Register the tasks.

        This must happen post-init so that we can dynamically determine
        the tasks to use, so it doesn't have to be specified in the config.

        Args:
            task_names: The names of the tasks.
        """
        super().register_tasks(task_names)

    def forward(
        self,
        features: list[torch.tensor],
        inputs: list[dict[str, Any]],
        embeds: torch.Tensor | None = None,
    ) -> list[torch.tensor]:
        """Compute and apply task-specific embeddings to encoder features.

        Also apply the MHA layer across the spatial dimension, with the task embedding
        as the query and the features as the key and value.

        Args:
            features: The encoder features, a 1-list of B x C x H x W features.
            inputs: The inputs to the model.
            embeds: Already-computed task embeddings, if provided, skip the computation.

        Returns:
            The encoder features with the task-specific embeddings added.
        """
        x = torch.flatten(features[0], start_dim=2)  # B x C x T, T = HW
        if embeds is None:
            embeds = self.compute_embeds(features, inputs)  # B x T x C
        out = self.mha(
            embeds,  # B x T x C
            torch.einsum("bct->btc", x),
            torch.einsum("bct->btc", x),
        )[0]  # B x T x C
        out = torch.einsum("btc->bct", out)
        out = out.view(*features[0].shape)  # B x C x H x W
        return [out]
