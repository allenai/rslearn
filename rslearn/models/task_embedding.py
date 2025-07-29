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


class TaskChannelEmbedding(BaseTaskEmbedding):
    """Registers task-specific 'tokens', i.e. embeddings.

    Each embedding is learned per-channel and copied over the full spatial dimensions.
    """

    def __init__(self, encoder_embedding_size: int, default_idx: int = 0) -> None:
        """Initialize the task channel embedding module.

        Args:
            encoder_embedding_size: The size of the encoder embedding.
            default_idx: The index of the default task, useful if loading a merged model.
        """
        super().__init__(encoder_embedding_size)
        self.default_idx = default_idx

    def register_tasks(self, task_names: list[str]) -> None:
        """Register the tasks.

        This must happen post-init so that we can dynamically determine
        the tasks to use, so it doesn't have to be specified in the config.

        Args:
            task_names: The names of the tasks.
        """
        self.embed = torch.nn.Embedding(len(task_names), self.encoder_embedding_size)
        self.target_to_embed_idx = {name: i for i, name in enumerate(task_names)}

    def _compute_embeds(
        self, inputs: list[dict[str, Any]], device: torch.device
    ) -> torch.Tensor:
        """Compute the task-specific embeddings.

        Args:
            inputs: The inputs to the model.
            device: The device to compute the embeddings on.
        """
        try:
            idx = [self.target_to_embed_idx[inp["dataset_source"]] for inp in inputs]
        except KeyError:
            idx = [self.default_idx] * len(inputs)
        return self.embed(torch.tensor(idx).to(device))

    def forward(
        self, features: list[torch.tensor], inputs: list[dict[str, Any]]
    ) -> list[torch.tensor]:
        """Compute and apply task-specific embeddings to encoder features.

        Args:
            features: The encoder features, a 1-list of B x C x H x W features.
            inputs: The inputs to the model.

        Returns:
            The encoder features with the task-specific embeddings added.
        """
        # Add per-dataset, per-channel task embedding (B x C)
        x = features[0]
        embeds = self._compute_embeds(inputs, x.device)
        x += embeds.unsqueeze(-1).unsqueeze(-1)  # B x C x 1 x 1
        return [x]


class TaskMHAEmbedding(TaskChannelEmbedding):
    """Multi-headed cross-attention over the spatial dimensions.

    The task embedding is the query and the features are the key and value.
    We copy the task embedding over the spatial dimensions, and add a sinusoidal
    positional embedding before the MHA layer.
    """

    def __init__(
        self, encoder_embedding_size: int, num_heads: int, default_idx: int = 0
    ) -> None:
        """Initialize the task MHA embedding module.

        Args:
            encoder_embedding_size: The size of the encoder embedding.
            num_heads: The number of attention heads.
            default_idx: The index of the default task, useful if loading a merged model.
        """
        super().__init__(encoder_embedding_size, default_idx)
        self.pos_embed = PositionalEncoding(encoder_embedding_size)
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
        self, features: list[torch.tensor], inputs: list[dict[str, Any]]
    ) -> list[torch.tensor]:
        """Compute and apply task-specific embeddings to encoder features.

        Also apply the MHA layer across the spatial dimension, with the task embedding
        as the query and the features as the key and value.

        Args:
            features: The encoder features, a 1-list of B x C x H x W features.
            inputs: The inputs to the model.

        Returns:
            The encoder features with the task-specific embeddings added.
        """
        x = torch.flatten(features[0], start_dim=2)  # B x C x T, T = HW
        embeds = self._compute_embeds(inputs, x.device)  # B x C
        embeds = embeds.unsqueeze(0).repeat(x.shape[-1], 1, 1)  # T x B x C
        embeds = self.pos_embed(embeds)  # T x B x C
        out = self.mha(
            torch.einsum("tbc->btc", embeds),
            torch.einsum("bct->btc", x),
            torch.einsum("bct->btc", x),
        )[0]  # B x T x C
        out = torch.einsum("btc->bct", out)
        out = out.view(*features[0].shape)  # B x C x H x W
        return [out]
