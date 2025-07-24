"""Task embedding modules."""

from typing import Any

import torch


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
        idx = [self.target_to_embed_idx[inp["dataset_source"]] for inp in inputs]
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

    (With the task embedding as the query and the features as the key and value.)
    """

    def __init__(
        self, encoder_embedding_size: int, num_heads: int, height: int, width: int
    ) -> None:
        """Initialize the task MHA embedding module.

        Args:
            encoder_embedding_size: The size of the encoder embedding.
            num_heads: The number of attention heads.
            height: height of encoder embeds
            width: width of encoder embeds
        """
        super().__init__(encoder_embedding_size)
        self.embed_proj = torch.nn.Linear(1, height * width)
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
        x = torch.flatten(features[0], start_dim=2)  # B x C x HW
        embeds = self._compute_embeds(inputs, x.device).view(-1, 1)  # BC x 1
        embeds = self.embed_proj(embeds).view(*x.shape)  # B x C x HW
        out = self.mha(
            torch.einsum("bct->btc", embeds),
            torch.einsum("bct->btc", x),
            torch.einsum("bct->btc", x),
        )[0]  # B x T x C
        out = torch.einsum("btc->bct", out)
        out = out.view(*features[0].shape)  # B x C x H x W
        return [out]
