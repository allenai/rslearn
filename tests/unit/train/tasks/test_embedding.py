import torch

from rslearn.models.component import TokenFeatureMaps
from rslearn.train.model_context import ModelContext
from rslearn.train.tasks.embedding import EmbeddingHead


def test_embedding_head_token_feature_maps() -> None:
    head = EmbeddingHead()
    B, C, H, W, N = 2, 4, 8, 8, 3
    tensor = torch.randn(B, C, H, W, N)
    intermediates = TokenFeatureMaps([tensor])
    context = ModelContext(inputs=[], metadatas=[])
    output = head(intermediates=intermediates, context=context, targets=None)
    # Token dim merged into channels: (N*C) = 12
    assert output.outputs.shape == (B, N * C, H, W)
    assert output.loss_dict["loss"] == 0

    # Verify reshape order: output should have N groups of C channels,
    # where group i corresponds to token i.
    for n in range(N):
        expected = tensor[:, :, :, :, n]  # (B, C, H, W)
        actual = output.outputs[:, n * C : (n + 1) * C, :, :]
        assert torch.equal(actual, expected), f"token group {n} mismatch"
