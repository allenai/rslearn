import numpy as np
import pytest
import torch

from rslearn.models.component import FeatureMaps, FeatureVector
from rslearn.train.model_context import ModelContext, SampleMetadata
from rslearn.train.tasks.embedding import EmbeddingHead, EmbeddingTask


def test_embedding_task_process_output_raster(
    empty_sample_metadata: SampleMetadata,
) -> None:
    task = EmbeddingTask()
    output = torch.ones((3, 4, 5), dtype=torch.float32)

    processed = task.process_output(output, empty_sample_metadata)

    assert isinstance(processed, np.ndarray)
    assert processed.shape == (3, 4, 5)


def test_embedding_task_process_output_vector(
    empty_sample_metadata: SampleMetadata,
) -> None:
    task = EmbeddingTask(property_name="embedding")
    output = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

    processed = task.process_output(output, empty_sample_metadata)

    assert len(processed) == 1
    assert processed[0].properties is not None
    assert processed[0].properties["embedding"] == [1.0, 2.0, 3.0]


def test_embedding_task_invalid_output_rank(
    empty_sample_metadata: SampleMetadata,
) -> None:
    task = EmbeddingTask()

    with pytest.raises(
        ValueError,
        match="output for EmbeddingTask must be a tensor with one or three dimensions",
    ):
        task.process_output(
            torch.zeros((2, 3), dtype=torch.float32), empty_sample_metadata
        )


def test_embedding_head_accepts_feature_maps() -> None:
    head = EmbeddingHead()
    intermediates = FeatureMaps([torch.ones((2, 3, 4, 5), dtype=torch.float32)])

    output = head(intermediates, ModelContext(inputs=[], metadatas=[]))

    assert output.outputs.shape == (2, 3, 4, 5)


def test_embedding_head_accepts_feature_vector() -> None:
    head = EmbeddingHead()
    intermediates = FeatureVector(
        feature_vector=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    )

    output = head(intermediates, ModelContext(inputs=[], metadatas=[]))

    assert output.outputs.shape == (2, 2)
