"""Tests for the rslearn.models.embedding_cache module."""

from datetime import datetime

import torch
from pyproj import CRS

from rslearn.models.component import FeatureExtractor, FeatureMaps
from rslearn.models.embedding_cache import EmbeddingCache
from rslearn.train.model_context import ModelContext, RasterImage, SampleMetadata
from rslearn.utils.geometry import Projection


def _make_metadata(
    window_name: str = "win0",
    crop_bounds: tuple[int, int, int, int] = (0, 0, 8, 8),
) -> SampleMetadata:
    return SampleMetadata(
        window_group="group",
        window_name=window_name,
        window_bounds=(0, 0, 128, 128),
        crop_bounds=crop_bounds,
        crop_idx=0,
        num_crops_in_window=1,
        time_range=(datetime(2024, 1, 1), datetime(2024, 1, 2)),
        projection=Projection(CRS.from_epsg(4326), 1.0, 1.0),
        dataset_source=None,
    )


class DummyEncoder(FeatureExtractor):
    """Deterministic encoder that copies the first timestep of each RasterImage input."""

    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    def forward(self, context: ModelContext) -> FeatureMaps:
        self.call_count += 1
        tensors = []
        for inp_dict in context.inputs:
            img = inp_dict["image"]
            assert isinstance(img, RasterImage)
            # Take first timestep: CTHW -> CHW.
            tensors.append(img.image[:, 0, :, :])
        return FeatureMaps(feature_maps=[torch.stack(tensors, dim=0)])


def _make_context(
    metadatas: list[SampleMetadata],
    values: list[float] | None = None,
    device: torch.device = torch.device("cpu"),
) -> ModelContext:
    """Build a ModelContext with a constant-valued image per metadata entry.

    Args:
        metadatas: one SampleMetadata per example.
        values: per-example scalar to fill each image with. If None, uses 0, 1, 2, ...
        device: torch device.
    """
    if values is None:
        values = [float(i) for i in range(len(metadatas))]
    inputs: list[dict[str, torch.Tensor | RasterImage]] = []
    for val in values:
        img = torch.full((3, 1, 8, 8), val, device=device)
        inputs.append({"image": RasterImage(image=img)})
    return ModelContext(inputs=inputs, metadatas=metadatas)


def test_forward_basic() -> None:
    """Compute embeddings for a single example."""
    encoder = DummyEncoder()
    cache = EmbeddingCache(encoder=[encoder])
    meta = _make_metadata()
    ctx = _make_context([meta], values=[5.0])

    result = cache(ctx)

    assert isinstance(result, FeatureMaps)
    assert len(result.feature_maps) == 1
    assert result.feature_maps[0].shape == (1, 3, 8, 8)
    assert (result.feature_maps[0] == 5.0).all()
    assert encoder.call_count == 1


def test_cache_hit_skips_encoder() -> None:
    """Second call with the same key should use the cache, not the encoder."""
    encoder = DummyEncoder()
    cache = EmbeddingCache(encoder=[encoder])
    meta = _make_metadata()

    ctx1 = _make_context([meta], values=[3.0])
    result1 = cache(ctx1)

    ctx2 = _make_context([meta], values=[7.0])
    result2 = cache(ctx2)

    # Encoder should only have been called once
    assert encoder.call_count == 1
    # Both results should reflect the first (cached) value, not the second
    assert (result1.feature_maps[0] == 3.0).all()
    assert (result2.feature_maps[0] == 3.0).all()


def test_different_keys_miss_cache() -> None:
    """Different window names should be separate cache entries."""
    encoder = DummyEncoder()
    cache = EmbeddingCache(encoder=[encoder])

    meta_a = _make_metadata(window_name="win_a")
    meta_b = _make_metadata(window_name="win_b")

    cache(_make_context([meta_a]))
    cache(_make_context([meta_b]))

    assert encoder.call_count == 2


def test_different_crop_bounds_miss_cache() -> None:
    """Different crop bounds should be separate cache entries."""
    encoder = DummyEncoder()
    cache = EmbeddingCache(encoder=[encoder])

    meta_a = _make_metadata(crop_bounds=(0, 0, 32, 32))
    meta_b = _make_metadata(crop_bounds=(32, 32, 64, 64))

    cache(_make_context([meta_a]))
    cache(_make_context([meta_b]))

    assert encoder.call_count == 2


def test_batch_partial_cache() -> None:
    """Batch where some examples are cached and some are not."""
    encoder = DummyEncoder()
    cache = EmbeddingCache(encoder=[encoder])

    meta_a = _make_metadata(window_name="a")
    meta_b = _make_metadata(window_name="b")

    # Warm up cache for meta_a only
    cache(_make_context([meta_a], values=[1.0]))

    # Now call with [a, b] -- a should come from cache (1.0), b computed fresh (2.0)
    ctx = _make_context([meta_a, meta_b], values=[9.0, 2.0])
    result = cache(ctx)

    assert (result.feature_maps[0][0] == 1.0).all()  # cached, ignores 9.0
    assert (result.feature_maps[0][1] == 2.0).all()  # freshly computed


def test_batch_all_cached() -> None:
    """Batch where all examples are already cached."""
    encoder = DummyEncoder()
    cache = EmbeddingCache(encoder=[encoder])

    meta_a = _make_metadata(window_name="a")
    meta_b = _make_metadata(window_name="b")

    # Warm up both
    cache(_make_context([meta_a]))
    cache(_make_context([meta_b]))
    assert encoder.call_count == 2

    # Both cached -- encoder should NOT be called again
    result = cache(_make_context([meta_a, meta_b]))
    assert encoder.call_count == 2
    assert result.feature_maps[0].shape[0] == 2


def test_batch_order_preserved() -> None:
    """Output batch order should match input order, even with partial cache."""
    encoder = DummyEncoder()
    cache = EmbeddingCache(encoder=[encoder])

    meta_a = _make_metadata(window_name="a")
    meta_b = _make_metadata(window_name="b")
    meta_c = _make_metadata(window_name="c")

    # Warm up b only
    cache(_make_context([meta_b], values=[2.0]))

    # Now [a, b, c] -- a and c uncached, b cached
    ctx = _make_context([meta_a, meta_b, meta_c], values=[1.0, 9.0, 3.0])
    result = cache(ctx)

    assert (result.feature_maps[0][0] == 1.0).all()  # a: freshly computed
    assert (result.feature_maps[0][1] == 2.0).all()  # b: cached, ignores 9.0
    assert (result.feature_maps[0][2] == 3.0).all()  # c: freshly computed
