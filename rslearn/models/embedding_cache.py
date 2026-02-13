"""Cache FeatureMaps by window name and crop bounds."""

import torch

from rslearn.train.model_context import ModelContext, SampleMetadata
from rslearn.utils.geometry import PixelBounds

from .component import FeatureExtractor, FeatureMaps, IntermediateComponent


class EmbeddingCache(FeatureExtractor):
    """Wraps an encoder and caches the FeatureMaps it produces.

    The cache is keyed by the window name and crop bounds. A stop gradient is always
    added since the assumption is that the user does not want gradients to pass to the
    encoder.

    The typical use case is to wrap a frozen encoder and train just a decoder/probe,
    using the cache to speed up training by skipping encoder computation on cache hits.
    """

    def __init__(
        self,
        encoder: list[FeatureExtractor | IntermediateComponent],
        cache_on_cpu: bool = True,
    ) -> None:
        """Create a new EmbeddingCache.

        Args:
            encoder: list of encoder modules. The first must be a FeatureExtractor,
                and following modules must be IntermediateComponents.
            cache_on_cpu: if True, store cached tensors on CPU to save GPU memory.
        """
        super().__init__()
        self.encoder = torch.nn.ModuleList(encoder)
        self.cache_on_cpu = cache_on_cpu
        # Cache maps (window_name, crop_bounds) -> list of detached tensors per level
        # Not a nn.Parameter so it won't be part of model state_dict
        self._cache: dict[tuple[str, PixelBounds], list[torch.Tensor]] = {}

    def _get_cache_key(self, metadata: SampleMetadata) -> tuple[str, PixelBounds]:
        """Get the cache key for a sample."""
        return (metadata.window_name, metadata.crop_bounds)

    def _run_encoder(self, context: ModelContext) -> FeatureMaps:
        """Run the encoder on the given context."""
        cur = self.encoder[0](context)
        for module in self.encoder[1:]:
            cur = module(cur, context)
        return cur

    def _get_device(self, context: ModelContext) -> torch.device:
        """Get the device from the context inputs."""
        for value in context.inputs[0].values():
            if hasattr(value, "image"):
                return value.image.device
            if isinstance(value, torch.Tensor):
                return value.device
        raise RuntimeError("could not determine device from context inputs")

    def forward(self, context: ModelContext) -> FeatureMaps:
        """Get cached embeddings or compute and cache them.

        Only runs the encoder on examples that are not cached.

        Args:
            context: the model context.

        Returns:
            FeatureMaps with detached (no gradient) tensors.
        """
        device = self._get_device(context)

        # Check which examples are not in the cache (need to compute embeddings)
        cache_keys = [self._get_cache_key(m) for m in context.metadatas]
        uncached_indices = [
            i for i, key in enumerate(cache_keys) if key not in self._cache
        ]

        # Run encoder on uncached examples and store results in cache
        if uncached_indices:
            subset_context = ModelContext(
                inputs=[context.inputs[i] for i in uncached_indices],
                metadatas=[context.metadatas[i] for i in uncached_indices],
                context_dict=context.context_dict,
            )
            intermediates = self._run_encoder(subset_context)

            for i, batch_idx in enumerate(uncached_indices):
                cache_key = cache_keys[batch_idx]
                example_tensors: list[torch.Tensor] = []
                for level_idx in range(len(intermediates.feature_maps)):
                    tensor = intermediates.feature_maps[level_idx][i : i + 1]
                    detached = tensor.detach()
                    if self.cache_on_cpu:
                        example_tensors.append(detached.cpu())
                    else:
                        example_tensors.append(detached.clone())
                self._cache[cache_key] = example_tensors

        # Build output in original batch order (everything is now cached)
        num_feature_maps = len(self._cache[cache_keys[0]])
        output_tensors: list[list[torch.Tensor]] = [[] for _ in range(num_feature_maps)]

        for cache_key in cache_keys:
            cached = self._cache[cache_key]
            for level_idx in range(num_feature_maps):
                tensor = cached[level_idx]
                if tensor.device != device:
                    tensor = tensor.to(device)
                output_tensors[level_idx].append(tensor)

        return FeatureMaps(feature_maps=[torch.cat(t, dim=0) for t in output_tensors])
