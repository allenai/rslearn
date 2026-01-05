"""Module wrapper provided for backwards compatibility."""

from typing import Any

import torch

from rslearn.train.model_context import ModelContext

from .component import (
    FeatureExtractor,
    FeatureMaps,
    IntermediateComponent,
)


class EncoderModuleWrapper(FeatureExtractor):
    """Wraps one or more IntermediateComponents to function as the feature extractor.

    The first component should input a FeatureMaps, which will be computed from the
    overall inputs by stacking the "image" key from each input dict.
    """

    def __init__(
        self,
        module: IntermediateComponent | None = None,
        modules: list[IntermediateComponent] = [],
    ):
        """Initialize an EncoderModuleWrapper.

        Args:
            module: the IntermediateComponent to wrap for use as a FeatureExtractor.
                Exactly one of module or modules must be set.
            modules: list of modules to wrap
        """
        super().__init__()
        if module is not None and len(modules) > 0:
            raise ValueError("only one of module or modules should be set")
        if module is not None:
            self.encoder_modules = torch.nn.ModuleList([module])
        elif len(modules) > 0:
            self.encoder_modules = torch.nn.ModuleList(modules)
        else:
            raise ValueError("one of module or modules must be set")

    def forward(self, context: ModelContext) -> Any:
        """Compute outputs from the wrapped module.

        Args:
            context: the model context. Input dicts must include "image" key containing
                the image to convert to a FeatureMaps, which will be passed to the
                first wrapped module.

        Returns:
            the output from the last wrapped module.
        """
        images = torch.stack([inp["image"].image for inp in context.inputs], dim=0)
        cur: Any = FeatureMaps([images])
        for m in self.encoder_modules:
            cur = m(cur, context)
        return cur
