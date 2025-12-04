"""Feature pyramid network."""

import collections
from typing import Any

import torchvision

from rslearn.train.model_context import ModelContext

from .component import FeatureMaps, IntermediateComponent


class Fpn(IntermediateComponent):
    """A feature pyramid network (FPN).

    The FPN inputs a multi-scale feature map. At each scale, it computes new features
    of a configurable depth based on all input features. So it is best used for maps
    that were computed sequentially where earlier features don't have the context from
    later features, but comprehensive features at each resolution are desired.
    """

    def __init__(
        self, in_channels: list[int], out_channels: int = 128, prepend: bool = False
    ):
        """Initialize a new Fpn instance.

        Args:
            in_channels: the input channels for each feature map
            out_channels: output depth at each resolution
            prepend: prepend the outputs of FPN rather than replacing the input
                features
        """
        super().__init__()
        self.prepend = prepend
        self.fpn = torchvision.ops.FeaturePyramidNetwork(
            in_channels_list=in_channels, out_channels=out_channels
        )

    def forward(self, intermediates: Any, context: ModelContext) -> FeatureMaps:
        """Compute outputs of the FPN.

        Args:
            intermediates: the output from the previous component, which must be a FeatureMaps.
            context: the model context.

        Returns:
            new multi-scale feature maps from the FPN.
        """
        if not isinstance(intermediates, FeatureMaps):
            raise ValueError("input to Fpn must be FeatureMaps")

        feature_maps = intermediates.feature_maps
        inp = collections.OrderedDict(
            [(f"feat{i}", el) for i, el in enumerate(feature_maps)]
        )
        output = self.fpn(inp)
        output = list(output.values())

        if self.prepend:
            return FeatureMaps(output + feature_maps)
        else:
            return FeatureMaps(output)
