"""Feature pyramid network."""

import collections

import torch
import torchvision


class Fpn(torch.nn.Module):
    """A feature pyramid network (FPN).

    The FPN inputs a multi-scale feature map. At each scale, it computes new features
    of a configurable depth based on all input features. So it is best used for maps
    that were computed sequentially where earlier features don't have the context from
    later features, but comprehensive features at each resolution are desired.
    """

    def __init__(self, prev_module: torch.nn.Module, out_channels: int = 128):
        """Initialize a new Fpn instance.

        Args:
            prev_module: the preceding module, used to determine the input channels
            out_channels: output depth at each resolution
        """
        super().__init__()

        backbone_channels = prev_module.get_backbone_channels()
        in_channels_list = [ch[1] for ch in backbone_channels]
        self.fpn = torchvision.ops.FeaturePyramidNetwork(
            in_channels_list=in_channels_list, out_channels=out_channels
        )
        self.out_channels = [[ch[0], out_channels] for ch in backbone_channels]

    def forward(self, x: list[torch.Tensor]):
        """Compute outputs of the FPN.

        Args:
            x: the multi-scale feature maps

        Returns:
            new multi-scale feature maps from the FPN
        """
        inp = collections.OrderedDict([(f"feat{i}", el) for i, el in enumerate(x)])
        output = self.fpn(inp)
        output = list(output.values())

        if self.prepend:
            return output + x
        else:
            return output

    def get_backbone_channels(self):
        """Returns the output channels of this model when used as a backbone.

        The output channels is a list of (downsample_factor, depth) that corresponds
        to the feature maps that the backbone returns. For example, an element [2, 32]
        indicates that the corresponding feature map is 1/2 the input resolution and
        has 32 channels.

        Returns:
            the output channels of the backbone as a list of (downsample_factor, depth)
            tuples.
        """
        return self.out_channels
