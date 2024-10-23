"""Swin Transformer."""

from typing import Any

import torch
import torchvision
from torchvision.models.swin_transformer import (
    Swin_B_Weights,
    Swin_S_Weights,
    Swin_T_Weights,
    Swin_V2_B_Weights,
    Swin_V2_S_Weights,
    Swin_V2_T_Weights,
)


class Swin(torch.nn.Module):
    """A Swin Transformer model.

    It can either be used stand-alone for classification, or as a feature extractor in
    combination with other decoder modules.
    """

    def __init__(
        self,
        arch: str = "swin_v2_b",
        pretrained: bool = False,
        input_channels: int = 3,
        output_layers: list[int] | None = None,
        num_outputs: int = 1000,
    ) -> None:
        """Instantiate a new Swin instance.

        Args:
            arch: the architecture, e.g. "swin_v2_b" (default) or "swin_t"
            pretrained: set True to use ImageNet pre-trained weights
            input_channels: number of input channels (default 3)
            output_layers: list of layers to output, default use as classification
                model. For feature extraction, [1, 3, 5, 7] is recommended.
            num_outputs: number of output logits, defaults to 1000 which matches the
                pretrained models.
        """
        super().__init__()
        self.arch = arch
        self.output_layers = output_layers

        kwargs = {}

        if arch == "swin_t":
            if pretrained:
                kwargs["weights"] = Swin_T_Weights.IMAGENET1K_V1
            self.model = torchvision.models.swin_t(**kwargs)
        elif arch == "swin_s":
            if pretrained:
                kwargs["weights"] = Swin_S_Weights.IMAGENET1K_V1
            self.model = torchvision.models.swin_s(**kwargs)
        elif arch == "swin_b":
            if pretrained:
                kwargs["weights"] = Swin_B_Weights.IMAGENET1K_V1
            self.model = torchvision.models.swin_b(**kwargs)
        elif arch == "swin_v2_t":
            if pretrained:
                kwargs["weights"] = Swin_V2_T_Weights.IMAGENET1K_V1
            self.model = torchvision.models.swin_v2_t(**kwargs)
        elif arch == "swin_v2_s":
            if pretrained:
                kwargs["weights"] = Swin_V2_S_Weights.IMAGENET1K_V1
            self.model = torchvision.models.swin_v2_s(**kwargs)
        elif arch == "swin_v2_b":
            if pretrained:
                kwargs["weights"] = Swin_V2_B_Weights.IMAGENET1K_V1
            self.model = torchvision.models.swin_v2_b(**kwargs)
        else:
            raise ValueError(f"unknown swin architecture {arch}")

        # Adjust first layer to accommodate different number of input channels
        # if needed.
        if input_channels != 3:
            self.model.features[0][0] = torch.nn.Conv2d(
                input_channels,
                self.model.features[0][0].out_channels,
                kernel_size=(4, 4),
                stride=(4, 4),
            )

        # Similarly adjust last layer.
        # We do this rather than passing num_classes since passing it would result in
        # incompatibility with loading the pretrained weights.
        if num_outputs != self.model.head.out_features:
            self.model.head = torch.nn.Linear(self.model.head.in_features, num_outputs)

    def get_backbone_channels(self) -> list[tuple[int, int]]:
        """Returns the output channels of this model when used as a backbone.

        The output channels is a list of (downsample_factor, depth) that corresponds
        to the feature maps that the backbone returns. For example, an element [2, 32]
        indicates that the corresponding feature map is 1/2 the input resolution and
        has 32 channels.

        Returns:
            the output channels of the backbone as a list of (downsample_factor, depth)
            tuples.
        """
        assert self.output_layers

        if self.arch in ["swin_b", "swin_v2_b"]:
            all_out_channels = [
                (4, 128),
                (4, 128),
                (4, 128),
                (8, 256),
                (8, 256),
                (16, 512),
                (16, 512),
                (32, 1024),
                (32, 1024),
            ]
        elif self.arch in ["swin_s", "swin_v2_s", "swin_t", "swin_v2_t"]:
            all_out_channels = [
                (4, 96),
                (4, 96),
                (8, 192),
                (8, 192),
                (16, 384),
                (16, 384),
                (32, 768),
                (32, 768),
            ]
        return [all_out_channels[idx] for idx in self.output_layers]

    def forward(
        self,
        inputs: list[dict[str, Any]],
    ) -> list[torch.Tensor]:
        """Compute outputs from the backbone.

        If output_layers is set, then the outputs are multi-scale feature maps;
        otherwise, the model is being used for classification so the outputs are class
        probabilities and the loss.

        Inputs:
            inputs: input dicts that must include "image" key containing the image to
                process.
        """
        images = torch.stack([inp["image"] for inp in inputs], dim=0)

        if self.output_layers:
            layer_features = []
            x = images
            for layer in self.model.features:
                x = layer(x)
                layer_features.append(x.permute(0, 3, 1, 2))
            return [layer_features[idx] for idx in self.output_layers]

        else:
            return self.model(images)
