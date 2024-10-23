"""SSL4EO-S12 models."""

from typing import Any

import torch
import torchvision


class Ssl4eoS12(torch.nn.Module):
    """The SSL4EO-S12 family of pretrained models."""

    def __init__(
        self,
        backbone_ckpt_path: str,
        arch: str = "resnet50",
        output_layers: list[int] = [0, 1, 2, 3],
    ) -> None:
        """Instantiate a new Swin instance.

        Args:
            backbone_ckpt_path: the path to the backbone checkpoint, need to download
                from https://github.com/zhu-xlab/SSL4EO-S12
            arch: model architecture, currently only resnet50 is supported
            output_layers: return the outputs from these layers, defaults to all of the
                four resnet outputs.
        """
        super().__init__()
        self.arch = arch
        self.output_layers = output_layers

        if arch == "resnet50":
            self.model = torchvision.models.resnet50()
            self.model.conv1 = torch.nn.Conv2d(
                13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )

        else:
            raise ValueError(f"unknown SSL4EO-S12 architecture {arch}")

        state_dict = torch.load(backbone_ckpt_path)
        state_dict = state_dict["teacher"]
        prefix = "module.backbone."
        state_dict = {
            k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)
        }
        missing_keys, unexpected_keys = self.model.load_state_dict(
            state_dict, strict=False
        )
        if missing_keys or unexpected_keys:
            print(
                f"warning: got missing_keys={missing_keys}, unexpected_keys={unexpected_keys} when loading SSL4EO-S12 state dict"
            )

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
        if self.arch == "resnet50":
            all_out_channels = [
                (4, 256),
                (8, 512),
                (16, 1024),
                (32, 2048),
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
        x = torch.stack([inp["image"] for inp in inputs], dim=0)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        layer1 = self.model.layer1(x)
        layer2 = self.model.layer2(layer1)
        layer3 = self.model.layer3(layer2)
        layer4 = self.model.layer4(layer3)
        all_features = [layer1, layer2, layer3, layer4]
        return [all_features[idx] for idx in self.output_layers]
