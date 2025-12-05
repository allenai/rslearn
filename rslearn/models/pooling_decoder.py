"""Simple pooling decoder."""

from typing import Any

import torch

from rslearn.train.model_context import ModelContext

from .component import (
    FeatureMaps,
    FeatureVector,
    IntermediateComponent,
)


class PoolingDecoder(IntermediateComponent):
    """Decoder that computes flat vector from a 2D feature map.

    It inputs multi-scale features, but only uses the last feature map. Then applies a
    configurable number of convolutional layers before pooling, and a configurable
    number of fully connected layers after pooling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_conv_layers: int = 0,
        num_fc_layers: int = 0,
        conv_channels: int = 128,
        fc_channels: int = 512,
    ) -> None:
        """Initialize a PoolingDecoder.

        Args:
            in_channels: input channels (channels in the last feature map passed to
                this module)
            out_channels: channels for the output flat feature vector
            num_conv_layers: number of convolutional layers to apply, default 0
            num_fc_layers: number of fully-connected layers to apply, default 0
            conv_channels: number of channels to use for convolutional layers
            fc_channels: number of channels to use for fully-connected layers
        """
        super().__init__()
        conv_layers = []
        prev_channels = in_channels
        for _ in range(num_conv_layers):
            conv_layer = torch.nn.Sequential(
                torch.nn.Conv2d(prev_channels, conv_channels, 3, padding=1),
                torch.nn.ReLU(inplace=True),
            )
            conv_layers.append(conv_layer)
            prev_channels = conv_channels
        self.conv_layers = torch.nn.Sequential(*conv_layers)

        fc_layers = []
        for _ in range(num_fc_layers):
            fc_layer = torch.nn.Sequential(
                torch.nn.Linear(prev_channels, fc_channels),
                torch.nn.ReLU(inplace=True),
            )
            fc_layers.append(fc_layer)
            prev_channels = fc_channels
        self.fc_layers = torch.nn.Sequential(*fc_layers)

        self.output_layer = torch.nn.Linear(prev_channels, out_channels)

    def forward(self, intermediates: Any, context: ModelContext) -> Any:
        """Compute flat output vector from multi-scale feature map.

        Args:
            intermediates: the output from the previous component, which must be a FeatureMaps.
            context: the model context.

        Returns:
            flat feature vector
        """
        if not isinstance(intermediates, FeatureMaps):
            raise ValueError("input to PoolingDecoder must be a FeatureMaps")

        # Only use last feature map.
        features = intermediates.feature_maps[-1]

        features = self.conv_layers(features)
        features = torch.amax(features, dim=(2, 3))
        features = self.fc_layers(features)
        return FeatureVector(self.output_layer(features))


class SegmentationPoolingDecoder(PoolingDecoder):
    """Like PoolingDecoder, but copy output to all pixels.

    This allows for the model to produce a global output while still being compatible
    with SegmentationTask. This only makes sense for very small windows, since the
    output probabilities will be the same at all pixels. The main use case is to train
    for a classification-like task on small windows, but still produce a raster during
    inference on large windows.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_key: str = "image",
        **kwargs: Any,
    ):
        """Create a new SegmentationPoolingDecoder.

        Args:
            in_channels: input channels (channels in the last feature map passed to
                this module)
            out_channels: channels for the output flat feature vector
            image_key: the key in inputs for the image from which the expected width
                and height is derived.
            kwargs: other arguments to pass to PoolingDecoder.
        """
        super().__init__(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.image_key = image_key

    def forward(self, intermediates: Any, context: ModelContext) -> Any:
        """Extend PoolingDecoder forward to upsample the output to a segmentation mask.

        This only works when all of the pixels have the same segmentation target.
        """
        output_probs = super().forward(intermediates, context)
        # BC -> BCHW
        h, w = context.inputs[0][self.image_key].shape[1:3]
        feat_map = output_probs.feature_vector[:, :, None, None].repeat([1, 1, h, w])
        return FeatureMaps([feat_map])
