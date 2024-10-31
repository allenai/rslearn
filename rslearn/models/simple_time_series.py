"""SimpleTimeSeries encoder."""

from typing import Any

import torch


class SimpleTimeSeries(torch.nn.Module):
    """SimpleTimeSeries wraps another encoder and applies it on an image time series.

    It independently applies the other encoder on each image in the time series to
    extract feature maps. It then provides a few ways to combine the features into one
    final feature map:
    - Temporal max pooling.
    - ConvRNN.
    - 3D convolutions.
    - 1D convolutions (per-pixel, just apply it over time).
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        image_channels: int,
        op: str = "max",
        groups: list[list[int]] | None = None,
        num_layers: int | None = None,
    ) -> None:
        """Create a new SimpleTimeSeries.

        Args:
            encoder: the underlying encoder. It must provide get_backbone_channels
                function that returns the output channels.
            image_channels: the number of channels per image of the time series. The
                input should have multiple images concatenated on the channel axis, so
                this parameter is used to distinguish the different images.
            op: one of max, mean, convrnn, conv3d, or conv1d
            groups: sets of images for which to combine features. Within each set,
                features are combined using the specified operation; then, across sets,
                the features are concatenated and returned. The default is to combine
                features across all input images. For an application comparing
                before/after images of an event, it would make sense to concatenate the
                combined before features and the combined after features. groups is a
                list of sets, and each set is a list of image indices.
            num_layers: the number of layers for convrnn, conv3d, and conv1d ops.
        """
        super().__init__()
        self.encoder = encoder
        self.image_channels = image_channels
        self.op = op
        self.groups = groups

        out_channels = self.encoder.get_backbone_channels()
        if self.groups:
            self.num_groups = len(self.groups)
        else:
            self.num_groups = 1

        if self.op in ["convrnn", "conv3d", "conv1d"]:
            if num_layers is None:
                raise ValueError(f"num_layers must be specified for {self.op} op")

            if self.op == "convrnn":
                rnn_kernel_size = 3
                rnn_layers = []
                for _, count in out_channels:
                    cur_layer = [
                        torch.nn.Sequential(
                            torch.nn.Conv2d(
                                2 * count, count, rnn_kernel_size, padding="same"
                            ),
                            torch.nn.ReLU(inplace=True),
                        )
                    ]
                    for _ in range(num_layers - 1):
                        cur_layer.append(
                            torch.nn.Sequential(
                                torch.nn.Conv2d(
                                    count, count, rnn_kernel_size, padding="same"
                                ),
                                torch.nn.ReLU(inplace=True),
                            )
                        )
                    cur_layer = torch.nn.Sequential(*cur_layer)
                    rnn_layers.append(cur_layer)
                self.rnn_layers = torch.nn.ModuleList(rnn_layers)

            elif self.op == "conv3d":
                conv3d_layers = []
                for _, count in out_channels:
                    cur_layer = [
                        torch.nn.Sequential(
                            torch.nn.Conv3d(
                                count, count, 3, padding=1, stride=(2, 1, 1)
                            ),
                            torch.nn.ReLU(inplace=True),
                        )
                        for _ in range(num_layers)
                    ]
                    cur_layer = torch.nn.Sequential(*cur_layer)
                    conv3d_layers.append(cur_layer)
                self.conv3d_layers = torch.nn.ModuleList(conv3d_layers)

            elif self.op == "conv1d":
                conv1d_layers = []
                for _, count in out_channels:
                    cur_layer = [
                        torch.nn.Sequential(
                            torch.nn.Conv1d(count, count, 3, padding=1, stride=2),
                            torch.nn.ReLU(inplace=True),
                        )
                        for _ in range(num_layers)
                    ]
                    cur_layer = torch.nn.Sequential(*cur_layer)
                    conv1d_layers.append(cur_layer)
                self.conv1d_layers = torch.nn.ModuleList(conv1d_layers)

        else:
            assert self.op in ["max", "mean"]

    def get_backbone_channels(self) -> list:
        """Returns the output channels of this model when used as a backbone.

        The output channels is a list of (downsample_factor, depth) that corresponds
        to the feature maps that the backbone returns. For example, an element [2, 32]
        indicates that the corresponding feature map is 1/2 the input resolution and
        has 32 channels.

        Returns:
            the output channels of the backbone as a list of (downsample_factor, depth)
            tuples.
        """
        out_channels = []
        for downsample_factor, depth in self.encoder.get_backbone_channels():
            out_channels.append((downsample_factor, depth * self.num_groups))
        return out_channels

    def forward(
        self,
        inputs: list[dict[str, Any]],
    ) -> list[torch.Tensor]:
        """Compute outputs from the backbone.

        Inputs:
            inputs: input dicts that must include "image" key containing the image time
                series to process (with images concatenated on the channel dimension).
        """
        # First get features of each image.
        # To do so, we need to split up each grouped image into its component images (which have had their channels stacked).
        images = torch.stack([inp["image"] for inp in inputs], dim=0)
        n_batch = images.shape[0]
        n_images = images.shape[1] // self.image_channels
        n_height = images.shape[2]
        n_width = images.shape[3]
        batched_images = images.reshape(
            n_batch * n_images, self.image_channels, n_height, n_width
        )
        batched_inputs = [{"image": image} for image in batched_images]
        all_features = [
            feat_map.reshape(
                n_batch,
                n_images,
                feat_map.shape[1],
                feat_map.shape[2],
                feat_map.shape[3],
            )
            for feat_map in self.encoder(batched_inputs)
        ]

        # Groups defaults to flattening all the feature maps.
        groups = self.groups
        if not groups:
            groups = [list(range(n_images))]

        # Now compute aggregation over each group.
        # We handle each element of the multi-scale feature map separately.
        output_features = []
        for feature_idx in range(len(all_features)):
            aggregated_features = []
            for group in groups:
                group_features_list = []
                for image_idx in group:
                    group_features_list.append(
                        all_features[feature_idx][:, image_idx, :, :, :]
                    )
                # Resulting group features are (depth, batch, C, height, width).
                group_features = torch.stack(group_features_list, dim=0)

                if self.op == "max":
                    group_features = torch.amax(group_features, dim=0)
                elif self.op == "mean":
                    group_features = torch.mean(group_features, dim=0)
                elif self.op == "convrnn":
                    hidden = torch.zeros_like(group_features[0])
                    for cur in group_features:
                        hidden = self.rnn_layers[feature_idx](
                            torch.cat([hidden, cur], dim=1)
                        )
                    group_features = hidden
                elif self.op == "conv3d":
                    # Conv3D expects input to be (batch, C, depth, height, width).
                    group_features = group_features.permute(1, 2, 0, 3, 4)
                    group_features = self.conv3d_layers[feature_idx](group_features)
                    assert group_features.shape[2] == 1
                    group_features = group_features[:, :, 0, :, :]
                elif self.op == "conv1d":
                    # Conv1D expects input to be (batch, C, depth).
                    # We put width/height on the batch dimension.
                    group_features = group_features.permute(1, 3, 4, 2, 0)
                    n_batch, n_h, n_w, n_c, n_d = group_features.shape[0:5]
                    group_features = group_features.reshape(
                        n_batch * n_h * n_w, n_c, n_d
                    )
                    group_features = self.conv1d_layers[feature_idx](group_features)
                    assert group_features.shape[2] == 1
                    # Now we have to recover the batch/width/height dimensions.
                    group_features = (
                        group_features[:, :, 0]
                        .reshape(n_batch, n_h, n_w, n_c)
                        .permute(0, 3, 1, 2)
                    )
                else:
                    raise Exception(f"unknown aggregation op {self.op}")

                aggregated_features.append(group_features)

            # Finally at each scale we concatenate across groups.
            aggregated_features = torch.cat(aggregated_features, dim=1)

            output_features.append(aggregated_features)

        return output_features
