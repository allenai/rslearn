"""SimpleTimeSeries encoder."""

import warnings
from typing import Any

import torch
from einops import rearrange

from rslearn.train.model_context import ModelContext, RasterImage

from .component import FeatureExtractor, FeatureMaps


class SimpleTimeSeries(FeatureExtractor):
    """SimpleTimeSeries wraps another FeatureExtractor and applies it on an image time series.

    It independently applies the other FeatureExtractor on each image in the time series to
    extract feature maps. It then provides a few ways to combine the features into one
    final feature map:
    - Temporal max pooling.
    - ConvRNN.
    - 3D convolutions.
    - 1D convolutions (per-pixel, just apply it over time).
    """

    def __init__(
        self,
        encoder: FeatureExtractor,
        num_timesteps_per_forward_pass: int = 1,
        op: str = "max",
        groups: list[list[int]] | None = None,
        num_layers: int | None = None,
        image_key: str = "image",
        backbone_channels: list[tuple[int, int]] | None = None,
        image_keys: list[str] | dict[str, int] | None = None,
        image_channels: int | None = None,
    ) -> None:
        """Create a new SimpleTimeSeries.

        Args:
            encoder: the underlying FeatureExtractor. It must provide get_backbone_channels
                function that returns the output channels, or backbone_channels must be set.
                It must output a FeatureMaps.
            num_timesteps_per_forward_pass: how many timesteps to pass to the encoder
                in each forward pass. Defaults to 1 (one timestep per forward pass).
                Set to a higher value to batch multiple timesteps together, e.g. for
                pre/post change detection where you want 4 pre and 4 post images
                processed together.
            op: one of max, mean, convrnn, conv3d, or conv1d
            groups: sets of images for which to combine features. Within each set,
                features are combined using the specified operation; then, across sets,
                the features are concatenated and returned. The default is to combine
                features across all input images. For an application comparing
                before/after images of an event, it would make sense to concatenate the
                combined before features and the combined after features. groups is a
                list of sets, and each set is a list of image indices.
            num_layers: the number of layers for convrnn, conv3d, and conv1d ops.
            image_key: the key to access the images (used when image_keys is not set).
            backbone_channels: manually specify the backbone channels. Can be set if
                the encoder does not provide get_backbone_channels function.
            image_keys: list of keys in input dict to process as multimodal inputs.
                All keys use the same num_timesteps_per_forward_pass. If not set,
                only the single image_key is used. Passing a dict[str, int] is
                deprecated and will be removed on 2026-04-01.
            image_channels: Deprecated, use num_timesteps_per_forward_pass instead.
                Will be removed on 2026-04-01.
        """
        # Handle deprecated image_channels parameter
        if image_channels is not None:
            warnings.warn(
                "image_channels is deprecated and will be removed on 2026-04-01. "
                "Use num_timesteps_per_forward_pass instead. The new parameter directly "
                "specifies the number of timesteps per forward pass rather than requiring "
                "image_channels // actual_channels.",
                FutureWarning,
                stacklevel=2,
            )

        # Handle deprecated dict form of image_keys
        deprecated_image_keys_dict: dict[str, int] | None = None
        if isinstance(image_keys, dict):
            warnings.warn(
                "Passing image_keys as a dict is deprecated and will be removed on "
                "2026-04-01. Use image_keys as a list[str] and set "
                "num_timesteps_per_forward_pass instead.",
                FutureWarning,
                stacklevel=2,
            )
            deprecated_image_keys_dict = image_keys
            image_keys = None  # Will use deprecated path in forward

        super().__init__()
        self.encoder = encoder
        self.num_timesteps_per_forward_pass = num_timesteps_per_forward_pass
        # Store deprecated parameters for runtime conversion
        self._deprecated_image_channels = image_channels
        self._deprecated_image_keys_dict = deprecated_image_keys_dict
        self.op = op
        self.groups = groups
        # Normalize image_key to image_keys list form
        if image_keys is not None:
            self.image_keys = image_keys
        else:
            self.image_keys = [image_key]

        if backbone_channels is not None:
            out_channels = backbone_channels
        else:
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

    def _get_batched_images(
        self, input_dicts: list[dict[str, Any]], image_key: str, num_timesteps: int
    ) -> list[RasterImage]:
        """Collect and reshape images across input dicts.

        The BTCHW image time series are reshaped to (B*T)CHW so they can be passed to
        the forward pass of a per-image (unitemporal) model.

        Args:
            input_dicts: list of input dictionaries containing RasterImage objects.
            image_key: the key to access the RasterImage in each input dict.
            num_timesteps: how many timesteps to batch together per forward pass.
        """
        images = torch.stack(
            [input_dict[image_key].image for input_dict in input_dicts], dim=0
        )  # B, C, T, H, W
        timestamps = [input_dict[image_key].timestamps for input_dict in input_dicts]
        # num_timesteps specifies how many timesteps to batch together per forward pass.
        # For example, if the input has 8 timesteps and num_timesteps=4, we do 2
        # forward passes, each with 4 timesteps batched together.
        batched_timesteps = images.shape[2] // num_timesteps
        images = rearrange(
            images,
            "b c (b_t k_t) h w -> (b b_t) c k_t h w",
            b_t=batched_timesteps,
            k_t=num_timesteps,
        )
        if timestamps[0] is None:
            new_timestamps = [None] * images.shape[0]
        else:
            # we also need to split the timestamps
            new_timestamps = []
            for t in timestamps:
                for i in range(batched_timesteps):
                    new_timestamps.append(
                        t[i * num_timesteps : (i + 1) * num_timesteps]
                    )
        return [
            RasterImage(image=image, timestamps=timestamps)
            for image, timestamps in zip(images, new_timestamps)
        ]  # C, T, H, W

    def forward(
        self,
        context: ModelContext,
    ) -> FeatureMaps:
        """Compute outputs from the backbone.

        Args:
            context: the model context. Input dicts must include "image" key containing the image time
                series to process (with images concatenated on the channel dimension).

        Returns:
            the FeatureMaps aggregated temporally.
        """
        # First get features of each image.
        # To do so, we need to split up each grouped image into its component images (which have had their channels stacked).
        batched_inputs: list[dict[str, Any]] | None = None
        n_batch = len(context.inputs)
        n_images: int | None = None

        if self._deprecated_image_keys_dict is not None:
            # Deprecated dict form: each key has its own channels_per_timestep.
            # The channels_per_timestep could be used to group multiple timesteps,
            # together, so we need to divide by the actual image channel count to get
            # the number of timesteps to be grouped.
            for (
                image_key,
                channels_per_timestep,
            ) in self._deprecated_image_keys_dict.items():
                # For deprecated image_keys dict, the value is channels per timestep,
                # so we need to compute num_timesteps from the actual image channels
                sample_image = context.inputs[0][image_key].image
                actual_channels = sample_image.shape[0]  # C in CTHW
                num_timesteps = channels_per_timestep // actual_channels
                batched_images = self._get_batched_images(
                    context.inputs, image_key, num_timesteps
                )

                if batched_inputs is None:
                    batched_inputs = [{} for _ in batched_images]
                    n_images = len(batched_images) // n_batch
                elif n_images != len(batched_images) // n_batch:
                    raise ValueError(
                        "expected all modalities to have the same number of timesteps"
                    )

                for i, image in enumerate(batched_images):
                    batched_inputs[i][image_key] = image

        else:
            # Determine num_timesteps - either from deprecated image_channels or
            # directly from num_timesteps_per_forward_pass
            if self._deprecated_image_channels is not None:
                # Backwards compatibility: compute num_timesteps from image_channels
                # (which should be a multiple of the actual per-timestep channels).
                sample_image = context.inputs[0][self.image_keys[0]].image
                actual_channels = sample_image.shape[0]  # C in CTHW
                num_timesteps = self._deprecated_image_channels // actual_channels
            else:
                num_timesteps = self.num_timesteps_per_forward_pass

            for image_key in self.image_keys:
                batched_images = self._get_batched_images(
                    context.inputs, image_key, num_timesteps
                )

                if batched_inputs is None:
                    batched_inputs = [{} for _ in batched_images]
                    n_images = len(batched_images) // n_batch
                elif n_images != len(batched_images) // n_batch:
                    raise ValueError(
                        "expected all modalities to have the same number of timesteps"
                    )

                for i, image in enumerate(batched_images):
                    batched_inputs[i][image_key] = image

        assert n_images is not None
        # Now we can apply the underlying FeatureExtractor.
        # Its output must be a FeatureMaps.
        assert batched_inputs is not None
        encoder_output = self.encoder(
            ModelContext(
                inputs=batched_inputs,
                metadatas=context.metadatas,
            )
        )
        if not isinstance(encoder_output, FeatureMaps):
            raise ValueError(
                "output of underlying FeatureExtractor in SimpleTimeSeries must be a FeatureMaps"
            )
        all_features = [
            feat_map.reshape(
                n_batch,
                n_images,
                feat_map.shape[1],
                feat_map.shape[2],
                feat_map.shape[3],
            )
            for feat_map in encoder_output.feature_maps
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
                    raise ValueError(f"unknown aggregation op {self.op}")

                aggregated_features.append(group_features)

            # Finally at each scale we concatenate across groups.
            aggregated_features = torch.cat(aggregated_features, dim=1)

            output_features.append(aggregated_features)

        return FeatureMaps(output_features)
