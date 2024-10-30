"""OpenAI CLIP models."""

from typing import Any

import torch
from transformers import AutoModelForZeroShotImageClassification, AutoProcessor


class CLIP(torch.nn.Module):
    """CLIP image encoder."""

    def __init__(
        self,
        model_name: str,
    ):
        """Instantiate a new CLIP instance.

        Args:
            model_name: the model name like "openai/clip-vit-large-patch14-336".
        """
        super().__init__()

        self.processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForZeroShotImageClassification.from_pretrained(model_name)
        self.encoder = model.vision_model

        # Get number of features and token map size from encoder attributes.
        self.num_features = self.encoder.post_layernorm.normalized_shape[0]
        crop_size = self.processor.image_processor.crop_size
        stride = self.encoder.embeddings.patch_embedding.stride
        self.height = crop_size["height"] // stride[0]
        self.width = crop_size["width"] // stride[1]

    def forward(self, inputs: list[dict[str, Any]]) -> list[torch.Tensor]:
        """Compute outputs from the backbone.

        Inputs:
            inputs: input dicts that must include "image" key containing the image to
                process. The images should have values 0-255.

        Returns:
            list of feature maps. The ViT produces features at one scale, so the list
                contains a single Bx24x24x1024 feature map.
        """
        device = inputs[0]["image"].device
        clip_inputs = self.processor(
            images=[inp["image"].cpu().numpy().transpose(1, 2, 0) for inp in inputs],
            return_tensors="pt",
            padding=True,
        )
        pixel_values = clip_inputs["pixel_values"].to(device)
        output = self.encoder(pixel_values=pixel_values)
        # Ignore class token output which is before the patch tokens.
        image_features = output.last_hidden_state[:, 1:, :]
        batch_size = image_features.shape[0]

        # 576x1024 -> HxWxC
        return [
            image_features.reshape(
                batch_size, self.height, self.width, self.num_features
            ).permute(0, 3, 1, 2)
        ]
