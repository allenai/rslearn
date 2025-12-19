"""OpenAI CLIP models."""

from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

from rslearn.train.model_context import ModelContext

from .component import FeatureExtractor, FeatureMaps


class CLIP(FeatureExtractor):
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

        self.processor = AutoProcessor.from_pretrained(model_name)  # nosec
        model = AutoModelForZeroShotImageClassification.from_pretrained(model_name)  # nosec
        self.encoder = model.vision_model

        # Get number of features and token map size from encoder attributes.
        self.num_features = self.encoder.post_layernorm.normalized_shape[0]
        crop_size = self.processor.image_processor.crop_size
        stride = self.encoder.embeddings.patch_embedding.stride
        self.height = crop_size["height"] // stride[0]
        self.width = crop_size["width"] // stride[1]

    def forward(self, context: ModelContext) -> FeatureMaps:
        """Compute outputs from the backbone.

        Args:
            context: the model context. Input dicts must include "image" key containing
                the image to process. The images should have values 0-255.

        Returns:
            a FeatureMaps with one feature map from the ViT, which is always Bx24x24x1024.
        """
        inputs = context.inputs
        device = inputs[0]["image"].image.device
        clip_inputs = self.processor(
            images=[
                inp["image"].image.cpu().numpy().transpose(1, 2, 0) for inp in inputs
            ],
            return_tensors="pt",
            padding=True,
        )
        pixel_values = clip_inputs["pixel_values"].to(device)
        output = self.encoder(pixel_values=pixel_values)
        # Ignore class token output which is before the patch tokens.
        image_features = output.last_hidden_state[:, 1:, :]
        batch_size = image_features.shape[0]

        # 576x1024 -> HxWxC
        return FeatureMaps(
            [
                image_features.reshape(
                    batch_size, self.height, self.width, self.num_features
                ).permute(0, 3, 1, 2)
            ]
        )
