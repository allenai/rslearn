"""Molmo model."""

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from rslearn.train.model_context import ModelContext

from .component import FeatureExtractor, FeatureMaps


class Molmo(FeatureExtractor):
    """Molmo image encoder."""

    def __init__(
        self,
        model_name: str,
    ):
        """Instantiate a new Molmo instance.

        Args:
            model_name: the model name like "allenai/Molmo-7B-D-0924".
        """
        super().__init__()

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="cpu",
        )  # nosec
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="cpu",
        )  # nosec
        self.encoder = model.model.vision_backbone

    def forward(self, context: ModelContext) -> FeatureMaps:
        """Compute outputs from the backbone.

        Args:
            context: the model context. Input dicts must include "image" key containing
                the image to process. The images should have values 0-255.

        Returns:
            a FeatureMaps. Molmo produces features at one scale, so it will contain one
                feature map that is a Bx24x24x2048 tensor.
        """
        device = context.inputs[0]["image"].device
        molmo_inputs_list = []
        # Process each one so we can isolate just the full image without any crops.
        for inp in context.inputs:
            image = inp["image"].cpu().numpy().transpose(1, 2, 0)
            processed = self.processor.process(
                images=[image],
                text="",
            )
            molmo_inputs_list.append(processed["images"][0])
        molmo_inputs: torch.Tensor = torch.stack(molmo_inputs_list, dim=0).unsqueeze(1)

        image_features, _ = self.encoder.encode_image(molmo_inputs.to(device))

        # 576x2048 -> 24x24x2048
        return FeatureMaps(
            [image_features[:, 0, :, :].reshape(-1, 24, 24, 2048).permute(0, 3, 1, 2)]
        )
