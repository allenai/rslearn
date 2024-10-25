"""Molmo model."""

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoProcessor


class Molmo(torch.nn.Module):
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
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="cpu",
        )
        self.encoder = model.model.vision_backbone

    def forward(self, inputs: list[dict[str, Any]]) -> list[torch.Tensor]:
        """Compute outputs from the backbone.

        Inputs:
            inputs: input dicts that must include "image" key containing the image to
                process. The images should have values 0-255.

        Returns:
            list of feature maps. Molmo produces features at one scale, so the list
                contains a single Bx24x24x2048 tensor.
        """
        device = inputs[0]["image"].device
        molmo_inputs_list = []
        # Process each one so we can isolate just the full image without any crops.
        for inp in inputs:
            image = inp["image"].cpu().numpy().transpose(1, 2, 0)
            processed = self.processor.process(
                images=[image],
                text="",
            )
            molmo_inputs_list.append(processed["images"][0])
        molmo_inputs: torch.Tensor = torch.stack(molmo_inputs_list, dim=0).unsqueeze(1)

        image_features, _ = self.encoder.encode_image(molmo_inputs.to(device))

        # 576x2048 -> 24x24x2048
        return [
            image_features[:, 0, :, :].reshape(-1, 24, 24, 2048).permute(0, 3, 1, 2)
        ]
