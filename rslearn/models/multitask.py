"""MultiTaskModel for rslearn."""

from typing import Any

import torch

from rslearn.models.pooling import AveragePool, BasePool


def apply_decoder(
    features: list[torch.Tensor],
    inputs: list[dict[str, Any]],
    targets: list[dict[str, Any]] | None,
    decoder: list[torch.nn.Module],
    name: str,
    outputs: list[dict[str, Any]],
    losses: dict[str, torch.Tensor],
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    """Apply a decoder to a list of inputs and targets.

    Args:
        features: list of features
        inputs: list of input dicts
        targets: list of target dicts
        decoder: list of decoder modules
        name: the name of the decoder/task (which must match)
        outputs: list of output dicts
        losses: dictionary of loss values

    Returns:
        tuple of (outputs, losses)
    """
    # First, apply all but the last module in the decoder to the features
    cur = features
    for module in decoder[:-1]:
        if isinstance(module, BasePool):
            cur, weights = module(cur, inputs, return_weights=True)
        else:
            cur = module(cur, inputs)
    print("=" * 100)
    print("ATTN MEAN:", weights.mean(0))
    print("ATTN 0:", weights[0, :])
    print("=" * 100)

    if targets is None:
        cur_targets = None
    else:
        cur_targets = [target[name] for target in targets]

    # Then, apply the last module to the features and targets
    cur_output, cur_loss_dict = decoder[-1](cur, inputs, cur_targets)
    for idx, entry in enumerate(cur_output):
        outputs[idx][name] = entry
    for loss_name, loss_value in cur_loss_dict.items():
        losses[f"{name}_{loss_name}"] = loss_value
    return outputs, losses


class MultiTaskModel(torch.nn.Module):
    """MultiTask model wrapper.

    MultiTaskModel first passes its inputs through the sequential encoder models.

    Then, it applies one sequential decoder for each configured task. It computes
    outputs and loss using the final module in the decoder.
    """

    def __init__(
        self,
        encoder: list[torch.nn.Module],
        decoders: dict[str, list[torch.nn.Module]],
        checkpoint_path: str | None = None,
        lazy_decode: bool = False,
    ):
        """Initialize a new MultiTaskModel.

        Args:
            encoder: modules to compute intermediate feature representations.
            decoders: modules to compute outputs and loss, should match number of tasks.
            checkpoint_path: path to checkpoint to load decoder weights from
            lazy_decode: if True, only decode the outputs specified in the batch.
        """
        super().__init__()
        self.lazy_decode = lazy_decode
        self.encoder = torch.nn.Sequential(*encoder)

        if lazy_decode:
            print(
                "INFO: lazy decoding enabled, check source is consistent across batch"
            )

        if checkpoint_path is not None:
            print(f"INFO: loading full model weights from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path)["state_dict"]
            self.load_state_dict(
                {k.replace("model.", "", 1): v for k, v in state_dict.items()}
            )

        for name, decoder in decoders.items():
            if not isinstance(decoder[0], BasePool):
                # Add a default pool module for backwards compatibility with old configs
                print(
                    f"INFO: pooling decoder not found for {name}, using default average pool"
                )
                decoders[name].insert(0, AveragePool())
            else:
                print(f"INFO: using {decoder[0].__class__.__name__} for {name} pooling")

        self.decoders = torch.nn.ModuleDict(
            {name: torch.nn.ModuleList(decoder) for name, decoder in decoders.items()}
        )

    def forward(
        self,
        inputs: list[dict[str, Any]],
        targets: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
        """Apply the sequence of modules on the inputs.

        Args:
            inputs: list of input dicts
            targets: optional list of target dicts

        Returns:
            tuple (outputs, loss_dict) from the last module.
        """
        features = self.encoder(inputs)
        outputs: list[dict[str, Any]] = [{} for _ in inputs]
        losses: dict[str, torch.Tensor] = {}
        if self.lazy_decode:
            # Assume that all inputs have the same dataset_source
            dataset_source = inputs[0]["dataset_source"]
            decoder = self.decoders[dataset_source]
            apply_decoder(
                features, inputs, targets, decoder, dataset_source, outputs, losses
            )
        else:
            for name, decoder in self.decoders.items():
                apply_decoder(features, inputs, targets, decoder, name, outputs, losses)
        return outputs, losses
