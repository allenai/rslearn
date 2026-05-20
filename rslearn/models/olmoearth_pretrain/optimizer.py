"""Layer-wise learning rate decay for OlmoEarth fine-tuning."""

from collections import defaultdict
from dataclasses import asdict, dataclass

import lightning as L
import torch
import torch.optim
from lightning.pytorch import Callback, LightningModule, Trainer
from torch.optim import Optimizer

from rslearn.log_utils import get_logger
from rslearn.train.optimizer import OptimizerFactory

logger = get_logger(__name__)

ENCODER_PREFIX = "model.encoder.0"


def get_layer_id(
    name: str, num_layers: int, encoder_prefix: str = ENCODER_PREFIX
) -> int:
    """Map a parameter name to its layer index for LR scaling.

    encoder_prefix should be the path from the LightningModule root to the
    OlmoEarth module (the FeatureExtractor in MultiTaskModel.encoder). OlmoEarth
    stores the actual Encoder at `self.model`, so parameter paths under the prefix
    look like ``<prefix>.model.blocks.3.attn.qkv.weight``.

    Typically ``"model.encoder.0"`` when OlmoEarth is the first encoder in
    MultiTaskModel.

    Returns 0..num_layers-1 for encoder blocks, 0 for patch_embeddings,
    and num_layers for everything else (norm, decoders = full LR).
    """
    if not name.startswith(encoder_prefix):
        return num_layers
    relative = name[len(encoder_prefix) + 1 :]
    if relative.startswith("model.blocks."):
        return int(relative.split(".")[2])
    if relative.startswith("model.patch_embeddings"):
        return 0
    return num_layers


@dataclass
class LayerDecayAdamW(OptimizerFactory):
    """AdamW with per-layer learning rate decay for OlmoEarth.

    Creates one param group per encoder depth level. LR for layer i is
    base_lr * layer_decay_rate ** (num_layers - i). Includes all params
    (even frozen ones) so it composes with SimpleFreeze.

    num_layers should be set based on the model size (12 for tiny and base,
    4 for nano).

    encoder_prefix: path from the LightningModule root to the OlmoEarth module,
        typically "model.encoder.0" when OlmoEarth is the first encoder in
        MultiTaskModel.
    """

    lr: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float | None = None
    weight_decay: float | None = None
    layer_decay_rate: float = 0.65
    num_layers: int = 12
    encoder_prefix: str = ENCODER_PREFIX

    def build(self, lm: L.LightningModule) -> Optimizer:
        """Build AdamW with per-layer param groups."""
        groups: dict[int, list] = defaultdict(list)
        for name, param in lm.named_parameters():
            layer_id = get_layer_id(name, self.num_layers, self.encoder_prefix)
            logger.debug("param %s -> layer_id %d", name, layer_id)
            if layer_id > self.num_layers:
                raise ValueError(
                    f"param {name!r} mapped to layer_id={layer_id} "
                    f"which exceeds num_layers={self.num_layers}"
                )
            groups[layer_id].append(param)

        expected = set(range(self.num_layers + 1))
        missing = expected - groups.keys()
        if missing:
            raise ValueError(
                f"layer decay expected layers 0..{self.num_layers} but "
                f"missing layers: {sorted(missing)}"
            )

        param_groups = []
        for layer_id in sorted(groups.keys()):
            scale = self.layer_decay_rate ** (self.num_layers - layer_id)
            lr = self.lr * scale
            param_groups.append({"params": groups[layer_id], "lr": lr})
            logger.info(
                f"layer_decay group layer={layer_id} lr={lr:.2e} "
                f"params={len(groups[layer_id])}"
            )

        exclude = {"lr", "layer_decay_rate", "num_layers", "encoder_prefix"}
        adamw_kwargs = {
            k: v for k, v in asdict(self).items() if v is not None and k not in exclude
        }
        return torch.optim.AdamW(param_groups, **adamw_kwargs)


class SimpleFreeze(Callback):
    """Freeze a module until a given epoch, then unfreeze.

    Unlike BaseFinetuning/FreezeUnfreeze, this does NOT manipulate optimizer
    param groups. It only toggles requires_grad. Designed to compose with
    LayerDecayAdamW which already includes all params in the optimizer.
    """

    def __init__(
        self,
        module_selector: list[str | int],
        unfreeze_at_epoch: int,
    ) -> None:
        """Create a new SimpleFreeze.

        Args:
            module_selector: the selector that identifies the model component to freeze.
                See FreezeUnfreeze.
            unfreeze_at_epoch: unfreeze the component at the start of this epoch.
        """
        super().__init__()
        self.module_selector = module_selector
        self.unfreeze_at_epoch = unfreeze_at_epoch

    def _get_target_module(self, pl_module: LightningModule) -> torch.nn.Module:
        target: torch.nn.Module = pl_module
        for k in self.module_selector:
            target = target[k] if isinstance(k, int) else getattr(target, k)
        return target

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Toggle requires_grad based on current epoch."""
        freeze = trainer.current_epoch < self.unfreeze_at_epoch
        for p in self._get_target_module(pl_module).parameters():
            p.requires_grad = not freeze
