"""DETR DEtection TRansformer decoder for object detection tasks.

Most of the modules here are adapted from here:
https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/detr.py#L258
The original code is:
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

import rslearn.models.detr.box_ops as box_ops
from rslearn.models.component import FeatureMaps, Predictor
from rslearn.train.model_context import ModelContext, ModelOutput

from .matcher import HungarianMatcher
from .position_encoding import PositionEmbeddingSine
from .transformer import Transformer
from .util import accuracy

DEFAULT_WEIGHT_DICT: dict[str, float] = {
    "loss_ce": 1,
    "loss_bbox": 5,
    "loss_giou": 2,
}


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ):
        """Create a new MLP.

        Args:
            input_dim: input dimension.
            hidden_dim: hidden dimension.
            output_dim: output dimension.
            num_layers: number of layers in this MLP.
        """
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP."""
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DetrPredictor(nn.Module):
    """DETR prediction module.

    This is DETR up to and excluding computing the loss.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_queries: int = 100,
        transformer: Transformer = Transformer(),
        aux_loss: bool = False,
    ):
        """Initializes the model.

        Args:
            in_channels: number of channels in features computed by the backbone.
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            transformer: torch module of the transformer architecture. See transformer.py
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.aux_loss = aux_loss

    def forward(
        self, feat_map: torch.Tensor, pos_embedding: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute the detection outputs.

        Args:
            feat_map: the input feature map.
            pos_embedding: positional embedding.

        Returns:
            output dict containing predicted boxes, classification logits, and
                aux_outputs (if aux_loss is enabled).
        """
        hs = self.transformer(
            src=self.input_proj(feat_map),
            query_embed=self.query_embed.weight,
            pos_embed=pos_embedding,
        )[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(
        self, outputs_class: torch.Tensor, outputs_coord: torch.Tensor
    ) -> list[dict[str, torch.Tensor]]:
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


class SetCriterion(nn.Module):
    """SetCriterion computes the loss for DETR.

    The process happens in two steps:
    (1) we compute hungarian assignment between ground truth boxes and the outputs of the model
    (2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher = HungarianMatcher(),
        weight_dict: dict[str, float] = DEFAULT_WEIGHT_DICT,
        eos_coef: float = 0.1,
        losses: list[str] = ["labels", "boxes", "cardinality"],
    ):
        """Create a SetCriterion.

        Args:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: int,
        log: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Compute classification loss (NLL).

        Args:
            outputs: the outputs from the model.
            targets: target dicts, which must contain the key "labels" containing a tensor of dim [nb_target_boxes].
            indices: the matching indices between outputs and targets.
            num_boxes: number of boxes, ignored.
            log: whether to add additional metrics to the loss dict for logging.

        Returns:
            loss dict, mapping from loss name to value. The actual loss is stored under
                loss_ce.
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )
        losses = {"loss_ce": loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: int,
    ) -> dict[str, torch.Tensor]:
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes.

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device
        )
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: int,
    ) -> dict[str, torch.Tensor]:
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(
        self, indices: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(
        self, indices: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(
        self,
        loss: str,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute the specified loss.

        Args:
            loss: the name of the loss to compute.
            outputs: the outputs from the model.
            targets: the targets.
            indices: the corresponding output/target indices from the matcher.
            num_boxes: the number of target boxes.
            kwargs: additional arguments to pass to the loss function.

        Returns:
            the loss dict.
        """
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(
        self, outputs: dict[str, Any], targets: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """This performs the loss computation.

        Args:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes])
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs = {"log": False}
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Apply weights.
        # We only keep the ones present in weight dict, since there may be others that
        # are only produced for logging purposes (not that we're logging them).
        final_losses = {
            k: loss * self.weight_dict[k]
            for k, loss in losses.items()
            if k in self.weight_dict
        }
        return final_losses


class PostProcess(nn.Module):
    """PostProcess converts the model output into the COCO format used by rslearn."""

    @torch.no_grad()
    def forward(
        self, outputs: dict[str, torch.Tensor], target_sizes: torch.Tensor
    ) -> list[dict[str, torch.Tensor]]:
        """Forward pass for PostProcess to perform the output format conversion.

        Args:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch.
                          For evaluation, this must be the original image size (before any data augmentation).
                          For visualization, this should be the image size after data augment, but before padding.
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [
            {"scores": cur_scores, "labels": cur_labels, "boxes": cur_boxes}
            for cur_scores, cur_labels, cur_boxes in zip(scores, labels, boxes)
        ]

        return results


class Detr(Predictor):
    """DETR prediction module.

    This combines PositionEmbeddingSine, DetrPredictor, SetCriterion, and PostProcess.

    This is the module that should be used as a decoder component in rslearn.
    """

    def __init__(self, predictor: DetrPredictor, criterion: SetCriterion):
        """Create a Detr.

        Args:
            predictor: the DetrPredictor.
            criterion: the SetCriterion.
        """
        super().__init__()
        self.predictor = predictor
        self.criterion = criterion
        self.pos_embedding = PositionEmbeddingSine(
            num_pos_feats=predictor.transformer.d_model // 2, normalize=True
        )
        self.postprocess = PostProcess()

        if predictor.aux_loss:
            # Hack to make sure it's included in the weight dict for the criterion.
            aux_weight_dict = {}
            num_dec_layers = len(predictor.transformer.decoder.layers)
            for i in range(num_dec_layers - 1):
                aux_weight_dict.update(
                    {f"{k}_{i}": v for k, v in self.criterion.weight_dict.items()}
                )
            self.criterion.weight_dict.update(aux_weight_dict)

    def forward(
        self,
        intermediates: Any,
        context: ModelContext,
        targets: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """Compute the detection outputs and loss from features.

        DETR will use only the last feature map, which should correspond to the lowest
        resolution one.

        Args:
            intermediates: the output from the previous component. It must be a FeatureMaps.
            context: the model context. Input dicts must contain an "image" key which we will
                be used to establish the original image size.
            targets: must contain class key that stores the class label.

        Returns:
            the model output.
        """
        if not isinstance(intermediates, FeatureMaps):
            raise ValueError("input to Detr must be a FeatureMaps")

        # We only use the last feature map (most fine-grained).
        features = intermediates.feature_maps[-1]

        # Get image sizes.
        image_sizes = torch.tensor(
            [[inp["image"].shape[2], inp["image"].shape[1]] for inp in context.inputs],
            dtype=torch.int32,
            device=features.device,
        )

        pos_embedding = self.pos_embedding(features)
        outputs = self.predictor(features, pos_embedding)

        if targets is not None:
            # Convert boxes from [x0, y0, x1, y1] to [cx, cy, w, h].
            converted_targets = []
            for target, image_size in zip(targets, image_sizes):
                boxes = target["boxes"]
                img_w, img_h = image_size
                scale_fct = torch.stack([img_w, img_h, img_w, img_h])
                boxes = boxes / scale_fct
                boxes = box_ops.box_xyxy_to_cxcywh(boxes)
                converted_targets.append(
                    {
                        "boxes": boxes,
                        "labels": target["labels"],
                    }
                )

            losses = self.criterion(outputs, converted_targets)
        else:
            losses = {}

        results = self.postprocess(outputs, image_sizes)

        return ModelOutput(
            outputs=results,
            loss_dict=losses,
        )
