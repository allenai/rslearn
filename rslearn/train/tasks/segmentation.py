"""Segmentation task."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torchmetrics.classification
from torchmetrics import Metric, MetricCollection

from rslearn.models.component import FeatureMaps, Predictor
from rslearn.train.model_context import (
    ModelContext,
    ModelOutput,
    RasterImage,
    SampleMetadata,
)
from rslearn.utils import Feature

from .task import BasicTask

# TODO: This is duplicated code fix it
DEFAULT_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (0, 128, 0),
    (255, 160, 122),
    (139, 69, 19),
    (128, 128, 128),
    (255, 255, 255),
    (143, 188, 143),
    (95, 158, 160),
    (255, 200, 0),
    (128, 0, 0),
]


class SegmentationTask(BasicTask):
    """A segmentation (per-pixel classification) task."""

    def __init__(
        self,
        num_classes: int,
        class_id_mapping: dict[int, int] | None = None,
        colors: list[tuple[int, int, int]] = DEFAULT_COLORS,
        zero_is_invalid: bool = False,
        nodata_value: int | None = None,
        enable_accuracy_metric: bool = True,
        enable_miou_metric: bool = False,
        enable_f1_metric: bool = False,
        report_metric_per_class: bool = False,
        f1_metric_thresholds: list[list[float]] = [[0.5]],
        metric_kwargs: dict[str, Any] = {},
        miou_metric_kwargs: dict[str, Any] = {},
        prob_scales: list[float] | None = None,
        other_metrics: dict[str, Metric] = {},
        output_probs: bool = False,
        output_class_idx: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a new SegmentationTask.

        Args:
            num_classes: the number of classes to predict
            colors: optional colors for each class
            zero_is_invalid: whether pixels labeled class 0 should be marked invalid
                Mutually exclusive with nodata_value.
            nodata_value: the value to use for nodata pixels. If None, all pixels are
                considered valid. Mutually exclusive with zero_is_invalid.
            class_id_mapping: optional mapping from original class IDs to new class IDs.
                If provided, class labels will be remapped according to this dictionary.
            enable_accuracy_metric: whether to enable the accuracy metric (default
                true).
            enable_f1_metric: whether to enable the F1 metric (default false).
            report_metric_per_class: whether to report chosen metrics for each class, in
                addition to the average score across classes.
            enable_miou_metric: whether to enable the mean IoU metric (default false).
            f1_metric_thresholds: list of list of thresholds to apply for F1 metric.
                Each inner list is used to initialize a separate F1 metric where the
                best F1 across the thresholds within the inner list is computed. If
                there are multiple inner lists, then multiple F1 scores will be
                reported.
            metric_kwargs: additional arguments to pass to underlying metric, see
                torchmetrics.classification.MulticlassAccuracy.
            miou_metric_kwargs: additional arguments to pass to MeanIoUMetric, if
                enable_miou_metric is passed.
            prob_scales: during inference, scale the output probabilities by this much
                before computing the argmax. There is one scale per class. Note that
                this is only applied during prediction, not when computing val or test
                metrics.
            other_metrics: additional metrics to configure on this task.
            output_probs: if True, output raw softmax probabilities instead of class IDs
                during prediction.
            output_class_idx: if set along with output_probs, only output the probability
                for this specific class index (single-channel output).
            kwargs: additional arguments to pass to BasicTask
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.class_id_mapping = class_id_mapping
        self.colors = colors
        self.nodata_value: int | None

        if zero_is_invalid and nodata_value is not None:
            raise ValueError("zero_is_invalid and nodata_value cannot both be set")
        if zero_is_invalid:
            self.nodata_value = 0
        else:
            self.nodata_value = nodata_value

        self.enable_accuracy_metric = enable_accuracy_metric
        self.enable_f1_metric = enable_f1_metric
        self.enable_miou_metric = enable_miou_metric
        self.report_metric_per_class = report_metric_per_class
        self.f1_metric_thresholds = f1_metric_thresholds
        self.metric_kwargs = metric_kwargs
        self.miou_metric_kwargs = miou_metric_kwargs
        self.prob_scales = prob_scales
        self.other_metrics = other_metrics
        self.output_probs = output_probs
        self.output_class_idx = output_class_idx

    def process_inputs(
        self,
        raw_inputs: Mapping[str, RasterImage | list[Feature]],
        metadata: SampleMetadata,
        load_targets: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Processes the data into targets.

        Args:
            raw_inputs: raster or vector data to process
            metadata: metadata about the patch being read
            load_targets: whether to load the targets or only inputs

        Returns:
            tuple (input_dict, target_dict) containing the processed inputs and targets
                that are compatible with both metrics and loss functions
        """
        if not load_targets:
            return {}, {}

        assert isinstance(raw_inputs["targets"], RasterImage)
        assert raw_inputs["targets"].image.shape[0] == 1
        assert raw_inputs["targets"].image.shape[1] == 1
        labels = raw_inputs["targets"].image[0, 0, :, :].long()

        if self.class_id_mapping is not None:
            new_labels = labels.clone()
            for old_id, new_id in self.class_id_mapping.items():
                new_labels[labels == old_id] = new_id
            labels = new_labels

        if self.nodata_value is not None:
            valid = (labels != self.nodata_value).float()
            # Labels, even masked ones, must be in the range 0 to num_classes-1
            if self.nodata_value >= self.num_classes:
                labels[labels == self.nodata_value] = 0
        else:
            valid = torch.ones(labels.shape, dtype=torch.float32)

        return {}, {
            "classes": labels,
            "valid": valid,
        }

    def process_output(
        self, raw_output: Any, metadata: SampleMetadata
    ) -> npt.NDArray[Any]:
        """Processes an output into raster or vector data.

        Args:
            raw_output: the output from prediction head, which must be a CHW tensor.
            metadata: metadata about the patch being read

        Returns:
            CHW numpy array. If output_probs is False, returns one channel with
            predicted class IDs. If output_probs is True, returns softmax probabilities
            (num_classes channels, or 1 channel if output_class_idx is set).
        """
        if not isinstance(raw_output, torch.Tensor) or len(raw_output.shape) != 3:
            raise ValueError("the output for SegmentationTask must be a CHW tensor")

        if self.prob_scales is not None:
            raw_output = (
                raw_output
                * torch.tensor(
                    self.prob_scales, device=raw_output.device, dtype=raw_output.dtype
                )[:, None, None]
            )

        if self.output_probs:
            # Return raw softmax probabilities
            probs = raw_output.cpu().numpy()
            if self.output_class_idx is not None:
                # Return only the specified class probability
                return probs[self.output_class_idx : self.output_class_idx + 1, :, :]
            return probs

        classes = raw_output.argmax(dim=0).cpu().numpy()
        return classes[None, :, :]

    def visualize(
        self,
        input_dict: dict[str, Any],
        target_dict: dict[str, Any] | None,
        output: Any,
    ) -> dict[str, npt.NDArray[Any]]:
        """Visualize the outputs and targets.

        Args:
            input_dict: the input dict from process_inputs
            target_dict: the target dict from process_inputs
            output: the prediction

        Returns:
            a dictionary mapping image name to visualization image
        """
        image = super().visualize(input_dict, target_dict, output)["image"]
        if target_dict is None:
            raise ValueError("target_dict is required for visualization")
        gt_classes = target_dict["classes"].cpu().numpy()
        pred_classes = output.cpu().numpy().argmax(axis=0)
        gt_vis = np.zeros((gt_classes.shape[0], gt_classes.shape[1], 3), dtype=np.uint8)
        pred_vis = np.zeros(
            (pred_classes.shape[0], pred_classes.shape[1], 3), dtype=np.uint8
        )
        for class_id in range(self.num_classes):
            color = self.colors[class_id % len(self.colors)]
            gt_vis[gt_classes == class_id] = color
            pred_vis[pred_classes == class_id] = color

        return {
            "image": np.array(image),
            "gt": gt_vis,
            "pred": pred_vis,
        }

    def get_metrics(self) -> MetricCollection:
        """Get the metrics for this task."""
        metrics = {}

        if self.enable_accuracy_metric:
            accuracy_metric_kwargs = dict(num_classes=self.num_classes)
            accuracy_metric_kwargs.update(self.metric_kwargs)
            metrics["accuracy"] = SegmentationMetric(
                torchmetrics.classification.MulticlassAccuracy(**accuracy_metric_kwargs)
            )

        if self.enable_f1_metric:
            for thresholds in self.f1_metric_thresholds:
                if len(self.f1_metric_thresholds) == 1:
                    suffix = ""
                else:
                    # Metric name can't contain "." so change to ",".
                    suffix = "_" + str(thresholds[0]).replace(".", ",")

                # Create one metric per type - it returns a dict with "avg" and optionally per-class keys
                metrics["F1" + suffix] = SegmentationMetric(
                    F1Metric(
                        num_classes=self.num_classes,
                        score_thresholds=thresholds,
                        report_per_class=self.report_metric_per_class,
                    ),
                )
                metrics["precision" + suffix] = SegmentationMetric(
                    F1Metric(
                        num_classes=self.num_classes,
                        score_thresholds=thresholds,
                        metric_mode="precision",
                        report_per_class=self.report_metric_per_class,
                    ),
                )
                metrics["recall" + suffix] = SegmentationMetric(
                    F1Metric(
                        num_classes=self.num_classes,
                        score_thresholds=thresholds,
                        metric_mode="recall",
                        report_per_class=self.report_metric_per_class,
                    ),
                )

        if self.enable_miou_metric:
            miou_metric_kwargs: dict[str, Any] = dict(
                num_classes=self.num_classes,
                report_per_class=self.report_metric_per_class,
            )
            if self.nodata_value is not None:
                miou_metric_kwargs["nodata_value"] = self.nodata_value
            miou_metric_kwargs.update(self.miou_metric_kwargs)

            # Create one metric - it returns a dict with "avg" and optionally per-class keys
            metrics["mean_iou"] = SegmentationMetric(
                MeanIoUMetric(**miou_metric_kwargs),
                pass_probabilities=False,
            )

        if self.other_metrics:
            metrics.update(self.other_metrics)

        return MetricCollection(metrics)


class SegmentationHead(Predictor):
    """Head for segmentation task."""

    def __init__(self, weights: list[float] | None = None, dice_loss: bool = False):
        """Initialize a new SegmentationTask.

        Args:
            weights: weights for cross entropy loss (Tensor of size C)
            dice_loss: weather to add dice loss to cross entropy
        """
        super().__init__()
        if weights is not None:
            self.register_buffer("weights", torch.Tensor(weights))
        else:
            self.weights = None
        self.dice_loss = dice_loss

    def forward(
        self,
        intermediates: Any,
        context: ModelContext,
        targets: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """Compute the segmentation outputs from logits and targets.

        Args:
            intermediates: a FeatureMaps with a single feature map containing the
                segmentation logits.
            context: the model context
            targets: list of target dicts, where each target dict must contain a key
                "classes" containing the per-pixel class labels, along with "valid"
                containing a mask indicating where the example is valid.

        Returns:
            tuple of outputs and loss dict
        """
        if not isinstance(intermediates, FeatureMaps):
            raise ValueError("input to SegmentationHead must be a FeatureMaps")
        if len(intermediates.feature_maps) != 1:
            raise ValueError(
                f"input to SegmentationHead must have one feature map, but got {len(intermediates.feature_maps)}"
            )

        logits = intermediates.feature_maps[0]
        outputs = torch.nn.functional.softmax(logits, dim=1)

        losses = {}
        if targets:
            labels = torch.stack([target["classes"] for target in targets], dim=0)
            mask = torch.stack([target["valid"] for target in targets], dim=0)
            per_pixel_loss = torch.nn.functional.cross_entropy(
                logits, labels, weight=self.weights, reduction="none"
            )
            mask_sum = torch.sum(mask)
            if mask_sum > 0:
                # Compute average loss over valid pixels.
                losses["cls"] = torch.sum(per_pixel_loss * mask) / torch.sum(mask)
            else:
                # If there are no valid pixels, we avoid dividing by zero and just let
                # the summed mask loss be zero.
                losses["cls"] = torch.sum(per_pixel_loss * mask)
            if self.dice_loss:
                dice_loss = DiceLoss()(outputs, labels, mask)
                losses["dice"] = dice_loss

        return ModelOutput(
            outputs=outputs,
            loss_dict=losses,
        )


class SegmentationMetric(Metric):
    """Metric for segmentation task."""

    def __init__(
        self,
        metric: Metric,
        pass_probabilities: bool = True,
        class_idx: int | None = None,
        output_key: str | None = None,
    ):
        """Initialize a new SegmentationMetric.

        Args:
            metric: the metric to wrap. This wrapping class will handle selecting the
                classes from the targets and masking out invalid pixels.
            pass_probabilities: whether to pass predicted probabilities to the metric.
                If False, argmax is applied to pass the predicted classes instead.
            class_idx: if set, return only this class index's value. For backward
                compatibility with configs using standard torchmetrics. Internally
                converted to output_key="cls_{class_idx}".
            output_key: if the wrapped metric returns a dict (or a tensor that gets
                converted to a dict), return only this key's value. For standard
                torchmetrics with average=None, tensors are converted to dicts with
                keys "cls_0", "cls_1", etc. If None, the full dict is returned.
        """
        super().__init__()
        self.metric = metric
        self.pass_probablities = pass_probabilities
        self.class_idx = class_idx
        self.output_key = output_key

    def update(
        self, preds: list[Any] | torch.Tensor, targets: list[dict[str, Any]]
    ) -> None:
        """Update metric.

        Args:
            preds: the predictions
            targets: the targets
        """
        if not isinstance(preds, torch.Tensor):
            preds = torch.stack(preds)
        labels = torch.stack([target["classes"] for target in targets])

        # Sub-select the valid labels.
        # We flatten the prediction and label images at valid pixels.
        # Prediction is changed from BCHW to BHWC so we can select the valid BHW mask.
        mask = torch.stack([target["valid"] > 0 for target in targets])
        preds = preds.permute(0, 2, 3, 1)[mask]
        labels = labels[mask]
        if len(preds) == 0:
            return

        if not self.pass_probablities:
            preds = preds.argmax(dim=1)

        self.metric.update(preds, labels)

    def compute(self) -> Any:
        """Returns the computed metric.

        If the wrapped metric returns a multi-element tensor (e.g., standard torchmetrics
        with average=None), it is converted to a dict with keys like "cls_0", "cls_1", etc.
        This allows uniform handling via output_key for both standard torchmetrics and
        custom dict-returning metrics.
        """
        result = self.metric.compute()

        # Convert multi-element tensors to dict for uniform handling.
        # This supports standard torchmetrics with average=None which return per-class tensors.
        if isinstance(result, torch.Tensor) and result.ndim >= 1:
            result = {f"cls_{i}": result[i] for i in range(len(result))}

        if self.output_key is not None:
            if not isinstance(result, dict):
                raise TypeError(
                    f"output_key is set to '{self.output_key}' but metric returned "
                    f"{type(result).__name__} instead of dict"
                )
            return result[self.output_key]
        if self.class_idx is not None:
            # For backward compatibility: class_idx can index into the converted dict
            if isinstance(result, dict):
                return result[f"cls_{self.class_idx}"]
            return result[self.class_idx]
        return result

    def reset(self) -> None:
        """Reset metric."""
        super().reset()
        self.metric.reset()

    def plot(self, *args: list[Any], **kwargs: dict[str, Any]) -> Any:
        """Returns a plot of the metric."""
        return self.metric.plot(*args, **kwargs)


class F1Metric(Metric):
    """F1 score for segmentation.

    It treats each class as a separate prediction task, and computes the maximum F1
    score under the different configured thresholds per-class.
    """

    def __init__(
        self,
        num_classes: int,
        score_thresholds: list[float],
        metric_mode: str = "f1",
        report_per_class: bool = False,
    ):
        """Create a new F1Metric.

        Args:
            num_classes: number of classes.
            score_thresholds: list of score thresholds to check F1 score for. The final
                metric is the best F1 across score thresholds.
            metric_mode: set to "precision" or "recall" to return that instead of F1
                (default "f1")
            report_per_class: whether to include per-class scores in the output dict.
                If False, only returns the "avg" key.
        """
        super().__init__()
        self.num_classes = num_classes
        self.score_thresholds = score_thresholds
        self.metric_mode = metric_mode
        self.report_per_class = report_per_class

        assert self.metric_mode in ["f1", "precision", "recall"]

        for cls_idx in range(self.num_classes):
            for thr_idx in range(len(self.score_thresholds)):
                cur_prefix = self._get_state_prefix(cls_idx, thr_idx)
                self.add_state(
                    cur_prefix + "tp", default=torch.tensor(0), dist_reduce_fx="sum"
                )
                self.add_state(
                    cur_prefix + "fp", default=torch.tensor(0), dist_reduce_fx="sum"
                )
                self.add_state(
                    cur_prefix + "fn", default=torch.tensor(0), dist_reduce_fx="sum"
                )

    def _get_state_prefix(self, cls_idx: int, thr_idx: int) -> str:
        return f"{cls_idx}_{thr_idx}_"

    def update(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        """Update metric.

        Args:
            preds: the predictions, NxC.
            labels: the targets, N, with values from 0 to C-1.
        """
        for cls_idx in range(self.num_classes):
            for thr_idx, score_threshold in enumerate(self.score_thresholds):
                pred_bin = preds[:, cls_idx] > score_threshold
                gt_bin = labels == cls_idx

                tp = torch.count_nonzero(pred_bin & gt_bin).item()
                fp = torch.count_nonzero(pred_bin & torch.logical_not(gt_bin)).item()
                fn = torch.count_nonzero(torch.logical_not(pred_bin) & gt_bin).item()

                cur_prefix = self._get_state_prefix(cls_idx, thr_idx)
                setattr(self, cur_prefix + "tp", getattr(self, cur_prefix + "tp") + tp)
                setattr(self, cur_prefix + "fp", getattr(self, cur_prefix + "fp") + fp)
                setattr(self, cur_prefix + "fn", getattr(self, cur_prefix + "fn") + fn)

    def compute(self) -> Any:
        """Compute metric.

        Returns:
            dict with "avg" key containing mean score across classes.
            If report_per_class is True, also includes "cls_N" keys for each class N.
        """
        cls_best_scores = {}

        for cls_idx in range(self.num_classes):
            best_score = None

            for thr_idx in range(len(self.score_thresholds)):
                cur_prefix = self._get_state_prefix(cls_idx, thr_idx)
                tp = getattr(self, cur_prefix + "tp")
                fp = getattr(self, cur_prefix + "fp")
                fn = getattr(self, cur_prefix + "fn")
                device = tp.device

                if tp + fp == 0:
                    precision = torch.tensor(0, dtype=torch.float32, device=device)
                else:
                    precision = tp / (tp + fp)

                if tp + fn == 0:
                    recall = torch.tensor(0, dtype=torch.float32, device=device)
                else:
                    recall = tp / (tp + fn)

                if precision + recall < 0.001:
                    f1 = torch.tensor(0, dtype=torch.float32, device=device)
                else:
                    f1 = 2 * precision * recall / (precision + recall)

                if self.metric_mode == "f1":
                    score = f1
                elif self.metric_mode == "precision":
                    score = precision
                elif self.metric_mode == "recall":
                    score = recall

                if best_score is None or score > best_score:
                    best_score = score

            cls_best_scores[f"cls_{cls_idx}"] = best_score

        report_scores = {"avg": torch.mean(torch.stack(list(cls_best_scores.values())))}
        if self.report_per_class:
            report_scores.update(cls_best_scores)
        return report_scores


class MeanIoUMetric(Metric):
    """Mean IoU for segmentation.

    This is the mean of the per-class intersection-over-union scores. The per-class
    intersection is the number of pixels across all examples where the predicted label
    and ground truth label are both that class, and the per-class union is defined
    similarly.

    This differs from torchmetrics.segmentation.MeanIoU, where the mean IoU is computed
    per-image, and averaged across images.
    """

    def __init__(
        self,
        num_classes: int,
        nodata_value: int | None = None,
        ignore_missing_classes: bool = False,
        report_per_class: bool = False,
    ):
        """Create a new MeanIoUMetric.

        Args:
            num_classes: the number of classes for the task.
            nodata_value: the value to treat as nodata/invalid. If set and is one of the
                classes, IoU will not be calculated for it. If None, or not one of the
                classes, IoU is calculated for all classes.
            ignore_missing_classes: whether to ignore classes that don't appear in
                either the predictions or the ground truth. If false, the IoU for a
                missing class will be 0.
            report_per_class: whether to include per-class IoU scores in the output dict.
                If False, only returns the "avg" key.
        """
        super().__init__()
        self.num_classes = num_classes
        self.nodata_value = nodata_value
        self.ignore_missing_classes = ignore_missing_classes
        self.report_per_class = report_per_class

        self.add_state(
            "intersections", default=torch.zeros(self.num_classes), dist_reduce_fx="sum"
        )
        self.add_state(
            "unions", default=torch.zeros(self.num_classes), dist_reduce_fx="sum"
        )

    def update(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        """Update metric.

        Like torchmetrics.segmentation.MeanIoU with input_format="index", we expect
        predictions and labels to both be class integers. This is achieved by passing
        pass_probabilities=False to the SegmentationMetric wrapper.

        Args:
            preds: the predictions, (N,), with values from 0 to C-1.
            labels: the targets, (N,), with values from 0 to C-1.
        """
        if preds.min() < 0 or preds.max() >= self.num_classes:
            raise ValueError("predicted class outside of expected range")
        if labels.min() < 0 or labels.max() >= self.num_classes:
            raise ValueError("label class outside of expected range")

        new_intersections = torch.zeros(
            self.num_classes, device=self.intersections.device
        )
        new_unions = torch.zeros(self.num_classes, device=self.unions.device)
        for cls_idx in range(self.num_classes):
            new_intersections[cls_idx] = (
                (preds == cls_idx) & (labels == cls_idx)
            ).sum()
            new_unions[cls_idx] = ((preds == cls_idx) | (labels == cls_idx)).sum()
        self.intersections += new_intersections
        self.unions += new_unions

    def compute(self) -> Any:
        """Compute metric.

        Returns:
            dict with "avg" containing the mean IoU across classes.
            If report_per_class is True, also includes "cls_N" keys for each valid class N.
        """
        cls_scores = {}
        valid_scores = []

        for cls_idx in range(self.num_classes):
            # Check if nodata_value is set and is one of the classes
            if self.nodata_value is not None and cls_idx == self.nodata_value:
                continue

            intersection = self.intersections[cls_idx]
            union = self.unions[cls_idx]

            if union == 0 and self.ignore_missing_classes:
                continue

            score = intersection / union
            cls_scores[f"cls_{cls_idx}"] = score
            valid_scores.append(score)

        report_scores = {"avg": torch.mean(torch.stack(valid_scores))}
        if self.report_per_class:
            report_scores.update(cls_scores)
        return report_scores


class DiceLoss(torch.nn.Module):
    """Mean Dice Loss for segmentation.

    This is the mean of the per-class dice loss (1 - 2*intersection / union scores).
    The per-class intersection is the number of pixels across all examples where
    the predicted label and ground truth label are both that class, and the per-class
    union is defined similarly.
    """

    def __init__(self, smooth: float = 1e-7):
        """Initialize a new DiceLoss."""
        super().__init__()
        self.smooth = smooth

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute Dice Loss.

        Returns:
            the mean Dicen Loss across classes
        """
        num_classes = inputs.shape[1]
        targets_one_hot = (
            torch.nn.functional.one_hot(targets, num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )

        # Expand mask to [B, C, H, W]
        mask = mask.unsqueeze(1).expand_as(inputs)

        dice_per_class = []
        for c in range(num_classes):
            pred_c = inputs[:, c] * mask[:, c]
            target_c = targets_one_hot[:, c] * mask[:, c]

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice_c = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_per_class.append(dice_c)

        return 1 - torch.stack(dice_per_class).mean()
