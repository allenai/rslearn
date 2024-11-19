"""Detection task."""

from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.optimize
import scipy.spatial
import shapely
import torch
import torchmetrics.classification
import torchvision
from torchmetrics import Metric, MetricCollection

from rslearn.utils import Feature, STGeometry

from .task import BasicTask

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


class DetectionTask(BasicTask):
    """A point or bounding box detection task."""

    def __init__(
        self,
        property_name: str,
        classes: list[str],
        filters: list[tuple[str, str]] | None = None,
        read_class_id: bool = False,
        skip_unknown_categories: bool = False,
        skip_empty_examples: bool = False,
        colors: list[tuple[int, int, int]] = DEFAULT_COLORS,
        box_size: int | None = None,
        clip_boxes: bool = True,
        exclude_by_center: bool = False,
        score_threshold: float = 0.5,
        enable_map_metric: bool = True,
        enable_f1_metric: bool = False,
        f1_metric_thresholds: list[list[float]] = [
            [
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                0.95,
            ]
        ],
        f1_metric_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> None:
        """Initialize a new SegmentationTask.

        Args:
            property_name: the property from which to extract the class name. The class
                is read from the first matching feature.
            classes: a list of class names.
            filters: optional list of (property_name, property_value) to only consider
                features with matching properties.
            read_class_id: whether to read an integer class ID instead of the class
                name.
            skip_unknown_categories: whether to skip examples with categories that are
                not passed via classes, instead of throwing error
            skip_empty_examples: whether to skip examples with zero labels.
            colors: optional colors for each class
            box_size: force all boxes to be this size, centered at the centroid of the
                geometry. Required for Point geometries.
            clip_boxes: whether to clip boxes to the image bounds.
            exclude_by_center: before optionally clipping boxes, exclude boxes if the
                center is outside the image bounds.
            score_threshold: confidence threshold for visualization and prediction.
            enable_map_metric: whether to compute mAP (default true)
            enable_f1_metric: whether to compute F1 (default false)
            f1_metric_thresholds: list of list of thresholds to apply for F1 metric.
                Each inner list is used to initialize a separate F1 metric where the
                best F1 across the thresholds within the inner list is computed. If
                there are multiple inner lists, then multiple F1 scores will be
                reported.
            f1_metric_kwargs: extra arguments to pass to F1 metric.
            kwargs: additional arguments to pass to BasicTask
        """
        super().__init__(**kwargs)
        self.property_name = property_name
        self.classes = classes
        self.filters = filters
        self.read_class_id = read_class_id
        self.skip_unknown_categories = skip_unknown_categories
        self.skip_empty_examples = skip_empty_examples
        self.colors = colors
        self.box_size = box_size
        self.clip_boxes = clip_boxes
        self.exclude_by_center = exclude_by_center
        self.score_threshold = score_threshold
        self.enable_map_metric = enable_map_metric
        self.enable_f1_metric = enable_f1_metric
        self.f1_metric_thresholds = f1_metric_thresholds
        self.f1_metric_kwargs = f1_metric_kwargs

        if not self.filters:
            self.filters = []

    def process_inputs(
        self,
        raw_inputs: dict[str, torch.Tensor | list[Feature]],
        metadata: dict[str, Any],
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

        boxes = []
        class_labels = []
        valid = 1

        data = raw_inputs["targets"]
        for feat in data:
            if feat.properties is None:
                continue
            if self.filters is not None:
                for property_name, property_value in self.filters:
                    if feat.properties.get(property_name) != property_value:
                        continue
            if self.property_name not in feat.properties:
                continue

            v = feat.properties[self.property_name]
            if self.read_class_id:
                class_id = int(v)
            else:
                if v in self.classes:
                    class_id = self.classes.index(v)
                else:
                    class_id = -1

            if class_id < 0 or class_id >= len(self.classes):
                # Throw error if unknown categories are not acceptable.
                assert self.skip_unknown_categories
                # Otherwise, skip this example.
                continue

            # Convert to relative coordinates for this patch.
            shp = feat.geometry.shp
            if self.box_size:
                box = [
                    int(shp.centroid.x) - self.box_size,
                    int(shp.centroid.y) - self.box_size,
                    int(shp.centroid.x) + self.box_size,
                    int(shp.centroid.y) + self.box_size,
                ]
            else:
                box = [int(val) for val in shp.bounds]

            if box[0] >= metadata["bounds"][2] or box[2] <= metadata["bounds"][0]:
                continue
            if box[1] >= metadata["bounds"][3] or box[3] <= metadata["bounds"][1]:
                continue

            if self.exclude_by_center:
                center_col = (box[0] + box[2]) // 2
                center_row = (box[1] + box[3]) // 2
                if (
                    center_col <= metadata["bounds"][0]
                    or center_col >= metadata["bounds"][2]
                ):
                    continue
                if (
                    center_row <= metadata["bounds"][1]
                    or center_row >= metadata["bounds"][3]
                ):
                    continue

            if self.clip_boxes:
                box = [
                    np.clip(box[0], metadata["bounds"][0], metadata["bounds"][2]),
                    np.clip(box[1], metadata["bounds"][1], metadata["bounds"][3]),
                    np.clip(box[2], metadata["bounds"][0], metadata["bounds"][2]),
                    np.clip(box[3], metadata["bounds"][1], metadata["bounds"][3]),
                ]

            # Convert to relative coordinates.
            box = [
                box[0] - metadata["bounds"][0],
                box[1] - metadata["bounds"][1],
                box[2] - metadata["bounds"][0],
                box[3] - metadata["bounds"][1],
            ]

            boxes.append(box)
            class_labels.append(class_id)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            class_labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            class_labels = torch.as_tensor(class_labels, dtype=torch.int64)

        if self.skip_empty_examples and len(boxes) == 0:
            valid = 0

        return {}, {
            "valid": torch.tensor(valid, dtype=torch.int32),
            "boxes": boxes,
            "labels": class_labels,
            "width": torch.tensor(
                metadata["bounds"][2] - metadata["bounds"][0], dtype=torch.float32
            ),
            "height": torch.tensor(
                metadata["bounds"][3] - metadata["bounds"][1], dtype=torch.float32
            ),
        }

    def process_output(
        self, raw_output: Any, metadata: dict[str, Any]
    ) -> npt.NDArray[Any] | list[Feature]:
        """Processes an output into raster or vector data.

        Args:
            raw_output: the output from prediction head.
            metadata: metadata about the patch being read

        Returns:
            either raster or vector data.
        """
        # Apply confidence threshold.
        wanted = raw_output["scores"].cpu().numpy() > self.score_threshold
        boxes = raw_output["boxes"].cpu().numpy()[wanted]
        class_ids = raw_output["labels"].cpu().numpy()[wanted]
        scores = raw_output["scores"].cpu().numpy()[wanted]

        features = []
        for box, class_id, score in zip(boxes, class_ids, scores):
            shp = shapely.box(
                metadata["bounds"][0] + float(box[0]),
                metadata["bounds"][1] + float(box[1]),
                metadata["bounds"][0] + float(box[2]),
                metadata["bounds"][1] + float(box[3]),
            )
            geom = STGeometry(metadata["projection"], shp, None)
            properties = {
                "score": float(score),
            }

            class_id = int(class_id)
            if self.read_class_id:
                properties[self.property_name] = class_id
            else:
                properties[self.property_name] = self.classes[class_id]

            features.append(Feature(geom, properties))

        return features

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

        def draw_boxes(
            image: npt.NDArray[Any], d: dict[str, torch.Tensor]
        ) -> npt.NDArray[Any]:
            boxes = d["boxes"].cpu().numpy()
            class_ids = d["labels"].cpu().numpy()
            if "scores" in d:
                wanted = d["scores"].cpu().numpy() > self.score_threshold
                boxes = boxes[wanted]
                class_ids = class_ids[wanted]

            for box, class_id in zip(boxes, class_ids):
                sx = int(np.clip(box[0], 0, image.shape[1]))
                sy = int(np.clip(box[1], 0, image.shape[0]))
                ex = int(np.clip(box[2], 0, image.shape[1]))
                ey = int(np.clip(box[3], 0, image.shape[0]))
                color = self.colors[class_id % len(self.colors)]
                image[sy:ey, sx : sx + 2, :] = color
                image[sy:ey, ex - 2 : ex, :] = color
                image[sy : sy + 2, sx:ex, :] = color
                image[ey - 2 : ey, sx:ex, :] = color

            return image

        if target_dict is None:
            raise ValueError("target_dict is required for visualization")
        return {
            "gt": draw_boxes(image.copy(), target_dict),
            "pred": draw_boxes(image.copy(), output),
        }

    def get_metrics(self) -> MetricCollection:
        """Get the metrics for this task."""
        metrics = {}

        if self.enable_map_metric:
            metrics["mAP"] = DetectionMetric(
                torchmetrics.detection.mean_ap.MeanAveragePrecision(),
                output_key="map",
            )

        if self.enable_f1_metric:
            kwargs = dict(
                num_classes=len(self.classes),
            )
            kwargs.update(self.f1_metric_kwargs)

            for thresholds in self.f1_metric_thresholds:
                if len(self.f1_metric_thresholds) == 1:
                    suffix = ""
                else:
                    # Metric name can't contain "." so change to ",".
                    suffix = "_" + str(thresholds[0]).replace(".", ",")

                metrics["F1" + suffix] = DetectionMetric(
                    F1Metric(score_thresholds=thresholds, **kwargs)  # type: ignore
                )
                metrics["precision" + suffix] = DetectionMetric(
                    F1Metric(
                        score_thresholds=thresholds,
                        metric_mode="precision",
                        **kwargs,  # type: ignore
                    )
                )
                metrics["recall" + suffix] = DetectionMetric(
                    F1Metric(
                        score_thresholds=thresholds,
                        metric_mode="recall",
                        **kwargs,  # type: ignore
                    )
                )

        return MetricCollection(metrics)


class DetectionMetric(Metric):
    """Metric for detection task."""

    def __init__(self, metric: Metric, output_key: str | None = None):
        """Initialize a new DetectionMetric.

        Args:
            metric: the metric to wrap.
            output_key: in case the metric returns a dict, return this element of the
                dict. Leave None if metric returns scalar.
        """
        super().__init__()
        self.metric = metric
        self.output_key = output_key

    def update(
        self, preds: list[dict[str, Any]], targets: list[dict[str, Any]]
    ) -> None:
        """Update metric.

        Args:
            preds: the predictions
            targets: the targets
        """
        new_preds = []
        new_targets = []
        for pred, target in zip(preds, targets):
            if not target["valid"]:
                continue
            new_preds.append(pred)
            new_targets.append(target)
        self.metric.update(new_preds, new_targets)

    def compute(self) -> Any:
        """Returns the computed metric."""
        val = self.metric.compute()
        if self.output_key:
            val = val[self.output_key]
        return val

    def reset(self) -> None:
        """Reset metric."""
        super().reset()
        self.metric.reset()

    def plot(self, *args: list[Any], **kwargs: dict[str, Any]) -> Any:
        """Returns a plot of the metric."""
        return self.metric.plot(*args, **kwargs)


class F1Metric(Metric):
    """F1 score for object detection."""

    def __init__(
        self,
        num_classes: int,
        score_thresholds: list[float],
        cmp_mode: str = "iou",
        cmp_threshold: float = 0.5,
        flatten_classes: bool = False,
        metric_mode: str = "f1",
    ):
        """Create a new F1Metric.

        Args:
            num_classes: number of classes.
            cmp_mode: how to compare boxes, either "iou" or "distance".
            cmp_threshold: similarity threshold, i.e. min IoU or max distance to
                consider two boxes as matching.
            score_thresholds: list of score thresholds to check F1 score for. The final
                metric is the best F1 across score thresholds.
            flatten_classes: sum true positives, false positives, and false negatives
                across classes and report combined F1 instead of computing F1 score for
                each class and then reporting the average.
            metric_mode: set to "precision" or "recall" to return that instead of F1
                (default "f1")
        """
        super().__init__()
        self.num_classes = num_classes
        self.cmp_mode = cmp_mode
        self.cmp_threshold = cmp_threshold
        self.score_thresholds = score_thresholds
        self.flatten_classes = flatten_classes
        self.metric_mode = metric_mode

        assert self.cmp_mode in ["iou", "distance"]
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
        if self.flatten_classes:
            cls_idx = 0
        return f"{cls_idx}_{thr_idx}_"

    def _single_update(self, pred: dict[str, Any], target: dict[str, Any]) -> None:
        for cls_idx in range(self.num_classes):
            # Get ground truth boxes for this class.
            gt_boxes = target["boxes"][target["labels"] == cls_idx]

            for thr_idx, score_threshold in enumerate(self.score_thresholds):
                # Get predicted boxes for this class under the current score threshold.
                selector = (pred["scores"] >= score_threshold) & (
                    pred["labels"] == cls_idx
                )
                pred_boxes = pred["boxes"][selector, :]

                # Compute comparison scores.
                if self.cmp_mode == "iou":
                    ious = torchvision.ops.box_iou(gt_boxes, pred_boxes)
                    cmp_result = ious.cpu().numpy() >= self.cmp_threshold

                elif self.cmp_mode == "distance":

                    def get_centers(boxes: torch.Tensor) -> torch.Tensor:
                        return torch.stack(
                            [
                                (boxes[:, 0] + boxes[:, 2]) / 2,
                                (boxes[:, 1] + boxes[:, 3]) / 2,
                            ],
                            dim=1,
                        )

                    gt_centers = get_centers(gt_boxes)
                    pred_centers = get_centers(pred_boxes)
                    distances = scipy.spatial.distance_matrix(
                        gt_centers.cpu().numpy(), pred_centers.cpu().numpy()
                    )
                    cmp_result = distances <= self.cmp_threshold

                # Using Hungarian matching algorithm to assign lowest-cost gt-pred pairs.
                rows, cols = scipy.optimize.linear_sum_assignment(
                    1 - cmp_result.astype(np.float32)
                )
                matches = cmp_result[rows, cols]
                tp = np.count_nonzero(matches)
                fp = len(pred_boxes) - tp
                fn = len(gt_boxes) - tp

                cur_prefix = self._get_state_prefix(cls_idx, thr_idx)
                setattr(self, cur_prefix + "tp", getattr(self, cur_prefix + "tp") + tp)
                setattr(self, cur_prefix + "fp", getattr(self, cur_prefix + "fp") + fp)
                setattr(self, cur_prefix + "fn", getattr(self, cur_prefix + "fn") + fn)

    def update(
        self, preds: list[dict[str, Any]], targets: list[dict[str, Any]]
    ) -> None:
        """Update metric.

        Args:
            preds: the predictions
            targets: the targets
        """
        for pred, target in zip(preds, targets):
            self._single_update(pred, target)

    def compute(self) -> Any:
        """Compute metric.

        Returns:
            the best F1 score across score thresholds and classes.
        """
        best_scores = []

        if self.flatten_classes:
            classes_to_check = 1
        else:
            classes_to_check = self.num_classes

        for cls_idx in range(classes_to_check):
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

            best_scores.append(best_score)

        return torch.mean(torch.stack(best_scores))
