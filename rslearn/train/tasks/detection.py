"""Detection task."""

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torchmetrics.classification
from torchmetrics import Metric, MetricCollection

from rslearn.utils import Feature

from .task import BasicTask

DEFAULT_COLORS = [
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [0, 255, 255],
    [255, 0, 255],
    [0, 128, 0],
    [255, 160, 122],
    [139, 69, 19],
    [128, 128, 128],
    [255, 255, 255],
    [143, 188, 143],
    [95, 158, 160],
    [255, 200, 0],
    [128, 0, 0],
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
        **kwargs,
    ):
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
            class_labels = torch.zeros((1,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            class_labels = torch.as_tensor(class_labels, dtype=torch.int64)

        if self.skip_empty_examples and len(boxes) == 0:
            valid = 0

        return {}, {
            "valid": torch.tensor(valid, dtype=torch.int32),
            "boxes": boxes,
            "labels": class_labels,
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
        raise NotImplementedError

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
        metrics["mAP"] = DetectionMetric(
            torchmetrics.detection.mean_ap.MeanAveragePrecision()
        )
        return MetricCollection(metrics)


class DetectionMetric(Metric):
    """Metric for detection task."""

    def __init__(self, metric: Metric):
        """Initialize a new DetectionMetric."""
        super().__init__()
        self.metric = metric

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
        return self.metric.compute()

    def plot(self, *args: list[Any], **kwargs: dict[str, Any]) -> Any:
        """Returns a plot of the metric."""
        return self.metric.plot(*args, **kwargs)
