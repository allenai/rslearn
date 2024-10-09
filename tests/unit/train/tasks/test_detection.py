import math

import numpy as np
import torch

from rslearn.train.tasks.detection import DetectionTask, F1Metric

EPSILON = 1e-4


def test_f1_metric():
    # Try 1 tp, 1 fp, 2 fn at best score threshold (0.4 F1).
    # (At 0.5 it should be 2 fp but at 0.9 it is 1 fp.)
    pred_dict = {
        "boxes": torch.tensor(
            [
                [0, 0, 10, 10],
                [100, 100, 110, 110],
                [200, 200, 210, 210],
            ],
            dtype=torch.float32,
        ),
        "scores": torch.tensor([0.9, 0.9, 0.5], dtype=torch.float32),
        "labels": torch.tensor([0, 0, 0], dtype=torch.int32),
    }
    gt_dict = {
        "boxes": torch.tensor(
            [
                [2, 2, 10, 10],
                [300, 300, 310, 310],
                [400, 400, 410, 410],
            ],
            dtype=torch.float32,
        ),
        "labels": torch.tensor([0, 0, 0], dtype=torch.int32),
    }

    metric = F1Metric(num_classes=1, cmp_mode="iou", cmp_threshold=0.5)
    metric.update([pred_dict], [gt_dict])
    f1 = metric.compute()
    assert abs(f1 - 0.4) < EPSILON

    # With stricter IoU threshold, we should get 0 tp.
    metric = F1Metric(num_classes=1, cmp_mode="iou", cmp_threshold=0.95)
    metric.update([pred_dict], [gt_dict])
    f1 = metric.compute()
    assert abs(f1 - 0) < EPSILON

    # Try distance threshold in same way (which compares centers).
    metric = F1Metric(num_classes=1, cmp_mode="distance", cmp_threshold=2)
    metric.update([pred_dict], [gt_dict])
    f1 = metric.compute()
    assert abs(f1 - 0.4) < EPSILON

    metric = F1Metric(num_classes=1, cmp_mode="distance", cmp_threshold=0.5)
    metric.update([pred_dict], [gt_dict])
    f1 = metric.compute()
    assert abs(f1 - 0) < EPSILON


def test_distance_nms():
    # Test NMS with 3 boxes, single class.
    boxes = torch.tensor(
        [
            [10, 10, 20, 20],  # Box 0
            [12, 12, 22, 22],  # Box 1, overlaps with Box 0
            [30, 30, 40, 40],  # Box 2, separate
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.9, 0.85, 0.8], dtype=torch.float32)
    class_ids = torch.tensor([0, 0, 0], dtype=torch.int32)
    grid_size = 10
    distance_threshold = 5

    detection = DetectionTask(None, None)
    keep_indices = detection._distance_nms(
        boxes.numpy(), scores.numpy(), class_ids.numpy(), grid_size, distance_threshold
    )

    # Expected: Box 0 (highest score) and Box 2 (no overlap) should be kept.
    expected = np.array([0, 2])
    assert np.array_equal(np.sort(keep_indices), np.sort(expected))

    # Test with a higher threshold where all boxes should be kept.
    distance_threshold = 2  # Smaller threshold to avoid suppression
    keep_indices = detection._distance_nms(
        boxes.numpy(), scores.numpy(), class_ids.numpy(), grid_size, distance_threshold
    )
    # Expected: All boxes should be kept.
    expected = np.array([0, 1, 2])
    assert np.array_equal(np.sort(keep_indices), np.sort(expected))

    # Test with no boxes provided.
    boxes = torch.tensor([], dtype=torch.float32).reshape(0, 4)
    scores = torch.tensor([], dtype=torch.float32)
    class_ids = torch.tensor([], dtype=torch.int32)
    grid_size = 10
    distance_threshold = 5

    keep_indices = detection._distance_nms(
        boxes.numpy(), scores.numpy(), class_ids.numpy(), grid_size, distance_threshold
    )
    # Expected: No boxes, so the result should be an empty array.
    expected = np.array([], dtype=int)
    assert np.array_equal(keep_indices, expected)

    # Test with multiple classes where NMS should be performed per class.
    boxes = torch.tensor(
        [
            [10, 10, 20, 20],  # Class 0, Box 0
            [12, 12, 22, 22],  # Class 0, Box 1 (overlapping with Box 0)
            [30, 30, 40, 40],  # Class 1, Box 2
            [32, 32, 42, 42],  # Class 1, Box 3 (overlapping with Box 2)
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.9, 0.85, 0.8, 0.95], dtype=torch.float32)
    class_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int32)  # Two classes (0 and 1)
    grid_size = 10
    distance_threshold = 5

    keep_indices = detection._distance_nms(
        boxes.numpy(), scores.numpy(), class_ids.numpy(), grid_size, distance_threshold
    )
    # Expected: For Class 0, Box 0 kept (higher score); Box 1 suppressed.
    # For Class 1, Box 3 kept (higher score); Box 2 suppressed.
    expected = np.array([0, 3])
    assert np.array_equal(np.sort(keep_indices), np.sort(expected))

    # Test with equal scores and overlapping boxes.
    boxes = torch.tensor(
        [
            [10, 10, 20, 20],  # Box 0
            [12, 12, 22, 22],  # Box 1 (overlapping with Box 0)
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.9, 0.9], dtype=torch.float32)  # Equal scores
    class_ids = torch.tensor([0, 0], dtype=torch.int32)  # Same class
    grid_size = 10
    distance_threshold = 5

    keep_indices = detection._distance_nms(
        boxes.numpy(), scores.numpy(), class_ids.numpy(), grid_size, distance_threshold
    )
    # Expected: Box 0 kept because it has a lower index (tie-breaking).
    expected = np.array([0])
    assert np.array_equal(np.sort(keep_indices), np.sort(expected))

    # Test with distance threshold edge case.
    boxes = torch.tensor(
        [
            [10, 10, 20, 20],  # Box 0 center at (15, 15)
            [20, 20, 30, 30],  # Box 1 center at (25, 25)
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
    class_ids = torch.tensor([0, 0], dtype=torch.int32)
    grid_size = 10
    distance_threshold = math.sqrt(200)  # Exact distance between centers

    keep_indices = detection._distance_nms(
        boxes.numpy(), scores.numpy(), class_ids.numpy(), grid_size, distance_threshold
    )
    # Expected: Box 0 should be kept, Box 1 eliminated due to tie-breaking on distance.
    expected = np.array([0])
    assert np.array_equal(keep_indices, expected)
