import torch

from rslearn.train.tasks.detection import F1Metric

EPSILON = 1e-4


def test_f1_metric() -> None:
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

    metric = F1Metric(
        num_classes=1, cmp_mode="iou", cmp_threshold=0.5, score_thresholds=[0.8]
    )
    metric.update([pred_dict], [gt_dict])
    f1 = metric.compute()
    assert abs(f1 - 0.4) < EPSILON

    metric = F1Metric(
        num_classes=1, cmp_mode="iou", cmp_threshold=0.5, score_thresholds=[0.4]
    )
    metric.update([pred_dict], [gt_dict])
    f1 = metric.compute()
    assert abs(f1 - 1 / 3) < EPSILON

    metric = F1Metric(
        num_classes=1, cmp_mode="iou", cmp_threshold=0.5, score_thresholds=[0.4, 0.8]
    )
    metric.update([pred_dict], [gt_dict])
    f1 = metric.compute()
    assert abs(f1 - 0.4) < EPSILON

    # With stricter IoU threshold, we should get 0 tp.
    metric = F1Metric(
        num_classes=1, cmp_mode="iou", cmp_threshold=0.95, score_thresholds=[0.8]
    )
    metric.update([pred_dict], [gt_dict])
    f1 = metric.compute()
    assert abs(f1 - 0) < EPSILON

    # Try distance threshold in same way (which compares centers).
    metric = F1Metric(
        num_classes=1, cmp_mode="distance", cmp_threshold=2, score_thresholds=[0.8]
    )
    metric.update([pred_dict], [gt_dict])
    f1 = metric.compute()
    assert abs(f1 - 0.4) < EPSILON

    metric = F1Metric(
        num_classes=1, cmp_mode="distance", cmp_threshold=0.5, score_thresholds=[0.8]
    )
    metric.update([pred_dict], [gt_dict])
    f1 = metric.compute()
    assert abs(f1 - 0) < EPSILON
