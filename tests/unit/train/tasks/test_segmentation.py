import numpy as np
import torch

from rslearn.train.tasks.segmentation import SegmentationTask


class TestProcessOutput:
    """Unit tests relating to SegmentationTask.process_output."""

    def test_with_prob_scales(self) -> None:
        """Ensure that prob_scales is reflected correctly in process_output."""
        task = SegmentationTask(num_classes=3, prob_scales=[1.0, 1.0, 3.0])
        out_probs = torch.tensor([[[0.3]], [[0.5]], [[0.2]]], dtype=torch.float32)
        classes = task.process_output(out_probs, {})
        # Output class should be 2 since the 0.2 probability is multiplied by 3 before
        # computing argmax.
        assert isinstance(classes, np.ndarray)
        assert classes.shape == (1, 1, 1)
        assert classes[0][0] == 2

    def test_no_prob_scales(self) -> None:
        """Test process_output without prob_scales."""
        task = SegmentationTask(num_classes=3)
        out_probs = torch.tensor([[[0.3]], [[0.5]], [[0.2]]], dtype=torch.float32)
        classes = task.process_output(out_probs, {})
        assert isinstance(classes, np.ndarray)
        assert classes.shape == (1, 1, 1)
        assert classes[0][0] == 1
