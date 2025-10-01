import numpy as np
import pytest
import torch

from rslearn.train.tasks.segmentation import SegmentationTask


class TestProcessInputs:
    """Unit tests relating to SegmentationTask.process_inputs."""

    def test_zero_is_invalid_true(self) -> None:
        """Test process_inputs with zero_is_invalid=True."""
        task = SegmentationTask(num_classes=3, zero_is_invalid=True)

        # Create test data with 0s (invalid) and other values
        raw_inputs = {
            "targets": torch.tensor(
                [[[0, 1, 2], [0, 1, 0], [2, 1, 0]]], dtype=torch.uint8
            )
        }

        _, target_dict = task.process_inputs(raw_inputs, {})

        # Check classes tensor (should be unchanged)
        expected_classes = torch.tensor(
            [[0, 1, 2], [0, 1, 0], [2, 1, 0]], dtype=torch.long
        )
        assert torch.equal(target_dict["classes"], expected_classes)

        # Check valid mask (0s should be invalid, others valid)
        expected_valid = torch.tensor(
            [[0.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=torch.float32
        )
        assert torch.equal(target_dict["valid"], expected_valid)

    def test_zero_is_invalid_false(self) -> None:
        """Test process_inputs with zero_is_invalid=False."""
        task = SegmentationTask(num_classes=3, zero_is_invalid=False)

        # Create test data with 0s and other values
        raw_inputs = {
            "targets": torch.tensor(
                [[[0, 1, 2], [0, 1, 0], [2, 1, 0]]], dtype=torch.uint8
            )
        }

        _, target_dict = task.process_inputs(raw_inputs, {})

        # Check classes tensor (should be unchanged)
        expected_classes = torch.tensor(
            [[0, 1, 2], [0, 1, 0], [2, 1, 0]], dtype=torch.long
        )
        assert torch.equal(target_dict["classes"], expected_classes)

        # Check valid mask (all should be valid)
        expected_valid = torch.ones((3, 3), dtype=torch.float32)
        assert torch.equal(target_dict["valid"], expected_valid)

    def test_nodata_value_none(self) -> None:
        """Test process_inputs with nodata_value=None."""
        task = SegmentationTask(num_classes=3, nodata_value=None)

        # Create test data
        raw_inputs = {
            "targets": torch.tensor(
                [[[0, 1, 2], [0, 1, 0], [2, 1, 0]]], dtype=torch.uint8
            )
        }

        _, target_dict = task.process_inputs(raw_inputs, {})

        # Check classes tensor (should be unchanged)
        expected_classes = torch.tensor(
            [[0, 1, 2], [0, 1, 0], [2, 1, 0]], dtype=torch.long
        )
        assert torch.equal(target_dict["classes"], expected_classes)

        # Check valid mask (all should be valid)
        expected_valid = torch.ones((3, 3), dtype=torch.float32)
        assert torch.equal(target_dict["valid"], expected_valid)

    def test_nodata_value_less_than_num_classes(self) -> None:
        """Test process_inputs with nodata_value < num_classes."""
        task = SegmentationTask(num_classes=3, nodata_value=1)

        # Create test data with 1s as nodata
        raw_inputs = {
            "targets": torch.tensor(
                [[[0, 1, 2], [1, 1, 0], [2, 1, 0]]], dtype=torch.uint8
            )
        }

        _, target_dict = task.process_inputs(raw_inputs, {})

        # Check classes tensor (should be unchanged)
        expected_classes = torch.tensor(
            [[0, 1, 2], [1, 1, 0], [2, 1, 0]], dtype=torch.long
        )
        assert torch.equal(target_dict["classes"], expected_classes)

        # Check valid mask (1s should be invalid, others valid)
        expected_valid = torch.tensor(
            [[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=torch.float32
        )
        assert torch.equal(target_dict["valid"], expected_valid)

    def test_nodata_value_greater_than_or_equal_num_classes(self) -> None:
        """Test process_inputs with nodata_value >= num_classes."""
        task = SegmentationTask(num_classes=3, nodata_value=5)

        # Create test data with 5s as nodata
        raw_inputs = {
            "targets": torch.tensor(
                [[[0, 1, 2], [5, 5, 0], [2, 5, 0]]], dtype=torch.uint8
            )
        }

        _, target_dict = task.process_inputs(raw_inputs, {})

        # Check classes tensor (5s should be transformed to 0)
        expected_classes = torch.tensor(
            [[0, 1, 2], [0, 0, 0], [2, 0, 0]], dtype=torch.long
        )
        assert torch.equal(target_dict["classes"], expected_classes)

        # Check valid mask (5s should be invalid, others valid)
        expected_valid = torch.tensor(
            [[1.0, 1.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=torch.float32
        )
        assert torch.equal(target_dict["valid"], expected_valid)

    def test_nodata_value_equals_num_classes(self) -> None:
        """Test process_inputs with nodata_value == num_classes."""
        task = SegmentationTask(num_classes=3, nodata_value=3)

        # Create test data with 3s as nodata
        raw_inputs = {
            "targets": torch.tensor(
                [[[0, 1, 2], [3, 3, 0], [2, 3, 0]]], dtype=torch.uint8
            )
        }

        _, target_dict = task.process_inputs(raw_inputs, {})

        # Check classes tensor (3s should be transformed to 0)
        expected_classes = torch.tensor(
            [[0, 1, 2], [0, 0, 0], [2, 0, 0]], dtype=torch.long
        )
        assert torch.equal(target_dict["classes"], expected_classes)

        # Check valid mask (3s should be invalid, others valid)
        expected_valid = torch.tensor(
            [[1.0, 1.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=torch.float32
        )
        assert torch.equal(target_dict["valid"], expected_valid)

    def test_mutual_exclusivity_error(self) -> None:
        """Test that zero_is_invalid and nodata_value cannot both be set."""
        with pytest.raises(
            ValueError, match="zero_is_invalid and nodata_value cannot both be set"
        ):
            SegmentationTask(num_classes=3, zero_is_invalid=True, nodata_value=5)

    def test_load_targets_false(self) -> None:
        """Test process_inputs with load_targets=False."""
        task = SegmentationTask(num_classes=3, nodata_value=1)

        raw_inputs = {
            "targets": torch.tensor(
                [[[0, 1, 2], [1, 1, 0], [2, 1, 0]]], dtype=torch.uint8
            )
        }

        input_dict, target_dict = task.process_inputs(
            raw_inputs, {}, load_targets=False
        )

        # Should return empty dicts
        assert input_dict == {}
        assert target_dict == {}

    def test_single_channel_assertion(self) -> None:
        """Test that process_inputs asserts single channel input."""
        task = SegmentationTask(num_classes=3)

        # Create multi-channel input (should fail)
        raw_inputs = {
            "targets": torch.tensor(
                [[[0, 1, 2], [1, 1, 0]], [[2, 1, 0], [0, 1, 2]]], dtype=torch.uint8
            )
        }

        with pytest.raises(AssertionError):
            task.process_inputs(raw_inputs, {})


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
