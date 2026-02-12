import numpy as np
import pytest
import torch

from rslearn.models.component import FeatureMaps
from rslearn.train.model_context import ModelContext, RasterImage, SampleMetadata
from rslearn.train.tasks.multi_task import MultiTask
from rslearn.train.tasks.segmentation import SegmentationHead, SegmentationTask


class TestProcessInputs:
    """Unit tests relating to SegmentationTask.process_inputs."""

    @staticmethod
    def _make_metadata(
        empty_sample_metadata: SampleMetadata,
        height: int,
        width: int,
        *,
        window_bounds: tuple[int, int, int, int] | None = None,
        crop_bounds: tuple[int, int, int, int] | None = None,
    ) -> SampleMetadata:
        window_bounds = window_bounds or (0, 0, width, height)
        crop_bounds = crop_bounds or window_bounds
        return SampleMetadata(
            window_group="",
            window_name="",
            window_bounds=window_bounds,
            crop_bounds=crop_bounds,
            crop_idx=0,
            num_crops_in_window=1,
            time_range=None,
            projection=empty_sample_metadata.projection,
            dataset_source=None,
        )

    def test_zero_is_invalid_true(self, empty_sample_metadata: SampleMetadata) -> None:
        """Test process_inputs with zero_is_invalid=True."""
        task = SegmentationTask(num_classes=3, zero_is_invalid=True)

        # Create test data with 0s (invalid) and other values
        raw_inputs = {
            "targets": RasterImage(
                torch.tensor([[[[0, 1, 2], [0, 1, 0], [2, 1, 0]]]], dtype=torch.uint8)
            )
        }

        metadata = self._make_metadata(empty_sample_metadata, height=3, width=3)
        _, target_dict = task.process_inputs(raw_inputs, metadata)

        # Check classes tensor (should be unchanged)
        expected_classes = torch.tensor(
            [[0, 1, 2], [0, 1, 0], [2, 1, 0]], dtype=torch.long
        )
        assert torch.equal(target_dict["classes"].get_hw_tensor(), expected_classes)

        # Check valid mask (0s should be invalid, others valid)
        expected_valid = torch.tensor(
            [[0.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=torch.float32
        )
        assert torch.equal(target_dict["valid"].get_hw_tensor(), expected_valid)

    def test_zero_is_invalid_false(self, empty_sample_metadata: SampleMetadata) -> None:
        """Test process_inputs with zero_is_invalid=False."""
        task = SegmentationTask(num_classes=3, zero_is_invalid=False)

        # Create test data with 0s and other values
        raw_inputs = {
            "targets": RasterImage(
                torch.tensor([[[[0, 1, 2], [0, 1, 0], [2, 1, 0]]]], dtype=torch.uint8)
            )
        }

        metadata = self._make_metadata(empty_sample_metadata, height=3, width=3)
        _, target_dict = task.process_inputs(raw_inputs, metadata)

        # Check classes tensor (should be unchanged)
        expected_classes = torch.tensor(
            [[0, 1, 2], [0, 1, 0], [2, 1, 0]], dtype=torch.long
        )
        assert torch.equal(target_dict["classes"].get_hw_tensor(), expected_classes)

        # Check valid mask (all should be valid)
        expected_valid = torch.ones((3, 3), dtype=torch.float32)
        assert torch.equal(target_dict["valid"].get_hw_tensor(), expected_valid)

    def test_nodata_value_none(self, empty_sample_metadata: SampleMetadata) -> None:
        """Test process_inputs with nodata_value=None."""
        task = SegmentationTask(num_classes=3, nodata_value=None)

        # Create test data
        raw_inputs = {
            "targets": RasterImage(
                torch.tensor([[[[0, 1, 2], [0, 1, 0], [2, 1, 0]]]], dtype=torch.uint8)
            )
        }

        metadata = self._make_metadata(empty_sample_metadata, height=3, width=3)
        _, target_dict = task.process_inputs(raw_inputs, metadata)

        # Check classes tensor (should be unchanged)
        expected_classes = torch.tensor(
            [[0, 1, 2], [0, 1, 0], [2, 1, 0]], dtype=torch.long
        )
        assert torch.equal(target_dict["classes"].get_hw_tensor(), expected_classes)

        # Check valid mask (all should be valid)
        expected_valid = torch.ones((3, 3), dtype=torch.float32)
        assert torch.equal(target_dict["valid"].get_hw_tensor(), expected_valid)

    def test_nodata_value_less_than_num_classes(
        self, empty_sample_metadata: SampleMetadata
    ) -> None:
        """Test process_inputs with nodata_value < num_classes."""
        task = SegmentationTask(num_classes=3, nodata_value=1)

        # Create test data with 1s as nodata
        raw_inputs = {
            "targets": RasterImage(
                torch.tensor([[[[0, 1, 2], [1, 1, 0], [2, 1, 0]]]], dtype=torch.uint8)
            )
        }

        metadata = self._make_metadata(empty_sample_metadata, height=3, width=3)
        _, target_dict = task.process_inputs(raw_inputs, metadata)

        # Check classes tensor (should be unchanged)
        expected_classes = torch.tensor(
            [[0, 1, 2], [1, 1, 0], [2, 1, 0]], dtype=torch.long
        )
        assert torch.equal(target_dict["classes"].get_hw_tensor(), expected_classes)

        # Check valid mask (1s should be invalid, others valid)
        expected_valid = torch.tensor(
            [[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=torch.float32
        )
        assert torch.equal(target_dict["valid"].get_hw_tensor(), expected_valid)

    def test_nodata_value_greater_than_or_equal_num_classes(
        self, empty_sample_metadata: SampleMetadata
    ) -> None:
        """Test process_inputs with nodata_value >= num_classes."""
        task = SegmentationTask(num_classes=3, nodata_value=5)

        # Create test data with 5s as nodata
        raw_inputs = {
            "targets": RasterImage(
                torch.tensor([[[[0, 1, 2], [5, 5, 0], [2, 5, 0]]]], dtype=torch.uint8)
            )
        }

        metadata = self._make_metadata(empty_sample_metadata, height=3, width=3)
        _, target_dict = task.process_inputs(raw_inputs, metadata)

        # Check classes tensor (5s should be transformed to 0)
        expected_classes = torch.tensor(
            [[0, 1, 2], [0, 0, 0], [2, 0, 0]], dtype=torch.long
        )
        assert torch.equal(target_dict["classes"].get_hw_tensor(), expected_classes)

        # Check valid mask (5s should be invalid, others valid)
        expected_valid = torch.tensor(
            [[1.0, 1.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=torch.float32
        )
        assert torch.equal(target_dict["valid"].get_hw_tensor(), expected_valid)

    def test_nodata_value_equals_num_classes(
        self, empty_sample_metadata: SampleMetadata
    ) -> None:
        """Test process_inputs with nodata_value == num_classes."""
        task = SegmentationTask(num_classes=3, nodata_value=3)

        # Create test data with 3s as nodata
        raw_inputs = {
            "targets": RasterImage(
                torch.tensor([[[[0, 1, 2], [3, 3, 0], [2, 3, 0]]]], dtype=torch.uint8)
            )
        }

        metadata = self._make_metadata(empty_sample_metadata, height=3, width=3)
        _, target_dict = task.process_inputs(raw_inputs, metadata)

        # Check classes tensor (3s should be transformed to 0)
        expected_classes = torch.tensor(
            [[0, 1, 2], [0, 0, 0], [2, 0, 0]], dtype=torch.long
        )
        assert torch.equal(target_dict["classes"].get_hw_tensor(), expected_classes)

        # Check valid mask (3s should be invalid, others valid)
        expected_valid = torch.tensor(
            [[1.0, 1.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=torch.float32
        )
        assert torch.equal(target_dict["valid"].get_hw_tensor(), expected_valid)

    def test_mutual_exclusivity_error(self) -> None:
        """Test that zero_is_invalid and nodata_value cannot both be set."""
        with pytest.raises(
            ValueError, match="zero_is_invalid and nodata_value cannot both be set"
        ):
            SegmentationTask(num_classes=3, zero_is_invalid=True, nodata_value=5)

    def test_load_targets_false(self, empty_sample_metadata: SampleMetadata) -> None:
        """Test process_inputs with load_targets=False."""
        task = SegmentationTask(num_classes=3, nodata_value=1)

        raw_inputs = {
            "targets": RasterImage(
                torch.tensor([[[[0, 1, 2], [1, 1, 0], [2, 1, 0]]]], dtype=torch.uint8)
            )
        }

        input_dict, target_dict = task.process_inputs(
            raw_inputs, empty_sample_metadata, load_targets=False
        )

        # Should return empty dicts
        assert input_dict == {}
        assert target_dict == {}

    def test_single_channel_assertion(
        self, empty_sample_metadata: SampleMetadata
    ) -> None:
        """Test that process_inputs validates single channel/timestep input."""
        task = SegmentationTask(num_classes=3)

        # Create multi-channel input (should fail)
        raw_inputs = {
            "targets": RasterImage(
                torch.tensor(
                    [[[[0, 1, 2], [1, 1, 0]], [[2, 1, 0], [0, 1, 2]]]],
                    dtype=torch.uint8,
                )
            )
        }

        with pytest.raises(ValueError):
            task.process_inputs(raw_inputs, empty_sample_metadata)

    def test_process_inputs_masks_out_of_window_padding(
        self, empty_sample_metadata: SampleMetadata
    ) -> None:
        """Ensure padded (out-of-window) pixels are marked invalid."""
        window_bounds = (0, 0, 10, 10)
        crop_bounds = (-5, -5, 15, 15)

        decoded = torch.zeros((20, 20), dtype=torch.int16)
        decoded[5:15, 5:15] = 2

        raw_inputs = {
            "targets": RasterImage(decoded[None, None, :, :], timestamps=None),
        }
        metadata = self._make_metadata(
            empty_sample_metadata,
            height=20,
            width=20,
            window_bounds=window_bounds,
            crop_bounds=crop_bounds,
        )

        task = SegmentationTask(num_classes=3, nodata_value=-1)
        _, target_dict = task.process_inputs(
            raw_inputs, metadata=metadata, load_targets=True
        )

        valid = target_dict["valid"].get_hw_tensor()
        assert int(valid.sum().item()) == 10 * 10
        assert valid[0, 0] == 0
        assert valid[5, 5] == 1


class TestProcessOutput:
    """Unit tests relating to SegmentationTask.process_output."""

    def test_with_prob_scales(self, empty_sample_metadata: SampleMetadata) -> None:
        """Ensure that prob_scales is reflected correctly in process_output."""
        task = SegmentationTask(num_classes=3, prob_scales=[1.0, 1.0, 3.0])
        out_probs = torch.tensor([[[0.3]], [[0.5]], [[0.2]]], dtype=torch.float32)
        classes = task.process_output(out_probs, empty_sample_metadata)
        # Output class should be 2 since the 0.2 probability is multiplied by 3 before
        # computing argmax.
        assert isinstance(classes, np.ndarray)
        assert classes.shape == (1, 1, 1)
        assert classes[0][0] == 2

    def test_no_prob_scales(self, empty_sample_metadata: SampleMetadata) -> None:
        """Test process_output without prob_scales."""
        task = SegmentationTask(num_classes=3)
        out_probs = torch.tensor([[[0.3]], [[0.5]], [[0.2]]], dtype=torch.float32)
        classes = task.process_output(out_probs, empty_sample_metadata)
        assert isinstance(classes, np.ndarray)
        assert classes.shape == (1, 1, 1)
        assert classes[0][0] == 1

    def test_output_probs_multiclass(
        self, empty_sample_metadata: SampleMetadata
    ) -> None:
        """Test output_probs=True with multi-class (3 classes) segmentation."""
        task = SegmentationTask(num_classes=3, output_probs=True)
        # Multi-class probabilities: 3 channels, 2x2 spatial
        out_probs = torch.tensor(
            [
                [[0.1, 0.2], [0.3, 0.1]],  # Class 0
                [[0.3, 0.5], [0.4, 0.6]],  # Class 1
                [[0.6, 0.3], [0.3, 0.3]],  # Class 2
            ],
            dtype=torch.float32,
        )
        result = task.process_output(out_probs, empty_sample_metadata)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2, 2)  # All 3 channels returned
        np.testing.assert_allclose(result, out_probs.numpy())

    def test_output_probs_with_class_idx_multiclass(
        self, empty_sample_metadata: SampleMetadata
    ) -> None:
        """Test output_probs=True with output_class_idx for multi-class segmentation."""
        task = SegmentationTask(num_classes=3, output_probs=True, output_class_idx=2)
        # Multi-class probabilities: 3 channels, 2x2 spatial
        out_probs = torch.tensor(
            [
                [[0.1, 0.2], [0.3, 0.1]],  # Class 0
                [[0.3, 0.5], [0.4, 0.6]],  # Class 1
                [[0.6, 0.3], [0.3, 0.3]],  # Class 2
            ],
            dtype=torch.float32,
        )
        result = task.process_output(out_probs, empty_sample_metadata)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 2, 2)  # Only class 2 returned
        expected = out_probs[2:3, :, :].numpy()
        np.testing.assert_allclose(result, expected)

    def test_output_probs_with_prob_scales(
        self, empty_sample_metadata: SampleMetadata
    ) -> None:
        """Test output_probs=True combined with prob_scales."""
        # prob_scales are applied before returning probabilities
        task = SegmentationTask(
            num_classes=3, output_probs=True, prob_scales=[1.0, 2.0, 1.0]
        )
        out_probs = torch.tensor(
            [
                [[0.3, 0.3]],  # Class 0
                [[0.4, 0.4]],  # Class 1
                [[0.3, 0.3]],  # Class 2
            ],
            dtype=torch.float32,
        )
        result = task.process_output(out_probs, empty_sample_metadata)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1, 2)
        # prob_scales should be applied: [0.3, 0.8, 0.3]
        expected = np.array([[[0.3, 0.3]], [[0.8, 0.8]], [[0.3, 0.3]]])
        np.testing.assert_allclose(result, expected)


def test_segmentation_head_temperature_confidence() -> None:
    """Verify temperature scaling affects output confidence."""
    logits = torch.tensor([[[[0.0]], [[1.0]], [[2.0]]]], dtype=torch.float32)  # BxCxHxW
    feature_maps = FeatureMaps([logits])
    context = ModelContext(inputs=[], metadatas=[])

    cold_head = SegmentationHead(temperature=0.5)
    hot_head = SegmentationHead(temperature=2.0)

    cold_output = cold_head(
        intermediates=feature_maps, context=context, targets=None
    ).outputs
    hot_output = hot_head(
        intermediates=feature_maps, context=context, targets=None
    ).outputs

    assert cold_output.shape == hot_output.shape == torch.Size([1, 3, 1, 1])
    assert cold_output.argmax(dim=1).equal(hot_output.argmax(dim=1))

    max_prob_cold = cold_output.max().item()
    max_prob_hot = hot_output.max().item()
    assert max_prob_cold > max_prob_hot


class TestSegmentationMetrics:
    """Tests for SegmentationTask.get_metrics() and metric computation.

    Uses a 2x2 3-class segmentation with one wrong pixel:
        GT labels:    [[0, 1], [2, 0]]
        Pred argmax:  [[0, 1], [2, 1]]  (pixel (1,1) wrong: pred=1, gt=0)
    """

    @pytest.fixture()
    def segmentation_preds(self) -> torch.Tensor:
        """BCHW softmax predictions for 3-class 2x2 segmentation."""
        return torch.tensor(
            [
                [
                    [[0.8, 0.1], [0.1, 0.1]],  # class 0 probs
                    [[0.1, 0.8], [0.1, 0.8]],  # class 1 probs
                    [[0.1, 0.1], [0.8, 0.1]],  # class 2 probs
                ]
            ]
        )  # [1, 3, 2, 2]

    @pytest.fixture()
    def segmentation_targets(self) -> list[dict[str, RasterImage]]:
        """Target dicts for 3-class 2x2 segmentation."""
        return [
            {
                "classes": RasterImage(
                    torch.tensor([[[[0, 1], [2, 0]]]], dtype=torch.long)
                ),
                "valid": RasterImage(torch.ones(1, 1, 2, 2)),
            }
        ]

    def test_scalar_metrics_keys_and_values(
        self,
        segmentation_preds: torch.Tensor,
        segmentation_targets: list[dict[str, RasterImage]],
    ) -> None:
        """Accuracy + mean_iou + F1 all return scalars with correct keys and values.

        With 3 classes and one misclassified pixel out of 4:
        - accuracy (macro, i.e. per-class recall averaged):
            cls0=1/2, cls1=1/1, cls2=1/1 -> 5/6
        - mean_iou = (1/2 + 1/2 + 1/1) / 3 = 2/3
        - F1 = (2/3 + 2/3 + 1) / 3 = 7/9
        """
        task = SegmentationTask(
            num_classes=3,
            enable_accuracy_metric=True,
            enable_miou_metric=True,
            enable_f1_metric=True,
        )
        metrics = task.get_metrics()
        metrics.update(segmentation_preds, segmentation_targets)
        result = metrics.compute()

        assert result["accuracy"] == pytest.approx(5 / 6, abs=1e-4)
        assert result["mean_iou"] == pytest.approx(2 / 3, abs=1e-4)
        assert result["F1"] == pytest.approx(7 / 9, abs=1e-4)

    def test_dict_metric_per_class_miou(
        self,
        segmentation_preds: torch.Tensor,
        segmentation_targets: list[dict[str, RasterImage]],
    ) -> None:
        """Mean IoU with per-class reporting returns a dict with per-class keys.

        Per-class IoU:
        - cls 0: intersection=1, union=2 -> 0.5
        - cls 1: intersection=1, union=2 -> 0.5
        - cls 2: intersection=1, union=1 -> 1.0
        """
        task = SegmentationTask(
            num_classes=3,
            enable_accuracy_metric=False,
            enable_miou_metric=True,
            report_metric_per_class=True,
        )
        metrics = task.get_metrics()
        metrics.update(segmentation_preds, segmentation_targets)
        result = metrics.compute()

        assert result["mean_iou/avg"] == pytest.approx(2 / 3, abs=1e-4)
        assert result["mean_iou/cls_0"] == pytest.approx(0.5, abs=1e-4)
        assert result["mean_iou/cls_1"] == pytest.approx(0.5, abs=1e-4)
        assert result["mean_iou/cls_2"] == pytest.approx(1.0, abs=1e-4)

    def test_multi_task_dict_metric_keys(
        self,
        segmentation_preds: torch.Tensor,
        segmentation_targets: list[dict[str, RasterImage]],
    ) -> None:
        """Verify that MultiTask wraps dict metric keys with task name.

        This verifies the MetricWrapper in multi_task.py properly prefixes dict keys
        so that per-class metrics from different tasks don't clash.
        """
        seg_task = SegmentationTask(
            num_classes=3,
            enable_accuracy_metric=True,
            enable_miou_metric=True,
            report_metric_per_class=True,
        )
        multi_task = MultiTask(
            tasks={"seg": seg_task},
            input_mapping={"seg": {"targets": "targets"}},
        )
        metrics = multi_task.get_metrics()

        # Wrap preds/targets in multi-task format (list of per-sample dicts).
        preds = [{"seg": segmentation_preds[0]}]  # each element is {task_name: CHW}
        targets = [{"seg": segmentation_targets[0]}]

        metrics.update(preds, targets)
        result = metrics.compute()

        # Scalar metric: MetricCollection uses the key "seg/accuracy" directly.
        assert result["seg/accuracy"] == pytest.approx(5 / 6, abs=1e-4)

        # Dict metric: MetricWrapper prefixes dict keys with "seg/".
        assert result["seg/mean_iou/avg"] == pytest.approx(2 / 3, abs=1e-4)
        assert result["seg/mean_iou/cls_0"] == pytest.approx(0.5, abs=1e-4)
        assert result["seg/mean_iou/cls_1"] == pytest.approx(0.5, abs=1e-4)
        assert result["seg/mean_iou/cls_2"] == pytest.approx(1.0, abs=1e-4)
