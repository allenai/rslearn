from rslearn.dataset import Dataset
from rslearn.train.dataset import DataInput, ModelDataset, SplitConfig
from rslearn.train.model_context import RasterImage
from rslearn.train.tasks.classification import ClassificationTask
from rslearn.train.transforms.flip import Flip
from rslearn.train.transforms.pad import Pad


class TestTransforms:
    """Test transforms working with ModelDataset."""

    def test_flip(self, image_to_class_dataset: Dataset) -> None:
        split_config = SplitConfig(transforms=[Flip()])
        image_data_input = DataInput(
            "raster", ["image"], bands=["band"], passthrough=True
        )
        target_data_input = DataInput("vector", ["label"])
        model_dataset = ModelDataset(
            image_to_class_dataset,
            split_config,
            {
                "image": image_data_input,
                "targets": target_data_input,
            },
            workers=1,
            task=ClassificationTask("label", ["cls0", "cls1"], read_class_id=True),
        )
        input_dict, _, _ = model_dataset[0]
        assert isinstance(input_dict["image"], RasterImage)
        assert input_dict["image"].shape == (1, 1, 4, 4)

    def test_pad(self, image_to_class_dataset: Dataset) -> None:
        # pad one smaller than the input shape
        split_config = SplitConfig(transforms=[Pad(size=3, mode="center")])
        image_data_input = DataInput(
            "raster", ["image"], bands=["band"], passthrough=True
        )
        target_data_input = DataInput("vector", ["label"])
        model_dataset = ModelDataset(
            image_to_class_dataset,
            split_config,
            {
                "image": image_data_input,
                "targets": target_data_input,
            },
            workers=1,
            task=ClassificationTask("label", ["cls0", "cls1"], read_class_id=True),
        )
        input_dict, _, _ = model_dataset[0]
        # check we have padded to input - 1
        assert isinstance(input_dict["image"], RasterImage)
        assert input_dict["image"].shape == (1, 1, 3, 3)
