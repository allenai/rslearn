from rslearn.dataset import Dataset
from rslearn.train.dataset import DataInput, ModelDataset, SplitConfig
from rslearn.train.tasks.classification import ClassificationTask
from rslearn.train.transforms.flip import Flip


class TestTransforms:
    """Test transforms working with ModelDataset."""

    def test_flip(self, image_to_class_dataset: Dataset):
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
        assert input_dict["image"].shape == (1, 4, 4)
