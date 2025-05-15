import json
import pathlib
from typing import Any

from rasterio.crs import CRS
from upath import UPath

from rslearn.dataset import Dataset, Window
from rslearn.train.dataset import ModelDataset, SplitConfig
from rslearn.train.tasks.classification import ClassificationTask
from rslearn.utils.geometry import Projection


class TestDataset:
    """Test ModelDataset."""

    def test_multiple_tags(self, tmp_path: pathlib.Path) -> None:
        """Ensure that ModelDataset filters correctly when multile tags are configured.

        Multiple tags should be treated as conjunction (logical and) so only windows
        matching all of the tags should be accepted.
        """
        ds_path = UPath(tmp_path)
        with (ds_path / "config.json").open("w") as f:
            json.dump({"layers": {}}, f)
        group = "default"
        projection = Projection(CRS.from_epsg(32614), 10, -10)
        bounds = (500000, 500000, 500040, 500040)

        def add_window(name: str, options: dict[str, Any]) -> None:
            Window(
                path=Window.get_window_root(ds_path, group, name),
                group=group,
                name=name,
                projection=projection,
                bounds=bounds,
                time_range=None,
                options=options,
            ).save()

        # (1) Window with first tag only (skip).
        add_window("window1", {"tag1": "yes"})
        # (2) Window with second tag only (skip).
        add_window("window2", {"tag2": "yes"})
        # (3) Window with both tags (match).
        add_window("window3", {"tag1": "yes", "tag2": "yes"})
        # (4) Window with both tags plus one more (match).
        add_window("window4", {"tag1": "yes", "tag2": "yes", "tag3": "yes"})

        dataset = ModelDataset(
            dataset=Dataset(ds_path),
            split_config=SplitConfig(tags={"tag1": "yes", "tag2": "yes"}),
            inputs={},
            task=ClassificationTask(property_name="prop_name", classes=[]),
            workers=4,
        )
        assert len(dataset) == 2
        window_names = {window.name for window in dataset.get_windows()}
        assert window_names == {"window3", "window4"}
