import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest
import shapely
from upath import UPath

pytest.importorskip("earthdaily")

from rslearn.config import QueryConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.earthdaily import Biophysical, EarthDailyItem
from rslearn.dataset import Dataset, Window, WindowLayerData
from rslearn.utils.geometry import STGeometry


class _FakeBiophysical(Biophysical):
    def __init__(
        self,
        items_by_name: dict[str, EarthDailyItem],
        *,
        variable: Literal["lai", "fapar", "fcover"] = "lai",
        match_source_layer: str = "sentinel2",
        match_source_item_template: str | None = None,
    ) -> None:
        super().__init__(
            variable=variable,
            match_source_layer=match_source_layer,
            match_source_item_template=match_source_item_template,
        )
        self.items_by_name = items_by_name

    def get_item_by_name(self, name: str) -> EarthDailyItem:
        if name not in self.items_by_name:
            raise KeyError(name)
        return self.items_by_name[name]


def _make_window(tmp_path: Path) -> Window:
    with (tmp_path / "config.json").open("w") as f:
        json.dump({"layers": {}}, f)
    dataset = Dataset(UPath(tmp_path))
    window = Window(
        storage=dataset.storage,
        group="default",
        name="window",
        projection=WGS84_PROJECTION,
        bounds=(0, 0, 10, 10),
        time_range=(
            datetime(2025, 4, 24, tzinfo=UTC),
            datetime(2025, 4, 25, tzinfo=UTC),
        ),
    )
    window.save()
    return window


def _make_item(name: str, *, product_id: str | None = None) -> EarthDailyItem:
    geometry = STGeometry(
        WGS84_PROJECTION,
        shapely.box(0, 0, 10, 10),
        (
            datetime(2025, 4, 24, tzinfo=UTC),
            datetime(2025, 4, 25, tzinfo=UTC),
        ),
    )
    return EarthDailyItem(
        name=name,
        geometry=geometry,
        asset_urls={
            "lai": "/tmp/lai.tif",
            "fapar": "/tmp/fapar.tif",
            "fcover": "/tmp/fcover.tif",
        },
        product_id=product_id,
    )


def test_biophysical_matches_source_item_name_with_default_template(
    tmp_path: Path,
) -> None:
    window = _make_window(tmp_path)
    source_item = _make_item("S2C_31TEJ_20250424_0_L2A")
    source_time_range = (
        datetime(2025, 4, 24, tzinfo=UTC),
        datetime(2025, 4, 24, 1, tzinfo=UTC),
    )
    window.save_layer_datas(
        {
            "sentinel2": WindowLayerData(
                layer_name="sentinel2",
                serialized_item_groups=[[source_item.serialize()]],
                group_time_ranges=[source_time_range],
            )
        }
    )

    lai_item = _make_item("S2C_31TEJ_20250424_0_L2A_LAI")
    data_source = _FakeBiophysical({lai_item.name: lai_item})

    results = data_source.get_items_for_windows(
        [window], [window.get_geometry()], QueryConfig(min_matches=1)
    )

    assert [[item.name for item in group.items] for group in results[0]] == [
        ["S2C_31TEJ_20250424_0_L2A_LAI"]
    ]
    assert results[0][0].request_time_range == source_time_range


def test_biophysical_match_template_can_use_source_product_id(tmp_path: Path) -> None:
    window = _make_window(tmp_path)
    source_item = _make_item(
        "earthdaily-stac-id",
        product_id="S2C_31TEJ_20250424_0_L2A",
    )
    window.save_layer_datas(
        {
            "sentinel2": WindowLayerData(
                layer_name="sentinel2",
                serialized_item_groups=[[source_item.serialize()]],
            )
        }
    )

    fapar_item = _make_item("S2C_31TEJ_20250424_0_L2A_FAPAR")
    data_source = _FakeBiophysical(
        {fapar_item.name: fapar_item},
        variable="fapar",
        match_source_item_template="{source_product_id}_{variable_upper}",
    )

    results = data_source.get_items_for_windows(
        [window], [window.get_geometry()], QueryConfig(min_matches=1)
    )

    assert [[item.name for item in group.items] for group in results[0]] == [
        ["S2C_31TEJ_20250424_0_L2A_FAPAR"]
    ]


def test_biophysical_missing_matched_items_respects_min_matches(
    tmp_path: Path,
) -> None:
    window = _make_window(tmp_path)
    source_item_1 = _make_item("S2C_31TEJ_20250424_0_L2A")
    source_item_2 = _make_item("S2C_31TEJ_20250425_0_L2A")
    window.save_layer_datas(
        {
            "sentinel2": WindowLayerData(
                layer_name="sentinel2",
                serialized_item_groups=[
                    [source_item_1.serialize()],
                    [source_item_2.serialize()],
                ],
            )
        }
    )

    lai_item = _make_item("S2C_31TEJ_20250424_0_L2A_LAI")
    data_source = _FakeBiophysical({lai_item.name: lai_item})

    results = data_source.get_items_for_windows(
        [window], [window.get_geometry()], QueryConfig(min_matches=2)
    )

    assert results == [[]]


def test_biophysical_match_template_requires_source_layer() -> None:
    with pytest.raises(ValueError, match="requires match_source_layer"):
        Biophysical(
            variable="lai",
            match_source_item_template="{source_item_name}_LAI",
        )
