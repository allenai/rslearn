from datetime import UTC, datetime
from unittest.mock import patch

import shapely
from rasterio.crs import CRS

from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.data_source import Item
from rslearn.data_sources.omnicloudmask_utils import sort_items_by_omnicloudmask
from rslearn.utils.geometry import Projection, STGeometry


def _make_item(name: str) -> Item:
    return Item(
        name=name,
        geometry=STGeometry(
            WGS84_PROJECTION,
            shapely.box(0, 0, 1, 1),
            (
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
            ),
        ),
    )


def _make_geometry() -> STGeometry:
    return STGeometry(
        Projection(CRS.from_epsg(32630), 20.0, -20.0),
        shapely.box(0, 0, 25, 25),
        (
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
        ),
    )


def test_sort_items_by_omnicloudmask_prioritizes_thick_cloud_fraction() -> None:
    item_a = _make_item("a")
    item_b = _make_item("b")
    item_c = _make_item("c")
    items = [item_a, item_b, item_c]

    fractions_by_name = {
        "a": (0.90, 0.10, 0.00, 0.00),  # clear, thick, thin, shadow
        "b": (0.70, 0.00, 0.30, 0.00),
        "c": (0.95, 0.05, 0.00, 0.00),
    }

    with patch(
        "rslearn.data_sources.omnicloudmask_utils._compute_cloud_class_fractions",
        side_effect=lambda item, *args, **kwargs: fractions_by_name[item.name],
    ):
        result = sort_items_by_omnicloudmask(
            items=items,
            geometry=_make_geometry(),
            get_url=lambda _item, _asset: "unused",
            red_asset_key="B04",
            green_asset_key="B03",
            nir_asset_key="B8A",
        )

    # Lowest thick cloud fraction first: b (0.00), c (0.05), a (0.10)
    assert [item.name for item in result] == ["b", "c", "a"]


def test_sort_items_by_omnicloudmask_tiebreaks_on_clear_then_other_classes() -> None:
    item_a = _make_item("a")
    item_b = _make_item("b")
    items = [item_a, item_b]

    fractions_by_name = {
        "a": (0.80, 0.02, 0.10, 0.08),
        "b": (0.70, 0.02, 0.20, 0.08),
    }

    with patch(
        "rslearn.data_sources.omnicloudmask_utils._compute_cloud_class_fractions",
        side_effect=lambda item, *args, **kwargs: fractions_by_name[item.name],
    ):
        result = sort_items_by_omnicloudmask(
            items=items,
            geometry=_make_geometry(),
            get_url=lambda _item, _asset: "unused",
            red_asset_key="B04",
            green_asset_key="B03",
            nir_asset_key="B8A",
        )

    # Thick cloud ties; higher clear fraction should win.
    assert [item.name for item in result] == ["a", "b"]


def test_sort_items_by_omnicloudmask_failed_items_are_last() -> None:
    item_a = _make_item("a")
    item_b = _make_item("b")
    items = [item_a, item_b]

    def _fractions(item: Item, *args: object, **kwargs: object) -> tuple[float, ...]:
        if item.name == "a":
            raise RuntimeError("boom")
        return (0.80, 0.01, 0.10, 0.09)

    with patch(
        "rslearn.data_sources.omnicloudmask_utils._compute_cloud_class_fractions",
        side_effect=_fractions,
    ):
        result = sort_items_by_omnicloudmask(
            items=items,
            geometry=_make_geometry(),
            get_url=lambda _item, _asset: "unused",
            red_asset_key="B04",
            green_asset_key="B03",
            nir_asset_key="B8A",
        )

    assert [item.name for item in result] == ["b", "a"]
