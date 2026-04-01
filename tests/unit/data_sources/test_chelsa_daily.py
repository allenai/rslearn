from datetime import UTC, datetime

import pytest
import shapely

from rslearn.config import QueryConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.chelsa import CHELSADaily, CHELSADailyItem
from rslearn.utils.geometry import STGeometry


def _window(start: datetime, end: datetime) -> STGeometry:
    return STGeometry(
        WGS84_PROJECTION,
        shapely.box(-1.0, -1.0, 1.0, 1.0),
        (start, end),
    )


def test_chelsa_daily_get_asset_url() -> None:
    data_source = CHELSADaily(band_names=["tas"])
    item = data_source.get_item_by_name("chelsa_daily_20230616")

    assert data_source.get_asset_url(item, "tas") == (
        "https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/daily/tas/2023/"
        "CHELSA_tas_16_06_2023_V.2.1.tif"
    )


def test_chelsa_daily_precipitation_alias_switches_for_pr_band() -> None:
    data_source = CHELSADaily(band_names=["pr"])

    pre_overlap = data_source.get_item_by_name("chelsa_daily_20191231")
    overlap = data_source.get_item_by_name("chelsa_daily_20200601")
    post_overlap = data_source.get_item_by_name("chelsa_daily_20210101")

    assert data_source.get_asset_url(pre_overlap, "pr") == (
        "https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/daily/pr/2019/"
        "CHELSA_pr_31_12_2019_V.2.1.tif"
    )
    assert data_source.get_asset_url(overlap, "pr") == (
        "https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/daily/pr/2020/"
        "CHELSA_pr_01_06_2020_V.2.1.tif"
    )
    assert data_source.get_asset_url(post_overlap, "pr") == (
        "https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/daily/prec/2021/"
        "CHELSA_prec_01_01_2021_V.2.1.tif"
    )


def test_chelsa_daily_precipitation_alias_switches_for_prec_band() -> None:
    data_source = CHELSADaily(band_names=["prec"])

    pre_overlap = data_source.get_item_by_name("chelsa_daily_20191231")
    overlap = data_source.get_item_by_name("chelsa_daily_20200601")
    post_overlap = data_source.get_item_by_name("chelsa_daily_20210101")

    assert data_source.get_asset_url(pre_overlap, "prec") == (
        "https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/daily/pr/2019/"
        "CHELSA_pr_31_12_2019_V.2.1.tif"
    )
    assert data_source.get_asset_url(overlap, "prec") == (
        "https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/daily/prec/2020/"
        "CHELSA_prec_01_06_2020_V.2.1.tif"
    )
    assert data_source.get_asset_url(post_overlap, "prec") == (
        "https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/daily/prec/2021/"
        "CHELSA_prec_01_01_2021_V.2.1.tif"
    )


def test_chelsa_daily_get_items_returns_daily_items() -> None:
    data_source = CHELSADaily(band_names=["tas"])

    groups = data_source.get_items(
        [
            _window(
                datetime(2023, 6, 16, 12, tzinfo=UTC),
                datetime(2023, 6, 18, 0, tzinfo=UTC),
            )
        ],
        query_config=QueryConfig(space_mode=SpaceMode.SINGLE_COMPOSITE),
    )

    assert len(groups) == 1
    assert len(groups[0]) == 1
    items = groups[0][0].items
    assert [item.name for item in items] == [
        "chelsa_daily_20230616",
        "chelsa_daily_20230617",
    ]


def test_chelsa_daily_get_items_clamps_to_configured_range() -> None:
    data_source = CHELSADaily(
        band_names=["tas"],
        start_date="2023-06-10",
        end_date="2023-06-12",
    )

    groups = data_source.get_items(
        [
            _window(
                datetime(2023, 6, 9, 0, tzinfo=UTC),
                datetime(2023, 6, 14, 0, tzinfo=UTC),
            )
        ],
        query_config=QueryConfig(space_mode=SpaceMode.SINGLE_COMPOSITE),
    )

    assert len(groups) == 1
    assert len(groups[0]) == 1
    items = groups[0][0].items
    assert [item.name for item in items] == [
        "chelsa_daily_20230610",
        "chelsa_daily_20230611",
        "chelsa_daily_20230612",
    ]


def test_chelsa_daily_get_items_outside_range_returns_empty_group() -> None:
    data_source = CHELSADaily(
        band_names=["tas"],
        start_date="2023-06-10",
        end_date="2023-06-12",
    )

    groups = data_source.get_items(
        [
            _window(
                datetime(2023, 6, 13, 0, tzinfo=UTC),
                datetime(2023, 6, 14, 0, tzinfo=UTC),
            )
        ],
        query_config=QueryConfig(space_mode=SpaceMode.SINGLE_COMPOSITE),
    )

    assert len(groups) == 1
    assert len(groups[0]) == 1
    assert len(groups[0][0].items) == 0


def test_chelsa_daily_requires_single_composite() -> None:
    data_source = CHELSADaily(band_names=["tas"])

    with pytest.raises(ValueError, match="SINGLE_COMPOSITE"):
        data_source.get_items(
            [
                _window(
                    datetime(2023, 6, 16, 0, tzinfo=UTC),
                    datetime(2023, 6, 17, 0, tzinfo=UTC),
                )
            ],
            query_config=QueryConfig(space_mode=SpaceMode.MOSAIC),
        )


def test_chelsa_daily_item_serialization_roundtrip() -> None:
    data_source = CHELSADaily(band_names=["tas"])
    item = data_source.get_item_by_name("chelsa_daily_20230616")

    restored = CHELSADailyItem.deserialize(item.serialize())

    assert restored.name == item.name
    assert restored.item_date == item.item_date
    assert restored.geometry.serialize() == item.geometry.serialize()
