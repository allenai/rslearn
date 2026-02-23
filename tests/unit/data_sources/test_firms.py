"""Unit tests for FIRMS data source."""

from datetime import UTC, datetime

import shapely

from rslearn.config import QueryConfig
from rslearn.data_sources.firms import FIRMS, FIRMSItem
from rslearn.utils.geometry import WGS84_PROJECTION, STGeometry


def test_get_items_single_item_per_geometry() -> None:
    """FIRMS should return one item per input geometry."""
    data_source = FIRMS(map_key="dummy")
    geometry = STGeometry(
        WGS84_PROJECTION,
        shapely.box(-7.7, 33.5, -7.5, 33.7),
        (
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 2, 1, tzinfo=UTC),
        ),
    )

    groups = data_source.get_items([geometry], QueryConfig())
    assert len(groups) == 1
    assert len(groups[0]) == 1
    assert len(groups[0][0]) == 1

    item = groups[0][0][0]
    assert isinstance(item, FIRMSItem)
    assert item.source == "VIIRS_SNPP_NRT"
    assert item.geometry.time_range == geometry.time_range


def test_get_items_requires_time_range() -> None:
    """FIRMS should require a time range for each geometry."""
    data_source = FIRMS(map_key="dummy")
    geometry = STGeometry(WGS84_PROJECTION, shapely.box(-7.7, 33.5, -7.5, 33.7), None)

    try:
        data_source.get_items([geometry], QueryConfig())
        assert False, "expected get_items to fail for missing time_range"
    except ValueError as exc:
        assert "time_range" in str(exc)


def test_iter_date_chunks_month_split() -> None:
    """Date range should be split by FIRMS chunk size."""
    data_source = FIRMS(map_key="dummy", max_days_per_request=5)
    chunks = data_source._iter_date_chunks(
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 2, 1, tzinfo=UTC),
    )
    assert [day_count for _, day_count in chunks] == [5, 5, 5, 5, 5, 5, 1]
    assert chunks[0][0].isoformat() == "2026-01-01"
    assert chunks[-1][0].isoformat() == "2026-01-31"


def test_iter_date_chunks_includes_partial_final_day() -> None:
    """If end time is not midnight, include that final date."""
    data_source = FIRMS(map_key="dummy", max_days_per_request=5)
    chunks = data_source._iter_date_chunks(
        datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        datetime(2026, 1, 2, 6, 0, tzinfo=UTC),
    )
    assert chunks == [(datetime(2026, 1, 1, tzinfo=UTC).date(), 2)]


def test_parse_csv_features() -> None:
    """CSV rows should be parsed into vector point features."""
    data_source = FIRMS(map_key="dummy")
    csv_text = (
        "latitude,longitude,acq_date,confidence\n"
        "33.5731,-7.5898,2026-01-05,h\n"
    )
    bbox = (-8.0, 33.0, -7.0, 34.0)

    features = data_source._parse_csv_features(csv_text, bbox)
    assert len(features) == 1
    assert features[0].geometry.shp.x == -7.5898
    assert features[0].geometry.shp.y == 33.5731
    assert features[0].properties["acq_date"] == "2026-01-05"
    assert features[0].properties["confidence"] == "h"
