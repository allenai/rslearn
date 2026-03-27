import pytest
import shapely

from rslearn.config import QueryConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import Item
from rslearn.data_sources.hf_srtm import SRTM
from rslearn.data_sources.utils import MatchedItemGroup
from rslearn.utils.geometry import STGeometry


def test_srtm_rejects_min_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    """SRTM custom matching should reject min_matches>0 explicitly."""
    monkeypatch.setattr(SRTM, "_load_file_index", lambda self: ({}, {}))
    data_source = SRTM()
    geometry = STGeometry(WGS84_PROJECTION, shapely.box(0, 0, 1, 1), None)

    with pytest.raises(ValueError, match="min_matches"):
        data_source.get_items(
            [geometry],
            query_config=QueryConfig(
                space_mode=SpaceMode.MOSAIC,
                max_matches=1,
                min_matches=1,
            ),
        )


def test_srtm_returns_matched_item_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    """SRTM should return MatchedItemGroup instances."""
    item = Item(
        "N00/SRTM3N00E000V2.tif",
        STGeometry(WGS84_PROJECTION, shapely.box(0, 0, 1, 1), None),
    )
    monkeypatch.setattr(
        SRTM,
        "_load_file_index",
        lambda self: ({item.name.split("/")[-1]: item}, {(0, 0): item}),
    )
    data_source = SRTM()
    geometry = STGeometry(WGS84_PROJECTION, shapely.box(0, 0, 1, 1), None)

    groups = data_source.get_items(
        [geometry],
        query_config=QueryConfig(space_mode=SpaceMode.MOSAIC, max_matches=1),
    )
    assert len(groups) == 1
    assert len(groups[0]) == 1
    assert isinstance(groups[0][0], MatchedItemGroup)
    assert groups[0][0].request_time_range is None
