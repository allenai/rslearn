import pytest
import shapely

from rslearn.config import QueryConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.hf_srtm import SRTM
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
