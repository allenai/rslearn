import pathlib
from datetime import datetime, timezone

import shapely
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.copernicus import get_sentinel2_tiles
from rslearn.utils.geometry import STGeometry


class TestGetSentinel2Tiles:
    """Tests for get_sentinel2_tiles."""

    def test_antimeridian_handling(self, tmp_path: pathlib.Path) -> None:
        """Make sure that get_sentinel2_tiles handles the antimeridian correctly.

        Previously we returned tiles that spanned the antimeridian for any geometry
        that had a matching latitude.
        """
        # We use a 1x1 degree geometry that should match with these tiles:
        # - 10UFU
        # - 10TFT
        # - 10UEU
        # - 10TET
        geom = STGeometry(
            WGS84_PROJECTION,
            shapely.box(-122, 47, -121, 48),
            (
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 2, 1, tzinfo=timezone.utc),
            ),
        )
        tiles = get_sentinel2_tiles(geom, UPath(tmp_path))
        assert set(tiles) == {
            "10UFU",
            "10TFT",
            "10UEU",
            "10TET",
        }, f"Got incorrect tile list {tiles}"
