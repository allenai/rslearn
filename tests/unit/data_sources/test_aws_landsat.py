"""Unit tests for the aws_landsat data source."""

import json
import pathlib
from datetime import datetime
from unittest.mock import MagicMock, patch

import shapely
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.aws_landsat import LandsatOliTirs, LandsatOliTirsItem
from rslearn.utils.geometry import STGeometry


def _make_cache_entry(name: str) -> dict:
    ts = datetime(2020, 11, 7)
    geometry = STGeometry(WGS84_PROJECTION, shapely.Point(0, 0).buffer(0.1), (ts, ts))
    return LandsatOliTirsItem(
        name=name,
        geometry=geometry,
        blob_path=f"collection02/level-1/standard/oli-tirs/2020/203/022/{name}/{name}_",
        cloud_cover=5.0,
    ).serialize()


def test_oli_only_scenes_filtered_from_read_products(tmp_path: pathlib.Path) -> None:
    """LO08_/LO09_ OLI-only items are excluded; LC08_/LC09_ OLI-TIRS items are kept.

    Regression test for https://github.com/allenai/rslearn/issues/645.
    LO08/LO09 products lack thermal bands B10/B11 and caused 404s when
    materializing those bands.
    """
    with patch("rslearn.data_sources.aws_landsat.boto3") as mock_boto3:
        mock_boto3.client.return_value = MagicMock()
        mock_boto3.resource.return_value.Bucket.return_value = MagicMock()
        data_source = LandsatOliTirs(metadata_cache_dir=UPath(tmp_path))

    cache_entries = [
        _make_cache_entry("LC08_L1TP_203022_20201107_20201111_02_T1"),
        _make_cache_entry("LC09_L1TP_203022_20201107_20201111_02_T1"),
        _make_cache_entry("LO08_L1TP_203022_20201107_20201111_02_T1"),
        _make_cache_entry("LO09_L1TP_203022_20201107_20201111_02_T1"),
    ]
    (tmp_path / "2020_203_022.json").write_text(json.dumps(cache_entries))

    items = list(data_source._read_products({(2020, "203", "022")}))
    returned_names = {item.name for item in items}

    assert "LC08_L1TP_203022_20201107_20201111_02_T1" in returned_names
    assert "LC09_L1TP_203022_20201107_20201111_02_T1" in returned_names
    assert "LO08_L1TP_203022_20201107_20201111_02_T1" not in returned_names
    assert "LO09_L1TP_203022_20201107_20201111_02_T1" not in returned_names
