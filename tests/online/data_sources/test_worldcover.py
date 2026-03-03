"""Online integration tests for the WorldCover data source (hits real S3)."""

import pathlib

from rslearn.config import QueryConfig, SpaceMode
from rslearn.data_sources.worldcover import WorldCover
from rslearn.utils.geometry import STGeometry


def test_read_raster(tmp_path: pathlib.Path, seattle2020: STGeometry) -> None:
    """Test direct materialization of WorldCover data for Seattle."""
    data_source = WorldCover(metadata_cache_dir=str(tmp_path / "cache"))
    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([seattle2020], query_config)[0]
    assert len(item_groups) >= 1
    assert len(item_groups[0]) >= 1
    item = item_groups[0][0]
    assert item.name == "N45W123"

    bounds = (
        int(seattle2020.shp.bounds[0]),
        int(seattle2020.shp.bounds[1]),
        int(seattle2020.shp.bounds[2]),
        int(seattle2020.shp.bounds[3]),
    )
    array = data_source.read_raster(
        layer_name="worldcover",
        item_name=item.name,
        bands=["B1"],
        projection=seattle2020.projection,
        bounds=bounds,
    )
    chw = array.get_chw_array()
    assert chw.shape[0] == 1
    # WorldCover values are land cover classes 10-100; make sure we got real data.
    assert chw.max() > 0
