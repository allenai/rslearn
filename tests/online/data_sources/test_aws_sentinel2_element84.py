import pathlib

import numpy as np
from upath import UPath

from rslearn.config import (
    QueryConfig,
    SpaceMode,
)
from rslearn.data_sources.aws_sentinel2_element84 import Sentinel2
from rslearn.data_sources.planetary_computer import (
    Sentinel2 as PlanetaryComputerSentinel2,
)
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils.geometry import STGeometry


def test_aws_sentinel2_element84(
    tmp_path: pathlib.Path, seattle2020: STGeometry
) -> None:
    """Test ingesting an item corresponding to seattle2020."""
    asset_name = "red"
    band_name = "B04"
    data_source = Sentinel2(assets=[asset_name])

    print("get items")
    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([seattle2020], query_config)[0]
    item = item_groups[0][0]

    tile_store_dir = UPath(tmp_path)
    tile_store = DefaultTileStore(str(tile_store_dir))
    tile_store.set_dataset_path(tile_store_dir)

    print("ingest")
    layer_name = "layer"
    data_source.ingest(
        TileStoreWithLayer(tile_store, layer_name), item_groups[0], [[seattle2020]]
    )
    assert tile_store.is_raster_ready(layer_name, item.name, [band_name])


def test_materialize_all_bands(tmp_path: pathlib.Path, seattle2020: STGeometry) -> None:
    """Test directly materializing all of the bands in a small geometry."""
    data_source = Sentinel2()
    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([seattle2020], query_config)[0]
    item = item_groups[0][0]

    bounds = (
        int(seattle2020.shp.bounds[0]),
        int(seattle2020.shp.bounds[1]),
        int(seattle2020.shp.bounds[2]),
        int(seattle2020.shp.bounds[3]),
    )

    for bands in Sentinel2.ASSET_BANDS.values():
        array = data_source.read_raster(
            # layer_name is ignored
            layer_name="fake",
            item_name=item.name,
            bands=bands,
            projection=seattle2020.projection,
            bounds=bounds,
        )
        assert array.max() > 0 and array.shape == (
            len(bands),
            bounds[3] - bounds[1],
            bounds[2] - bounds[0],
        )


def test_harmonization_preapplied(seattle2020: STGeometry) -> None:
    """Verify AWS COGs are already harmonized.

    To do so, we use a known scene present in both AWS and Planetary Computer with the
    same processing baseline, and make sure that when we materialize it
    """
    # Item names for the known scene. The item name is unique to each data source since
    # neither source uses the standard Sentinel-2 scene ID.
    aws_item_name = "S2B_10TET_20260124_0_L2A"
    pc_item_name = "S2B_MSIL2A_20260124T191559_R056_T10TET_20260124T211343"
    aws_data_source = Sentinel2(assets=["red"])
    pc_source = PlanetaryComputerSentinel2(harmonize=True, assets=["B04"])

    projection = seattle2020.projection
    bounds = (
        int(seattle2020.shp.bounds[0]),
        int(seattle2020.shp.bounds[1]),
        int(seattle2020.shp.bounds[2]),
        int(seattle2020.shp.bounds[3]),
    )
    aws_array = aws_data_source.read_raster(
        "unused", aws_item_name, ["B04"], projection, bounds
    )
    pc_array = pc_source.read_raster(
        "unused", pc_item_name, ["B04"], projection, bounds
    )
    # There could be slight difference due to rounding during resampling but it should
    # almost be exact.
    assert np.all(np.abs(aws_array - pc_array) <= 1)
