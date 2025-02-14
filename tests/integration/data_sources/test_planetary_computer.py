import pathlib

from upath import UPath

from rslearn.config import (
    QueryConfig,
    SpaceMode,
)
from rslearn.data_sources.planetary_computer import Sentinel1, Sentinel2
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils import STGeometry


def test_sentinel1(tmp_path: pathlib.Path, seattle2020: STGeometry) -> None:
    """Test ingesting an item corresponding to seattle2020 to local filesystem."""
    band_name = "vv"
    # The asset band is vv but in the STAC metadata it is capitalized.
    # We search for a VV+VH image since that is the standard one for GRD/IW.
    s1_query_dict = {"sar:polarizations": {"eq": ["VV", "VH"]}}
    data_source = Sentinel1(
        band_names=[band_name],
        query=s1_query_dict,
    )

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


def test_sentinel2(tmp_path: pathlib.Path, seattle2020: STGeometry) -> None:
    """Test ingesting an item corresponding to seattle2020 to local filesystem."""
    band_name = "B04"
    data_source = Sentinel2(assets=[band_name])

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
