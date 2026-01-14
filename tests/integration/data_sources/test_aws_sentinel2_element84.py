import pathlib

import numpy as np
from rasterio.crs import CRS
from upath import UPath

from rslearn.config import (
    QueryConfig,
    SpaceMode,
)
from rslearn.data_sources.aws_sentinel2_element84 import Sentinel2
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils.geometry import Projection, STGeometry


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


def test_harmonize_applied() -> None:
    """Make sure harmonize is applied for scenes that need it."""
    harmonized_data_source = Sentinel2(harmonize=True)
    unharmonized_data_source = Sentinel2()
    # This is a scene that is processed using baseline and needs harmonization.
    stac_id = "S2B_56MPC_20180101_1_L2A"
    item = harmonized_data_source.get_item_by_name(stac_id)
    assert item.properties["earthsearch:boa_offset_applied"]

    # Get bounds of the raster in WebMercator and read a subset of it.
    projection = Projection(CRS.from_epsg(3857), 10, -10)
    bounds = harmonized_data_source.get_raster_bounds(
        "unused", item.name, ["B04"], projection
    )
    center = ((bounds[0] + bounds[2]) // 2, (bounds[1] + bounds[3]) // 2)
    read_bounds = (
        center[0] - 8,
        center[1] - 8,
        center[0] + 8,
        center[1] + 8,
    )
    harmonized_array = harmonized_data_source.read_raster(
        "unused", item.name, ["B04"], projection, read_bounds
    )
    unharmonized_array = unharmonized_data_source.read_raster(
        "unused", item.name, ["B04"], projection, read_bounds
    )
    assert np.all(harmonized_array == (np.clip(unharmonized_array, 1000, None) - 1000))


def test_harmonize_not_applied() -> None:
    """Make sure harmonize is not applied if the scene is processed using old baseline."""
    harmonized_data_source = Sentinel2(harmonize=True)
    unharmonized_data_source = Sentinel2()
    # This scene is processed using old baseline so its values should not be changed.
    stac_id = "S2B_56KKE_20180101_0_L2A"
    item = harmonized_data_source.get_item_by_name(stac_id)
    assert not item.properties["earthsearch:boa_offset_applied"]

    # Get bounds of the raster in WebMercator and read a subset of it.
    projection = Projection(CRS.from_epsg(3857), 10, -10)
    bounds = harmonized_data_source.get_raster_bounds(
        "unused", item.name, ["B04"], projection
    )
    center = ((bounds[0] + bounds[2]) // 2, (bounds[1] + bounds[3]) // 2)
    read_bounds = (
        center[0] - 8,
        center[1] - 8,
        center[0] + 8,
        center[1] + 8,
    )
    harmonized_array = harmonized_data_source.read_raster(
        "unused", item.name, ["B04"], projection, read_bounds
    )
    unharmonized_array = unharmonized_data_source.read_raster(
        "unused", item.name, ["B04"], projection, read_bounds
    )
    assert np.all(harmonized_array == unharmonized_array)
