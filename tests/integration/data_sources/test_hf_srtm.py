import pathlib

from upath import UPath

from rslearn.config import QueryConfig, SpaceMode
from rslearn.data_sources.hf_srtm import SRTM
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils.geometry import STGeometry


def test_hf_srtm_use_srtm1(tmp_path: pathlib.Path, seattle2020: STGeometry) -> None:
    """Test ingesting SRTM1 (1 arc-second) data for Seattle."""
    band_name = "dem"
    data_source = SRTM()  # Default prefers SRTM1

    query_config = QueryConfig(space_mode=SpaceMode.MOSAIC, max_matches=1)
    item_groups = data_source.get_items([seattle2020], query_config)[0]
    item = item_groups[0][0]

    # Seattle should get SRTM1 (higher resolution available for US)
    assert "SRTM1" in item.name

    tile_store_dir = UPath(tmp_path)
    tile_store = DefaultTileStore(str(tile_store_dir))
    tile_store.set_dataset_path(tile_store_dir)

    layer_name = "layer"
    data_source.ingest(
        TileStoreWithLayer(tile_store, layer_name), item_groups[0], [[seattle2020]]
    )
    assert tile_store.is_raster_ready(layer_name, item.name, [band_name])


def test_hf_srtm_use_srtm3(tmp_path: pathlib.Path, seattle2020: STGeometry) -> None:
    """Test ingesting SRTM3 (3 arc-second) data for Seattle."""
    band_name = "dem"
    data_source = SRTM(always_use_3arcsecond=True)

    query_config = QueryConfig(space_mode=SpaceMode.MOSAIC, max_matches=1)
    item_groups = data_source.get_items([seattle2020], query_config)[0]
    item = item_groups[0][0]

    # With always_use_3arcsecond=True, should get SRTM3
    assert "SRTM3" in item.name

    tile_store_dir = UPath(tmp_path)
    tile_store = DefaultTileStore(str(tile_store_dir))
    tile_store.set_dataset_path(tile_store_dir)

    layer_name = "layer"
    data_source.ingest(
        TileStoreWithLayer(tile_store, layer_name), item_groups[0], [[seattle2020]]
    )
    assert tile_store.is_raster_ready(layer_name, item.name, [band_name])


def test_hf_srtm_with_cache_dir(
    tmp_path: pathlib.Path, seattle2020: STGeometry
) -> None:
    """Test ingesting with file list cache enabled."""
    band_name = "dem"
    cache_dir = tmp_path / "cache"
    data_source = SRTM(cache_dir=str(cache_dir))

    query_config = QueryConfig(space_mode=SpaceMode.MOSAIC, max_matches=1)
    item_groups = data_source.get_items([seattle2020], query_config)[0]
    item = item_groups[0][0]

    tile_store_dir = UPath(tmp_path / "tiles")
    tile_store = DefaultTileStore(str(tile_store_dir))
    tile_store.set_dataset_path(tile_store_dir)

    layer_name = "layer"
    data_source.ingest(
        TileStoreWithLayer(tile_store, layer_name), item_groups[0], [[seattle2020]]
    )
    assert tile_store.is_raster_ready(layer_name, item.name, [band_name])
    assert (cache_dir / data_source.FILE_LIST_FILENAME).exists()
