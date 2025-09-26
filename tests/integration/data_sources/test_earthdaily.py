import logging
import os
import pathlib
from datetime import UTC, datetime

import pytest
import shapely
from upath import UPath

from rslearn.config import (
    QueryConfig,
    SpaceMode,
)
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.earthdaily import EarthDaily, Sentinel1, Sentinel2
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils import STGeometry


pytestmark = pytest.mark.usefixtures("load_repository_env")

@pytest.fixture()
def edc_preview_geometry() -> STGeometry:
    return STGeometry(
        WGS84_PROJECTION,
        # These are the published spatial and temporal extents of the edc-preview collection.
        shapely.box(-105.14643, 39.53704, -104.76335, 39.95877),
        (
            datetime(2022, 8, 9, 18, 3, 33, tzinfo=UTC),
            datetime(2022, 8, 15, 18, 2, 42, tzinfo=UTC),
        ),
    )


def test_earthdaily_credentials_present() -> None:
    required = ("EDS_CLIENT_ID", "EDS_SECRET", "EDS_AUTH_URL", "EDS_API_URL")
    missing = [key for key in required if not os.getenv(key)]
    assert not missing, f"Missing EarthDaily credentials: {', '.join(missing)}"

    logger = logging.getLogger(__name__)
    logger.info(
        "EarthDaily endpoints: auth=%s, api=%s",
        os.getenv("EDS_AUTH_URL"),
        os.getenv("EDS_API_URL"),
    )


def test_edc_preview(tmp_path: pathlib.Path, edc_preview_geometry: STGeometry) -> None:
    """Test ingesting an item corresponding to test_geometry to local filesystem."""
    band_name = "NIR"
    data_source = EarthDaily(
        collection_name="edc-preview", asset_bands={"image_file_NIR": [band_name]}
    )

    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([edc_preview_geometry], query_config)[0]
    item = item_groups[0][0]

    tile_store_dir = UPath(tmp_path)
    tile_store = DefaultTileStore(str(tile_store_dir))
    tile_store.set_dataset_path(tile_store_dir)

    layer_name = "layer"
    data_source.ingest(
        TileStoreWithLayer(tile_store, layer_name),
        item_groups[0],
        [[edc_preview_geometry]],
    )
    assert tile_store.is_raster_ready(layer_name, item.name, [band_name])


def test_sentinel1(tmp_path: pathlib.Path, seattle2020: STGeometry) -> None:
    """Test ingesting a Sentinel-1 item corresponding to seattle2020."""
    band_name = "vv"
    data_source = Sentinel1(band_names=[band_name])

    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([seattle2020], query_config)[0]
    item = item_groups[0][0]

    tile_store_dir = UPath(tmp_path)
    tile_store = DefaultTileStore(str(tile_store_dir))
    tile_store.set_dataset_path(tile_store_dir)

    layer_name = "layer"
    data_source.ingest(
        TileStoreWithLayer(tile_store, layer_name),
        item_groups[0],
        [[seattle2020]],
    )
    assert tile_store.is_raster_ready(layer_name, item.name, [band_name])


def test_sentinel2(tmp_path: pathlib.Path, edc_preview_geometry: STGeometry) -> None:
    """Test ingesting a Sentinel-2 item corresponding to the edc preview region."""
    band_name = "B04"
    data_source = Sentinel2(assets=[band_name])

    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([edc_preview_geometry], query_config)[0]
    assert item_groups, "Expected at least one matching Sentinel-2 item"
    item = item_groups[0][0]

    tile_store_dir = UPath(tmp_path)
    tile_store = DefaultTileStore(str(tile_store_dir))
    tile_store.set_dataset_path(tile_store_dir)

    layer_name = "layer"
    data_source.ingest(
        TileStoreWithLayer(tile_store, layer_name),
        item_groups[0],
        [[edc_preview_geometry]],
    )
    assert tile_store.is_raster_ready(layer_name, item.name, [band_name])


def test_cache_dir(tmp_path: pathlib.Path, edc_preview_geometry: STGeometry) -> None:
    """Make sure cache directory is populated when set."""
    # Use a subdirectory so we also ensure the directory is automatically created.
    cache_dir = UPath(tmp_path / "cache_dir")
    band_name = "NIR"
    data_source = EarthDaily(
        collection_name="edc-preview",
        asset_bands={"image_file_NIR": [band_name]},
        cache_dir=cache_dir,
    )
    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    data_source.get_items([edc_preview_geometry], query_config)[0]
    assert len(list(cache_dir.iterdir())) > 0
