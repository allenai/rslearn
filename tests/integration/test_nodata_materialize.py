"""Integration test: nodata values persist through ingestion and materialization."""

import pathlib

import numpy as np
from shapely.geometry import Polygon
from upath import UPath

from rslearn.config.dataset import BandSetConfig, DType, LayerConfig, LayerType
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.data_source import Item
from rslearn.dataset.materialize import RasterMaterializer
from rslearn.dataset.storage.file import FileWindowStorage
from rslearn.dataset.window import Window
from rslearn.dataset.window_data_storage.per_item_group import PerItemGroupStorage
from rslearn.tile_stores.default import DefaultTileStore
from rslearn.tile_stores.tile_store import TileStoreWithLayer
from rslearn.utils.geometry import STGeometry
from rslearn.utils.raster_array import RasterArray, RasterMetadata
from rslearn.utils.raster_format import GeotiffRasterFormat

LAYER_NAME = "layer"
BANDS = ["band1"]
BOUNDS = (0, 0, 4, 4)
PROJECTION = WGS84_PROJECTION
NODATA_VAL = -32768.0


def test_nodata_persists_through_ingestion_and_materialization(
    tmp_path: pathlib.Path,
) -> None:
    """Write a raster with nodata metadata, run RasterMaterializer, read it back.

    Verifies that nodata_value propagates end-to-end:
      tile store write (ingestion) -> RasterMaterializer -> materialized GeoTIFF
    """
    ds_path = UPath(tmp_path)

    # Ingest: write a raster with nodata metadata into the tile store.
    tile_store = DefaultTileStore()
    tile_store.set_dataset_path(ds_path)

    bbox = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    item = Item("item", STGeometry(PROJECTION, bbox, None))

    src = np.ones((1, 4, 4), dtype=np.float32) * 42.0
    tile_store.write_raster(
        LAYER_NAME,
        item,
        BANDS,
        PROJECTION,
        BOUNDS,
        RasterArray(
            chw_array=src,
            metadata=RasterMetadata(nodata_value=NODATA_VAL),
        ),
    )

    # Sanity-check: tile store metadata has the nodata value.
    ts_with_layer = TileStoreWithLayer(tile_store, LAYER_NAME)
    assert ts_with_layer.get_raster_metadata(item, BANDS).nodata_value == NODATA_VAL

    # Materialize via RasterMaterializer (no explicit nodata_value in BandSetConfig).
    layer_cfg = LayerConfig(
        type=LayerType.RASTER,
        band_sets=[BandSetConfig(dtype=DType.FLOAT32, bands=BANDS)],
    )
    storage = FileWindowStorage(ds_path)
    window = Window(
        storage=storage,
        group="default",
        name="win",
        projection=PROJECTION,
        bounds=BOUNDS,
        time_range=None,
        data_storage=PerItemGroupStorage(),
    )
    window.save()

    RasterMaterializer().materialize(
        tile_store=ts_with_layer,
        window=window,
        layer_name=LAYER_NAME,
        layer_cfg=layer_cfg,
        item_groups=[[item]],
    )

    # -- Read back via Window.read_raster and verify nodata is set. --
    raster = window.read_raster(LAYER_NAME, BANDS, GeotiffRasterFormat())
    assert raster.metadata.nodata_value == NODATA_VAL
    np.testing.assert_array_equal(raster.get_chw_array(), 42.0)
