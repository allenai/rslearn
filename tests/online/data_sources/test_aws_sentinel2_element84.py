import pathlib

import numpy as np
import pytest
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
        raster_array = data_source.read_raster(
            # layer_name is ignored
            layer_name="fake",
            item_name=item.name,
            bands=bands,
            projection=seattle2020.projection,
            bounds=bounds,
        )
        array = raster_array.get_chw_array()
        assert array.max() > 0 and array.shape == (
            len(bands),
            bounds[3] - bounds[1],
            bounds[2] - bounds[0],
        )


@pytest.mark.parametrize(
    ("aws_item_name", "pc_item_name"),
    [
        (
            "S2B_10TET_20260124_0_L2A",
            "S2B_MSIL2A_20260124T191559_R056_T10TET_20260124T211343",
        ),
        (
            "S2A_10TET_20210624_0_L2A",
            "S2A_MSIL2A_20210624T190921_R056_T10TET_20210625T214830",
        ),
    ],
    ids=["jan_2026_baseline_05_11", "jun_2021_baseline_03_00"],
)
def test_harmonization_preapplied(
    seattle2020: STGeometry,
    aws_item_name: str,
    pc_item_name: str,
) -> None:
    """Verify AWS COGs are already harmonized.

    To do so, we use known scenes present in both AWS and Planetary Computer under the
    same processing baseline, and make sure that when we materialize it we get the same
    pixel values.
    """
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
    ).get_chw_array()
    pc_array = pc_source.read_raster(
        "unused", pc_item_name, ["B04"], projection, bounds
    ).get_chw_array()
    # The 2026 scenes are very similar (all pixels are at most 1 off). But the 2021
    # scenes seem to have differences despite being from the same processing baseline.
    # I checked the raw GeoTIFFs from each source and they do differ, so it probably
    # relates to how one of them used to process the original JP2 files into COGs.
    # They are still mostly the same though, so here we check that at most 10% of
    # pixels are more than 200 off from each other (much tighter than the 1000 that
    # would come from harmonization).
    assert (
        np.count_nonzero(
            np.abs(aws_array.astype(np.int32) - pc_array.astype(np.int32)) > 200
        )
        / aws_array.size
        < 0.1
    )
