"""Integration tests for Google Satellite Embedding V1 data source on AWS."""

import pathlib

import numpy as np
from upath import UPath

from rslearn.config import QueryConfig, SpaceMode
from rslearn.data_sources.aws_google_satellite_embedding_v1 import (
    BANDS,
    GoogleSatelliteEmbeddingV1,
)
from rslearn.utils import STGeometry


def test_read_raster_subset_bands(
    tmp_path: pathlib.Path, seattle2020: STGeometry
) -> None:
    """Test reading a subset of bands."""
    cache_dir = UPath(tmp_path / "cache")
    data_source = GoogleSatelliteEmbeddingV1(metadata_cache_dir=str(cache_dir))

    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([seattle2020], query_config)[0]
    item = item_groups[0][0]

    layer_name = "gse"
    projection = seattle2020.projection
    bounds = (
        int(seattle2020.shp.bounds[0]),
        int(seattle2020.shp.bounds[1]),
        int(seattle2020.shp.bounds[2]),
        int(seattle2020.shp.bounds[3]),
    )

    # Read only first 3 bands
    subset_bands = ["A00", "A01", "A02"]
    data = data_source.read_raster(
        layer_name=layer_name,
        item_name=item.name,
        bands=subset_bands,
        projection=projection,
        bounds=bounds,
    )

    expected_height = bounds[3] - bounds[1]
    expected_width = bounds[2] - bounds[0]
    assert data.shape == (3, expected_height, expected_width)
    # Check that it was dequantized to float32 correctly. Should be roughly [-1, 1].
    assert data.dtype == np.float32
    assert data.min() >= -1.5
    assert data.max() <= 1.5


def test_read_raster_no_dequantization(
    tmp_path: pathlib.Path, seattle2020: STGeometry
) -> None:
    """Test reading raw data without dequantization."""
    cache_dir = UPath(tmp_path / "cache")

    data_source = GoogleSatelliteEmbeddingV1(
        metadata_cache_dir=str(cache_dir), apply_dequantization=False
    )

    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS)
    item_groups = data_source.get_items([seattle2020], query_config)[0]
    item = item_groups[0][0]

    layer_name = "gse"
    projection = seattle2020.projection
    bounds = (
        int(seattle2020.shp.bounds[0]),
        int(seattle2020.shp.bounds[1]),
        int(seattle2020.shp.bounds[2]),
        int(seattle2020.shp.bounds[3]),
    )

    data = data_source.read_raster(
        layer_name=layer_name,
        item_name=item.name,
        bands=BANDS,
        projection=projection,
        bounds=bounds,
    )

    print(f"Raw data shape: {data.shape}")
    print(f"Raw data dtype: {data.dtype}")
    print(f"Raw data range: [{data.min()}, {data.max()}]")

    # Raw data should be integer type (int8 or int16 depending on resampling)
    assert np.issubdtype(data.dtype, np.integer)

    # Values should be in [-127, 127] range (the original int8 range)
    assert -150 < data.min() < -50
    assert 150 > data.max() > 50
