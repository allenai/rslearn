"""Online tests for the GCP Landsat data source.

These hit the requester-pays gs://gee-public-data-landsat bucket and the
earth-engine-public-data BigQuery index, so they require GCP credentials
(GOOGLE_APPLICATION_CREDENTIALS) and GS_USER_PROJECT to be set.
"""

import pathlib
from datetime import UTC, datetime

import pytest
import shapely
from rasterio import CRS
from upath import UPath

from rslearn.config import QueryConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.gcp_landsat import (
    Landsat,
    ProcessingLevel,
    SensorId,
    SpacecraftId,
)
from rslearn.tile_stores import DefaultTileStore, TileStoreWithLayer
from rslearn.utils import Projection, STGeometry


def _seattle_geometry(
    time_range: tuple[datetime, datetime],
) -> STGeometry:
    """Build a small Seattle-area geometry (UTM 10N, 30 m) for the time range."""
    wgs84_geom = STGeometry(WGS84_PROJECTION, shapely.Point(-122.33, 47.61), time_range)
    dst_projection = Projection(CRS.from_epsg(32610), 30, -30)
    point = wgs84_geom.to_projection(dst_projection).shp
    size = 64
    box = shapely.box(
        point.x - size // 2,
        point.y - size // 2,
        point.x + size // 2,
        point.y + size // 2,
    )
    return STGeometry(dst_projection, box, time_range)


class TestLandsat:
    """Tests ingesting different Landsat scene types from GCP."""

    @pytest.mark.parametrize(
        ("spacecraft_id", "sensor_id", "processing_level", "bands", "time_range"),
        [
            # Landsat 5 TM, Level-1 (legacy mission, B1-B7).
            (
                SpacecraftId.LANDSAT_5,
                SensorId.TM,
                ProcessingLevel.L1TP,
                ["B3"],
                (
                    datetime(2010, 6, 1, tzinfo=UTC),
                    datetime(2010, 10, 1, tzinfo=UTC),
                ),
            ),
            # Landsat 9 OLI-TIRS, Level-1.
            (
                SpacecraftId.LANDSAT_9,
                SensorId.OLI_TIRS,
                ProcessingLevel.L1TP,
                ["B4"],
                (
                    datetime(2022, 6, 1, tzinfo=UTC),
                    datetime(2022, 10, 1, tzinfo=UTC),
                ),
            ),
            # Landsat 9 OLI-TIRS, Level-2: exercises the SR_/ST_ band naming
            # (B4 -> SR_B4 surface reflectance, B10 -> ST_B10 surface temp).
            (
                SpacecraftId.LANDSAT_9,
                SensorId.OLI_TIRS,
                ProcessingLevel.L2SP,
                ["B4", "B10"],
                (
                    datetime(2022, 6, 1, tzinfo=UTC),
                    datetime(2022, 10, 1, tzinfo=UTC),
                ),
            ),
        ],
        ids=["landsat5_l1", "landsat9_l1", "landsat9_l2sp"],
    )
    def test_ingest(
        self,
        tmp_path: pathlib.Path,
        spacecraft_id: SpacecraftId,
        sensor_id: SensorId,
        processing_level: ProcessingLevel,
        bands: list[str],
        time_range: tuple[datetime, datetime],
    ) -> None:
        """Ingest a representative band for each scene type to a local tile store."""
        geometry = _seattle_geometry(time_range)
        data_source = Landsat(
            index_cache_dir=str(UPath(tmp_path) / "cache"),
            sensor_ids=[sensor_id],
            processing_levels=[processing_level],
            spacecraft_ids=[spacecraft_id],
            bands=bands,
            sort_by="cloud_cover",
            use_rtree_index=False,
        )

        query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS, max_matches=1)
        item_groups = data_source.get_items([geometry], query_config)[0]
        assert len(item_groups) > 0, "expected at least one matching scene"
        item = item_groups[0].items[0]
        assert item.spacecraft_id == spacecraft_id
        assert item.sensor_id == sensor_id
        assert item.processing_level == processing_level

        tile_store_dir = UPath(tmp_path) / "tiles"
        tile_store = DefaultTileStore(str(tile_store_dir))
        tile_store.set_dataset_path(tile_store_dir)
        layer_name = "layer"
        data_source.ingest(
            TileStoreWithLayer(tile_store, layer_name),
            item_groups[0].items,
            [[geometry]],
        )
        for band in bands:
            assert tile_store.is_raster_ready(layer_name, item, [band])
