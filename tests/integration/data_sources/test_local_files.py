import json
import os
import pathlib

import pytest
import shapely
from rasterio.crs import CRS
from upath import UPath

from rslearn.config import VectorLayerConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.dataset.manage import (
    ingest_dataset_windows,
    materialize_dataset_windows,
    prepare_dataset_windows,
)
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.vector_format import (
    GeojsonCoordinateMode,
    GeojsonVectorFormat,
    load_vector_format,
)


class TestLocalFiles:
    def test_sample_dataset(self, local_files_dataset: Dataset) -> None:
        # 1. Create GeoJSON as a local file to extract data from.
        # 2. Create a corresponding dataset config file.
        # 3. Create a window intersecting the features.
        # 4. Run prepare, ingest, materialize, and make sure it gets the features.
        windows = local_files_dataset.load_windows()
        prepare_dataset_windows(local_files_dataset, windows)
        ingest_dataset_windows(local_files_dataset, windows)
        materialize_dataset_windows(local_files_dataset, windows)

        assert len(windows) == 1

        window = windows[0]
        layer_config = local_files_dataset.layers["local_file"]
        assert isinstance(layer_config, VectorLayerConfig)
        vector_format = load_vector_format(layer_config.format)
        features = vector_format.decode_vector(
            window.path / "layers" / "local_file", window.bounds
        )

        assert len(features) == 2

    def test_large_dataset(self, tmp_path: pathlib.Path) -> None:
        """Test that LocalFiles successfully handles a large source dataset.

        There were previously issues when the source dataset spanned the entire world
        since the corners of the item's geometry would be so far apart that after
        re-projection they may actually not even intersect the bounds of the target
        projection.
        """
        ds_path = UPath(tmp_path)

        # Create a GeoJSON with three features.
        # The first two features are to make the bounds of the GeoJSON big (which is
        # the source of these problems) while the third is the one we will look for
        # after materialization.
        target_geometry = STGeometry(WGS84_PROJECTION, shapely.Point(0.5, 0.5), None)
        features = [
            Feature(STGeometry(WGS84_PROJECTION, shapely.Point(-179, -89), None)),
            Feature(STGeometry(WGS84_PROJECTION, shapely.Point(179, 89), None)),
            Feature(target_geometry, {"check": True}),
        ]
        src_data_dir = os.path.join(tmp_path, "src_data")
        os.makedirs(src_data_dir)
        with open(os.path.join(src_data_dir, "data.geojson"), "w") as f:
            json.dump(
                {
                    "type": "FeatureCollection",
                    "features": [feat.to_geojson() for feat in features],
                },
                f,
            )

        # Make an rslearn dataset that uses LocalFiles to ingest those features.
        dataset_config = {
            "layers": {
                "local_file": {
                    "type": "vector",
                    "data_source": {
                        "name": "rslearn.data_sources.local_files.LocalFiles",
                        "src_dir": src_data_dir,
                    },
                },
            },
        }
        with (ds_path / "config.json").open("w") as f:
            json.dump(dataset_config, f)

        # Create UTM window where we expect the geometry.
        utm_proj = get_utm_ups_projection(
            target_geometry.shp.x, target_geometry.shp.y, 10, -10
        )
        dst_geometry = target_geometry.to_projection(utm_proj)
        dst_bounds = (
            int(dst_geometry.shp.x) - 4,
            int(dst_geometry.shp.y) - 4,
            int(dst_geometry.shp.x) + 4,
            int(dst_geometry.shp.y) + 4,
        )
        Window(
            path=Window.get_window_root(ds_path, "default", "default"),
            group="default",
            name="default",
            projection=utm_proj,
            bounds=dst_bounds,
            time_range=None,
        ).save()

        # Now materialize the windows and check that it was done correctly.
        dataset = Dataset(ds_path)
        windows = dataset.load_windows()
        prepare_dataset_windows(dataset, windows)
        ingest_dataset_windows(dataset, windows)
        materialize_dataset_windows(dataset, windows)

        window = windows[0]
        layer_config = dataset.layers["local_file"]
        assert isinstance(layer_config, VectorLayerConfig)
        vector_format = load_vector_format(layer_config.format)
        features = vector_format.decode_vector(
            window.path / "layers" / "local_file", window.bounds
        )
        assert len(features) == 1
        assert features[0].properties is not None
        assert features[0].properties["check"]


class TestCoordinateModes:
    """Test LocalFiles again, focusing on using different coordinate modes.

    We can only use CRS and WGS84 modes since PIXEL doesn't actually produce a GeoJSON
    that tools like fiona can understand.
    """

    source_data_projection = Projection(CRS.from_epsg(3857), 10, -10)

    @pytest.fixture
    def seattle_point(self) -> STGeometry:
        return STGeometry(WGS84_PROJECTION, shapely.Point(-122.3, 47.6), None)

    @pytest.fixture(params=[GeojsonCoordinateMode.CRS, GeojsonCoordinateMode.WGS84])
    def vector_ds_path(
        self,
        tmp_path: pathlib.Path,
        seattle_point: STGeometry,
        request: pytest.FixtureRequest,
    ) -> UPath:
        # Make a vector dataset with one point in EPSG:3857.
        # We will use it to check that it intersects correctly with
        ds_path = UPath(tmp_path)

        features = [
            Feature(
                geometry=seattle_point.to_projection(self.source_data_projection),
            ),
        ]
        src_data_dir = ds_path / "src_data"
        src_data_dir.mkdir(parents=True, exist_ok=True)
        vector_format = GeojsonVectorFormat(coordinate_mode=request.param)
        vector_format.encode_vector(src_data_dir, self.source_data_projection, features)

        dataset_config = {
            "layers": {
                "local_file": {
                    "type": "vector",
                    "data_source": {
                        "name": "rslearn.data_sources.local_files.LocalFiles",
                        "src_dir": src_data_dir.path,
                    },
                },
            },
            "tile_store": {
                "name": "file",
                "root_dir": "tiles",
            },
        }
        with (ds_path / "config.json").open("w") as f:
            json.dump(dataset_config, f)

        return ds_path

    def test_matching_units_in_wrong_crs(
        self, seattle_point: STGeometry, vector_ds_path: UPath
    ) -> None:
        # Here we make a window that has the same coordinates as the dataset's source
        # data, but it is actually in a different CRS.
        # So it shouldn't match with anything.
        window_projection = Projection(CRS.from_epsg(32610), 5, -5)
        window_center = seattle_point.to_projection(self.source_data_projection).shp
        bad_window = Window(
            path=Window.get_window_root(vector_ds_path, "default", "bad"),
            group="default",
            name="bad",
            projection=window_projection,
            bounds=(
                int(window_center.x) - 10,
                int(window_center.y) - 10,
                int(window_center.x) + 10,
                int(window_center.y) + 10,
            ),
            time_range=None,
        )
        bad_window.save()

        dataset = Dataset(vector_ds_path)
        windows = dataset.load_windows()
        prepare_dataset_windows(dataset, windows)
        ingest_dataset_windows(dataset, windows)
        materialize_dataset_windows(dataset, windows)
        assert not (bad_window.path / "layers" / "local_file" / "data.geojson").exists()

    def test_match_in_different_crs(
        self, seattle_point: STGeometry, vector_ds_path: UPath
    ) -> None:
        # Now create a window again in EPSG:32610 but it has the right units to match
        # with the point.
        window_projection = Projection(CRS.from_epsg(32610), 5, -5)
        window_center = seattle_point.to_projection(window_projection).shp
        good_window = Window(
            path=Window.get_window_root(vector_ds_path, "default", "good"),
            group="default",
            name="good",
            projection=window_projection,
            bounds=(
                int(window_center.x) - 10,
                int(window_center.y) - 10,
                int(window_center.x) + 10,
                int(window_center.y) + 10,
            ),
            time_range=None,
        )
        good_window.save()

        dataset = Dataset(vector_ds_path)
        windows = dataset.load_windows()
        prepare_dataset_windows(dataset, windows)
        ingest_dataset_windows(dataset, windows)
        materialize_dataset_windows(dataset, windows)
        assert (good_window.path / "layers" / "local_file" / "data.geojson").exists()
