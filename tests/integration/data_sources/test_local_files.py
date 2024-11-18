import json
import os
import pathlib

import pytest
import shapely
from rasterio.crs import CRS
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.dataset.manage import (
    ingest_dataset_windows,
    materialize_dataset_windows,
    prepare_dataset_windows,
)
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import Projection, STGeometry
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
        vector_format = load_vector_format(layer_config.format)  # type: ignore
        features = vector_format.decode_vector(
            window.path / "layers" / "local_file", window.bounds
        )

        assert len(features) == 2


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
        coordinate_mode: GeojsonCoordinateMode,
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
        vector_format = GeojsonVectorFormat(coordinate_mode=coordinate_mode)
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
        with open(os.path.join(tmp_path, "config.json"), "w") as f:
            json.dump(dataset_config, f)

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
