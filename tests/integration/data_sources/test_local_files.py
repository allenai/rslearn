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

    @pytest.mark.parametrize(
        "coordinate_mode", [GeojsonCoordinateMode.CRS, GeojsonCoordinateMode.WGS84]
    )
    def test_geojson_with_crs(
        self, tmp_path: pathlib.Path, coordinate_mode: GeojsonCoordinateMode
    ) -> None:
        seattle = STGeometry(WGS84_PROJECTION, shapely.Point(-122.3, 47.6), None)
        ds_path = UPath(tmp_path)

        # This time we overwrite the GeoJSON with one in a different CRS and make sure
        # it works no matter how the GeoJSON is encoded.
        custom_projection = Projection(CRS.from_epsg(3857), 10, -10)
        features = [
            Feature(
                geometry=seattle.to_projection(custom_projection),
            ),
        ]
        src_data_dir = ds_path / "src_data"
        src_data_dir.mkdir(parents=True, exist_ok=True)
        vector_format = GeojsonVectorFormat(coordinate_mode=coordinate_mode)
        vector_format.encode_vector(src_data_dir, custom_projection, features)

        # Primary purpose of this test is to make sure LocalFiles can import the data,
        # but here we also verify that it has been encoded based on the desired
        # coordinate mode.
        with (src_data_dir / "data.geojson").open() as f:
            fc = json.load(f)
            feat_x, feat_y = fc["features"][0]["geometry"]["coordinates"]
        if coordinate_mode == GeojsonCoordinateMode.CRS:
            assert feat_x == pytest.approx(
                features[0].geometry.shp.x * custom_projection.x_resolution
            )
            assert feat_y == pytest.approx(
                features[0].geometry.shp.y * custom_projection.y_resolution
            )
        elif coordinate_mode == GeojsonCoordinateMode.WGS84:
            assert feat_x == pytest.approx(seattle.shp.x)
            assert feat_y == pytest.approx(seattle.shp.y)

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

        # Make the window in a different projection.
        # First window has same coordinates but it's in a different projection so if
        # GeoJSON is handled correctly, it shouldn't match with any items.
        # Second window has the correctly aligned coordinates.
        window_projection = Projection(CRS.from_epsg(32610), 5, -5)

        window_center = seattle.to_projection(custom_projection).shp
        bad_window = Window(
            path=Window.get_window_root(ds_path, "default", "bad"),
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

        window_center = seattle.to_projection(window_projection).shp
        good_window = Window(
            path=Window.get_window_root(ds_path, "default", "good"),
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

        dataset = Dataset(ds_path)

        windows = dataset.load_windows()
        prepare_dataset_windows(dataset, windows)
        ingest_dataset_windows(dataset, windows)
        materialize_dataset_windows(dataset, windows)

        assert not (bad_window.path / "layers" / "local_file" / "data.geojson").exists()
        assert (good_window.path / "layers" / "local_file" / "data.geojson").exists()
