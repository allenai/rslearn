import json
import pathlib
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from typing import Any

import google.api_core.exceptions
import pytest
import shapely
from upath import UPath

import rslearn.main
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.gcp_public_data import Sentinel2Item
from rslearn.data_sources.local_files import VectorItem
from rslearn.dataset import Dataset, Window, WindowLayerData
from rslearn.log_utils import get_logger
from rslearn.utils.geometry import STGeometry

logger = get_logger(__name__)


class TestIngestion:
    @pytest.fixture
    def prepared_dataset(self, tmp_path: pathlib.Path) -> Dataset:
        # Create a dataset that is ready for ingestion.
        # It has two layers:
        # - local_files: LocalFiles dataset where ingestion should succeed.
        # - sentinel2: Sentinel2 dataset where items.json has unknown scene name so
        #   ingestion should fail.

        # First make a vector file for LocalFiles to ingest from.
        logger.info("make vector file")
        src_fname = UPath(tmp_path) / "files" / "data.geojson"
        src_fname.parent.mkdir(parents=True)
        with src_fname.open("w") as f:
            json.dump(
                {
                    "type": "FeatureCollection",
                    "properties": {},
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {},
                            "geometry": {
                                "type": "Point",
                                "coordinates": [0, 0],
                            },
                        }
                    ],
                },
                f,
            )

        # Now create the dataset configuration.
        logger.info("make dataset")
        ds_path = UPath(tmp_path) / "dataset"
        ds_path.mkdir(parents=True)
        ds_config = {
            "layers": {
                "local_files": {
                    "type": "vector",
                    "data_source": {
                        "class_path": "rslearn.data_sources.local_files.LocalFiles",
                        "init_args": {
                            "src_dir": str(src_fname.parent),
                        },
                    },
                },
                "sentinel2": {
                    "type": "raster",
                    "band_sets": [
                        {
                            "bands": ["R", "G", "B"],
                            "dtype": "uint8",
                        }
                    ],
                    "data_source": {
                        "class_path": "rslearn.data_sources.gcp_public_data.Sentinel2",
                        "init_args": {
                            "index_cache_dir": "cache",
                            "use_rtree_index": False,
                        },
                    },
                },
            },
        }
        with (ds_path / "config.json").open("w") as f:
            json.dump(ds_config, f)

        # Add window intersecting the vector data file.
        logger.info("make window")
        window = Window(
            path=Window.get_window_root(ds_path, "default", "default"),
            group="default",
            name="default",
            projection=WGS84_PROJECTION,
            bounds=(-1, -1, 1, 1),
            time_range=(
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 2, 1, tzinfo=UTC),
            ),
        )
        window.save()

        # Manually set the window's items.json.
        item_geom = STGeometry(
            WGS84_PROJECTION, shapely.box(*window.bounds), window.time_range
        )
        local_files_item = VectorItem(
            name="foo", geometry=item_geom, path_uri=str(src_fname)
        )
        sentinel2_item = Sentinel2Item(
            name="foo", geometry=item_geom, blob_prefix="bad-path", cloud_cover=0
        )
        layer_datas = {
            "local_files": WindowLayerData(
                "local_files", [[local_files_item.serialize()]]
            ),
            "sentinel2": WindowLayerData("sentinel2", [[sentinel2_item.serialize()]]),
        }
        window.save_layer_datas(layer_datas)

        return Dataset(ds_path)

    @pytest.fixture
    def ingested_fname(self, prepared_dataset: Dataset) -> UPath:
        return prepared_dataset.path / "tiles" / "local_files" / "foo" / "data.geojson"

    def test_normal_ingest_fails(
        self, prepared_dataset: Dataset, monkeypatch: Any
    ) -> None:
        # Trying to ingest both layers should fail because Sentinel-2 layer is messed up.
        mock_args = [
            "rslearn",
            "dataset",
            "ingest",
            "--root",
            str(prepared_dataset.path),
        ]
        monkeypatch.setattr(sys, "argv", mock_args)

        with pytest.raises(google.api_core.exceptions.NotFound):
            rslearn.main.main()

    def test_ingest_one_disabled(
        self, prepared_dataset: Dataset, ingested_fname: UPath, monkeypatch: Any
    ) -> None:
        # Ingestion should succeed if we disable the Sentinel-2 layer.
        mock_args = [
            "rslearn",
            "dataset",
            "ingest",
            "--root",
            str(prepared_dataset.path),
            "--disabled-layers",
            "sentinel2",
        ]
        monkeypatch.setattr(sys, "argv", mock_args)

        rslearn.main.main()

        assert ingested_fname.exists()

    def test_ingest_all_disabled(
        self, prepared_dataset: Dataset, ingested_fname: UPath, monkeypatch: Any
    ) -> None:
        mock_args = [
            "rslearn",
            "dataset",
            "ingest",
            "--root",
            str(prepared_dataset.path),
            "--disabled-layers",
            "local_files,sentinel2",
        ]
        monkeypatch.setattr(sys, "argv", mock_args)

        rslearn.main.main()

        assert not ingested_fname.exists()

    def test_ingestion_from_command_line(
        self, prepared_dataset: Dataset, ingested_fname: UPath, monkeypatch: Any
    ) -> None:
        args = [
            "rslearn",
            "dataset",
            "ingest",
            "--root",
            str(prepared_dataset.path),
            "--disabled-layers",
            "sentinel2",
        ]
        monkeypatch.setattr(sys, "argv", args)

        subprocess.run(args, check=True)

        assert ingested_fname.exists()

    def test_ingest_ignore_errors(
        self, prepared_dataset: Dataset, ingested_fname: UPath, monkeypatch: Any
    ) -> None:
        # Should also succeed if we ignore errors.
        mock_args = [
            "rslearn",
            "dataset",
            "ingest",
            "--root",
            str(prepared_dataset.path),
            "--ignore-errors",
        ]
        monkeypatch.setattr(sys, "argv", mock_args)

        rslearn.main.main()

        assert ingested_fname.exists()


class TestMaterialization:
    def prepared_dataset(self, path: pathlib.Path) -> Dataset:
        # First make a vector file for LocalFiles to ingest from.
        logger.info("make vector file")
        src_fname = UPath(path) / "files" / "data.geojson"
        src_fname.parent.mkdir(parents=True)
        with src_fname.open("w") as f:
            json.dump(
                {
                    "type": "FeatureCollection",
                    "properties": {},
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {},
                            "geometry": {
                                "type": "Point",
                                "coordinates": [0, 0],
                            },
                        }
                    ],
                },
                f,
            )

        # Now create the dataset configuration.
        logger.info("make dataset")
        ds_path = UPath(path) / "dataset"
        ds_path.mkdir(parents=True)
        ds_config = {
            "layers": {
                "local_files": {
                    "type": "vector",
                    "data_source": {
                        "class_path": "rslearn.data_sources.local_files.LocalFiles",
                        "init_args": {
                            "src_dir": str(src_fname.parent),
                        },
                    },
                },
                "sentinel2": {
                    "type": "raster",
                    "band_sets": [
                        {
                            "bands": ["R", "G", "B"],
                            "dtype": "uint8",
                        }
                    ],
                    "data_source": {
                        "class_path": "rslearn.data_sources.gcp_public_data.Sentinel2",
                        "init_args": {
                            "index_cache_dir": "cache",
                            "use_rtree_index": False,
                        },
                    },
                },
            },
        }
        with (ds_path / "config.json").open("w") as f:
            json.dump(ds_config, f)

        # Add window intersecting the vector data file.
        logger.info("make window")
        # First window
        window1 = Window(
            path=Window.get_window_root(ds_path, "default", "default"),
            group="default",
            name="default",
            projection=WGS84_PROJECTION,
            bounds=(-1, -1, 1, 1),
            time_range=(
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 2, 1, tzinfo=UTC),
            ),
        )
        window1.save()

        # Second window
        window2 = Window(
            path=Window.get_window_root(ds_path, "default", "window2"),
            group="default",
            name="window2",
            projection=WGS84_PROJECTION,
            bounds=(0, 0, 2, 2),  # Different bounds
            time_range=(
                datetime(2024, 2, 1, tzinfo=UTC),
                datetime(2024, 3, 1, tzinfo=UTC),  # Different time range
            ),
        )
        window2.save()

        # Create items and layer data for both windows
        item_geom1 = STGeometry(
            WGS84_PROJECTION, shapely.box(*window1.bounds), window1.time_range
        )
        item_geom2 = STGeometry(
            WGS84_PROJECTION, shapely.box(*window2.bounds), window2.time_range
        )

        # Items for window1
        local_files_item1 = VectorItem(
            name="foo", geometry=item_geom1, path_uri=str(src_fname)
        )
        sentinel2_item1 = Sentinel2Item(
            name="foo", geometry=item_geom1, blob_prefix="bad-path", cloud_cover=0
        )

        # Items for window2
        local_files_item2 = VectorItem(
            name="bar", geometry=item_geom2, path_uri=str(src_fname)
        )
        sentinel2_item2 = Sentinel2Item(
            name="bar", geometry=item_geom2, blob_prefix="bad-path", cloud_cover=0
        )

        # Save layer data for both windows
        window1.save_layer_datas(
            {
                "local_files": WindowLayerData(
                    "local_files", [[local_files_item1.serialize()]]
                ),
                "sentinel2": WindowLayerData(
                    "sentinel2", [[sentinel2_item1.serialize()]]
                ),
            }
        )

        window2.save_layer_datas(
            {
                "local_files": WindowLayerData(
                    "local_files", [[local_files_item2.serialize()]]
                ),
                "sentinel2": WindowLayerData(
                    "sentinel2", [[sentinel2_item2.serialize()]]
                ),
            }
        )

        return Dataset(ds_path)

    @pytest.fixture
    def ingested_dataset(self, tmp_path: pathlib.Path, monkeypatch: Any) -> Dataset:
        print("tmp_path for ingested dataset", tmp_path)
        prepared_dataset = self.prepared_dataset(tmp_path)
        logger.info("prepared_dataset: %s", prepared_dataset.path)
        mock_args = [
            "rslearn",
            "dataset",
            "ingest",
            "--root",
            str(prepared_dataset.path),
            "--disabled-layers",
            "sentinel2",
        ]
        monkeypatch.setattr(sys, "argv", mock_args)

        rslearn.main.main()
        logger.info("ingested_dataset: %s", prepared_dataset.path)
        return prepared_dataset

    def expected_materialized_fname(self, ds_path: UPath) -> UPath:
        return (
            ds_path
            / "windows"
            / "default"
            / "default"
            / "layers"
            / "local_files"
            / "data.geojson"
        )

    def test_materialization(
        self,
        ingested_dataset: Dataset,
        monkeypatch: Any,
    ) -> None:
        logger.info("ingested_dataset: %s", ingested_dataset.path)
        mock_args = [
            "rslearn",
            "dataset",
            "materialize",
            "--root",
            str(ingested_dataset.path),
        ]
        monkeypatch.setattr(sys, "argv", mock_args)
        rslearn.main.main()
        logger.info("materialized_dataset: %s", ingested_dataset.path)
        assert self.expected_materialized_fname(ingested_dataset.path).exists()

    def test_materialization_errors_on_bad_geojson(
        self, ingested_dataset: Dataset, monkeypatch: Any, tmp_path: pathlib.Path
    ) -> None:
        # Copy the ingested dataset to a new location.
        new_path = UPath(tmp_path) / "new_dataset"
        logger.info("copying dataset to %s", new_path)
        shutil.copytree(ingested_dataset.path, new_path)

        # Modify the dataset configuration to use a bad data source.
        logger.info("modify dataset configuration")
        # Corrupt the data.geojson file by writing invalid JSON
        data_file = new_path / "tiles" / "local_files" / "foo" / "data.geojson"
        with data_file.open("w") as f:
            f.write("invalid json{")

        # Try to materialize the dataset.
        mock_args = [
            "rslearn",
            "dataset",
            "materialize",
            "--root",
            str(new_path),
        ]
        monkeypatch.setattr(sys, "argv", mock_args)
        with pytest.raises(json.decoder.JSONDecodeError):
            rslearn.main.main()

    def test_ignore_errors(
        self,
        ingested_dataset: Dataset,
        tmp_path: pathlib.Path,
        monkeypatch: Any,
    ) -> None:
        print("test_ignore_errors tmp_path", tmp_path)
        # Copy the ingested dataset to a new location.
        new_path = UPath(tmp_path) / "new_dataset"
        logger.info("copying dataset to %s", new_path)
        shutil.copytree(ingested_dataset.path, new_path)
        # Corrupt the data.geojson file by writing invalid JSON
        data_file = new_path / "tiles" / "local_files" / "bar" / "data.geojson"
        with data_file.open("w") as f:
            f.write("invalid json{")

        # Try to materialize the dataset.
        mock_args = [
            "rslearn",
            "dataset",
            "materialize",
            "--root",
            str(new_path),
            "--ignore-errors",
            "--workers",
            "0",
        ]
        monkeypatch.setattr(sys, "argv", mock_args)
        rslearn.main.main()
        assert self.expected_materialized_fname(new_path).exists()
