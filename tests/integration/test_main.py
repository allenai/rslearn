import json
import pathlib
import sys
from datetime import datetime, timezone
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
from rslearn.utils.geometry import STGeometry


class TestIngestion:
    @pytest.fixture
    def prepared_dataset(self, tmp_path: pathlib.Path) -> Dataset:
        # Create a dataset that is ready for ingestion.
        # It has two layers:
        # - local_files: LocalFiles dataset where ingestion should succeed.
        # - sentinel2: Sentinel2 dataset where items.json has unknown scene name so
        #   ingestion should fail.

        # First make a vector file for LocalFiles to ingest from.
        print("make vector file")
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
        print("make dataset")
        ds_path = UPath(tmp_path) / "dataset"
        ds_path.mkdir(parents=True)
        ds_config = {
            "layers": {
                "local_files": {
                    "type": "vector",
                    "data_source": {
                        "name": "rslearn.data_sources.local_files.LocalFiles",
                        "src_dir": str(src_fname.parent),
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
                        "name": "rslearn.data_sources.gcp_public_data.Sentinel2",
                        "modality": "L1C",
                        "index_cache_dir": "cache",
                        "use_rtree_index": False,
                    },
                },
            },
            "tile_store": {
                "name": "file",
                "root_dir": "tiles",
            },
        }
        with (ds_path / "config.json").open("w") as f:
            json.dump(ds_config, f)

        # Add window intersecting the vector data file.
        print("make window")
        window = Window(
            path=Window.get_window_root(ds_path, "default", "default"),
            group="default",
            name="default",
            projection=WGS84_PROJECTION,
            bounds=(-1, -1, 1, 1),
            time_range=(
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 2, 1, tzinfo=timezone.utc),
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
        return (
            prepared_dataset.path
            / "tiles"
            / "local_files"
            / "foo"
            / str(WGS84_PROJECTION)
            / "data.geojson"
        )

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
