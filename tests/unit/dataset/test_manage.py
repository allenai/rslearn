"""Unit tests for dataset management functions."""

import json
import pathlib

from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.dataset.manage import prepare_dataset_windows


class TestPrepareDatasetWindows:
    """Test suite for prepare_dataset_windows function."""

    def test_min_matches_skips_windows_with_insufficient_items(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Test that windows are skipped when min_matches is not met."""
        ds_path = UPath(tmp_path)

        # Create GeoJSON files - LocalFiles creates one item per file
        # Window 1 area: 2 files (will not meet min_matches=3)
        # Window 2 area: 4 files (will meet min_matches=3)
        src_data_dir = tmp_path / "src_data"
        src_data_dir.mkdir()

        # Create 2 files in area that window1 will intersect
        for i in range(2):
            features = [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Point",
                        "coordinates": [2 + i, 2 + i],
                    },
                },
            ]
            with (src_data_dir / f"file_{i}.geojson").open("w") as f:
                json.dump(
                    {
                        "type": "FeatureCollection",
                        "features": features,
                    },
                    f,
                )

        # Create 4 files in area that window2 will intersect
        for i in range(4):
            features = [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Point",
                        "coordinates": [12 + i, 12 + i],
                    },
                },
            ]
            with (src_data_dir / f"file2_{i}.geojson").open("w") as f:
                json.dump(
                    {
                        "type": "FeatureCollection",
                        "features": features,
                    },
                    f,
                )

        # Create dataset config with min_matches=3
        dataset_config = {
            "layers": {
                "local_file": {
                    "type": "vector",
                    "data_source": {
                        "name": "rslearn.data_sources.local_files.LocalFiles",
                        "src_dir": str(src_data_dir),
                        "query_config": {
                            "space_mode": "INTERSECTS",
                            "min_matches": 3,
                            "max_matches": 10,
                        },
                    },
                },
            },
        }
        with (ds_path / "config.json").open("w") as f:
            json.dump(dataset_config, f)

        # Create two windows:
        # Window 1: Intersects 2 files (will be skipped due to min_matches=3)
        # Window 2: Intersects 4 files (will be prepared)
        Window(
            path=Window.get_window_root(ds_path, "default", "window1"),
            group="default",
            name="window1",
            projection=WGS84_PROJECTION,
            bounds=(0, 0, 5, 5),  # Intersects file_0 and file_1
            time_range=None,
        ).save()

        Window(
            path=Window.get_window_root(ds_path, "default", "window2"),
            group="default",
            name="window2",
            projection=WGS84_PROJECTION,
            bounds=(10, 10, 20, 20),  # Intersects file2_0, file2_1, file2_2, file2_3
            time_range=None,
        ).save()

        # Run prepare_dataset_windows
        dataset = Dataset(ds_path)
        windows = dataset.load_windows()
        summary = prepare_dataset_windows(dataset, windows)

        # Verify the summary
        assert len(summary.layer_summaries) == 1
        layer_summary = summary.layer_summaries[0]
        assert layer_summary.layer_name == "local_file"
        assert layer_summary.windows_prepared == 1  # Only window2
        assert layer_summary.windows_skipped == 1  # window1 skipped due to min_matches
        assert summary.total_windows_requested == 2

        # Verify window1 has empty item groups (was skipped)
        window1 = next(w for w in windows if w.name == "window1")
        layer_datas = window1.load_layer_datas()
        assert "local_file" in layer_datas
        assert len(layer_datas["local_file"].serialized_item_groups) == 0

        # Verify window2 has item groups (was prepared)
        window2 = next(w for w in windows if w.name == "window2")
        layer_datas = window2.load_layer_datas()
        assert "local_file" in layer_datas
        assert (
            len(layer_datas["local_file"].serialized_item_groups) >= 3
        )  # At least 3 groups

    def test_min_matches_zero_does_not_skip_windows(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Test that windows are not skipped when min_matches=0."""
        ds_path = UPath(tmp_path)

        # Create a GeoJSON with 1 feature
        src_data_dir = tmp_path / "src_data"
        src_data_dir.mkdir()

        features = [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Point",
                    "coordinates": [5, 5],
                },
            },
        ]
        with (src_data_dir / "data.geojson").open("w") as f:
            json.dump(
                {
                    "type": "FeatureCollection",
                    "features": features,
                },
                f,
            )

        # Create dataset config with min_matches=0 (default)
        dataset_config = {
            "layers": {
                "local_file": {
                    "type": "vector",
                    "data_source": {
                        "name": "rslearn.data_sources.local_files.LocalFiles",
                        "src_dir": str(src_data_dir),
                        "query_config": {
                            "space_mode": "INTERSECTS",
                            "min_matches": 0,
                            "max_matches": 10,
                        },
                    },
                },
            },
        }
        with (ds_path / "config.json").open("w") as f:
            json.dump(dataset_config, f)

        # Create a window that intersects the feature
        Window(
            path=Window.get_window_root(ds_path, "default", "default"),
            group="default",
            name="default",
            projection=WGS84_PROJECTION,
            bounds=(0, 0, 10, 10),
            time_range=None,
        ).save()

        # Run prepare_dataset_windows
        dataset = Dataset(ds_path)
        windows = dataset.load_windows()
        summary = prepare_dataset_windows(dataset, windows)

        # Verify the summary - window should be prepared, not skipped
        assert len(summary.layer_summaries) == 1
        layer_summary = summary.layer_summaries[0]
        assert layer_summary.windows_prepared == 1
        assert layer_summary.windows_skipped == 0  # No skipping when min_matches=0
        assert summary.total_windows_requested == 1

    def test_min_matches_with_mixed_prepared_and_skipped_windows(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Test min_matches behavior with already-prepared windows and new windows."""
        ds_path = UPath(tmp_path)

        # Create GeoJSON files - LocalFiles creates one item per file
        # Window 2 area: 1 file (will not meet min_matches=2)
        # Window 3 area: 2 files (will meet min_matches=2)
        src_data_dir = tmp_path / "src_data"
        src_data_dir.mkdir()

        # Create 1 file in area that window2 will intersect
        features = [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Point",
                    "coordinates": [5, 5],
                },
            },
        ]
        with (src_data_dir / "data_1.geojson").open("w") as f:
            json.dump(
                {
                    "type": "FeatureCollection",
                    "features": features,
                },
                f,
            )

        # Create 2 files in area that window3 will intersect
        for i in range(2):
            features = [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Point",
                        "coordinates": [15 + i, 15 + i],
                    },
                },
            ]
            with (src_data_dir / f"data_2_{i}.geojson").open("w") as f:
                json.dump(
                    {
                        "type": "FeatureCollection",
                        "features": features,
                    },
                    f,
                )

        # Create dataset config with min_matches=2
        dataset_config = {
            "layers": {
                "local_file": {
                    "type": "vector",
                    "data_source": {
                        "name": "rslearn.data_sources.local_files.LocalFiles",
                        "src_dir": str(src_data_dir),
                        "query_config": {
                            "space_mode": "INTERSECTS",
                            "min_matches": 2,
                            "max_matches": 10,
                        },
                    },
                },
            },
        }
        with (ds_path / "config.json").open("w") as f:
            json.dump(dataset_config, f)

        # Create three windows:
        # Window 1: Already prepared (will be skipped, not counted in min_matches skip)
        # Window 2: Intersects 1 file (will be skipped due to min_matches)
        # Window 3: Intersects 2 files (will be prepared)
        window1 = Window(
            path=Window.get_window_root(ds_path, "default", "window1"),
            group="default",
            name="window1",
            projection=WGS84_PROJECTION,
            bounds=(0, 0, 1, 1),  # Doesn't intersect any files
            time_range=None,
        )
        window1.save()
        # Manually mark window1 as prepared by creating empty layer data
        from rslearn.dataset import WindowLayerData

        layer_datas = window1.load_layer_datas()
        layer_datas["local_file"] = WindowLayerData(
            layer_name="local_file",
            serialized_item_groups=[[]],
        )
        window1.save_layer_datas(layer_datas)

        Window(
            path=Window.get_window_root(ds_path, "default", "window2"),
            group="default",
            name="window2",
            projection=WGS84_PROJECTION,
            bounds=(0, 0, 10, 10),  # Intersects data_1.geojson (1 file)
            time_range=None,
        ).save()

        Window(
            path=Window.get_window_root(ds_path, "default", "window3"),
            group="default",
            name="window3",
            projection=WGS84_PROJECTION,
            bounds=(10, 10, 20, 20),  # Intersects data_2_0 and data_2_1 (2 files)
            time_range=None,
        ).save()

        # Run prepare_dataset_windows
        dataset = Dataset(ds_path)
        windows = dataset.load_windows()
        summary = prepare_dataset_windows(dataset, windows)

        # Verify the summary
        assert len(summary.layer_summaries) == 1
        layer_summary = summary.layer_summaries[0]
        assert layer_summary.windows_prepared == 1  # Only window3
        # window1 skipped (already prepared) + window2 skipped (min_matches)
        assert layer_summary.windows_skipped == 2
        assert summary.total_windows_requested == 3
