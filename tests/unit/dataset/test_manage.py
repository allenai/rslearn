"""Unit tests for dataset management functions."""

import json
import pathlib
from typing import Any

import pytest
from upath import UPath

from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.local_files import LocalFiles
from rslearn.dataset import Dataset, Window, WindowLayerData
from rslearn.dataset.handler_summaries import IngestCounts
from rslearn.dataset.manage import (
    materialize_dataset_windows,
    prepare_dataset_windows,
)
from rslearn.dataset.materialize import VectorMaterializer
from rslearn.main import IngestHandler


def _make_local_files_dataset(tmp_path: pathlib.Path) -> Dataset:
    """Helper: create a dataset with one LocalFiles vector layer and one window."""
    ds_path = UPath(tmp_path)
    src_data_dir = tmp_path / "src_data"
    src_data_dir.mkdir()

    with (src_data_dir / "data.geojson").open("w") as f:
        json.dump(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {"type": "Point", "coordinates": [5, 5]},
                    }
                ],
            },
            f,
        )

    dataset_config = {
        "layers": {
            "local_file": {
                "type": "vector",
                "data_source": {
                    "class_path": "rslearn.data_sources.local_files.LocalFiles",
                    "init_args": {"src_dir": str(src_data_dir)},
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

    dataset = Dataset(ds_path)
    Window(
        storage=dataset.storage,
        group="default",
        name="default",
        projection=WGS84_PROJECTION,
        bounds=(0, 0, 10, 10),
        time_range=None,
    ).save()

    return Dataset(ds_path)


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
                        "class_path": "rslearn.data_sources.local_files.LocalFiles",
                        "init_args": {
                            "src_dir": str(src_data_dir),
                        },
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

        dataset = Dataset(ds_path)
        storage = dataset.storage

        # Create two windows:
        # Window 1: Intersects 2 files (will be skipped due to min_matches=3)
        # Window 2: Intersects 4 files (will be prepared)
        Window(
            storage=storage,
            group="default",
            name="window1",
            projection=WGS84_PROJECTION,
            bounds=(0, 0, 5, 5),  # Intersects file_0 and file_1
            time_range=None,
        ).save()

        Window(
            storage=storage,
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
        layer_summary = summary.layer_summaries["local_file"]
        assert layer_summary.layer_name == "local_file"
        assert layer_summary.windows_prepared == 1  # Only window2
        assert layer_summary.windows_skipped == 0  # All windows were needed
        assert (
            layer_summary.windows_rejected == 1
        )  # window1 rejected due to min_matches
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
                        "class_path": "rslearn.data_sources.local_files.LocalFiles",
                        "init_args": {
                            "src_dir": str(src_data_dir),
                        },
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

        dataset = Dataset(ds_path)
        storage = dataset.storage

        # Create a window that intersects the feature
        Window(
            storage=storage,
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
        layer_summary = summary.layer_summaries["local_file"]
        assert layer_summary.windows_prepared == 1
        assert layer_summary.windows_skipped == 0
        assert layer_summary.windows_rejected == 0
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
                        "class_path": "rslearn.data_sources.local_files.LocalFiles",
                        "init_args": {
                            "src_dir": str(src_data_dir),
                        },
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

        dataset = Dataset(ds_path)
        storage = dataset.storage

        # Create three windows:
        # Window 1: Already prepared (will be skipped, not counted in min_matches skip)
        # Window 2: Intersects 1 file (will be skipped due to min_matches)
        # Window 3: Intersects 2 files (will be prepared)
        window1 = Window(
            storage=storage,
            group="default",
            name="window1",
            projection=WGS84_PROJECTION,
            bounds=(0, 0, 1, 1),  # Doesn't intersect any files
            time_range=None,
        )
        window1.save()
        # Manually mark window1 as prepared by creating empty layer data
        layer_datas = window1.load_layer_datas()
        layer_datas["local_file"] = WindowLayerData(
            layer_name="local_file",
            serialized_item_groups=[[]],
        )
        window1.save_layer_datas(layer_datas)

        Window(
            storage=storage,
            group="default",
            name="window2",
            projection=WGS84_PROJECTION,
            bounds=(0, 0, 10, 10),  # Intersects data_1.geojson (1 file)
            time_range=None,
        ).save()

        Window(
            storage=storage,
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
        layer_summary = summary.layer_summaries["local_file"]
        assert layer_summary.windows_prepared == 1  # Only window3
        assert (
            layer_summary.windows_skipped == 1
        )  # window1 not needed (already prepared)
        assert (
            layer_summary.windows_rejected == 1
        )  # window2 rejected due to min_matches
        assert summary.total_windows_requested == 3

    def test_previously_rejected_windows_counted_as_rejected_not_skipped(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Test that windows previously rejected due to min_matches are counted as rejected on second run."""
        ds_path = UPath(tmp_path)

        # Create GeoJSON file - LocalFiles creates one item per file
        # Window will intersect 1 file, but min_matches=2, so will be rejected
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

        # Create dataset config with min_matches=2
        dataset_config = {
            "layers": {
                "local_file": {
                    "type": "vector",
                    "data_source": {
                        "class_path": "rslearn.data_sources.local_files.LocalFiles",
                        "init_args": {
                            "src_dir": str(src_data_dir),
                        },
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

        dataset = Dataset(ds_path)
        storage = dataset.storage

        # Create a window that intersects 1 file (will not meet min_matches=2)
        Window(
            storage=storage,
            group="default",
            name="window1",
            projection=WGS84_PROJECTION,
            bounds=(0, 0, 10, 10),  # Intersects data.geojson (1 file)
            time_range=None,
        ).save()

        dataset = Dataset(ds_path)
        windows = dataset.load_windows()

        # First run: window should be rejected
        summary1 = prepare_dataset_windows(dataset, windows)
        assert len(summary1.layer_summaries) == 1
        layer_summary1 = summary1.layer_summaries["local_file"]
        assert layer_summary1.windows_prepared == 0
        assert layer_summary1.windows_skipped == 0
        assert layer_summary1.windows_rejected == 1  # Rejected due to min_matches

        # Verify window1 has empty item groups (was rejected)
        window1 = next(w for w in windows if w.name == "window1")
        layer_datas = window1.load_layer_datas()
        assert "local_file" in layer_datas
        assert len(layer_datas["local_file"].serialized_item_groups) == 0

        # Second run: window should still be counted as rejected, not skipped
        summary2 = prepare_dataset_windows(dataset, windows)
        assert len(summary2.layer_summaries) == 1
        layer_summary2 = summary2.layer_summaries["local_file"]
        assert layer_summary2.windows_prepared == 0
        assert layer_summary2.windows_skipped == 0
        assert layer_summary2.windows_rejected == 1  # Still rejected, not skipped

    def test_get_items_error_captured(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that errors from get_items are captured in the summary."""
        dataset = _make_local_files_dataset(tmp_path)
        windows = dataset.load_windows()

        def bad_get_items(self: Any, *a: Any, **kw: Any) -> Any:
            raise RuntimeError("boom")

        monkeypatch.setattr(LocalFiles, "get_items", bad_get_items)
        summary = prepare_dataset_windows(dataset, windows, ignore_errors=True)

        ls = summary.layer_summaries["local_file"]
        assert ls.windows_failed == 1
        assert ls.windows_prepared == 0
        assert any("boom" in msg for msg in ls.error_messages)


class TestIngestHandler:
    """Tests for IngestHandler."""

    def test_ingest_error_captured(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that ingest errors are captured in the summary."""
        dataset = _make_local_files_dataset(tmp_path)
        windows = dataset.load_windows()
        prepare_dataset_windows(dataset, windows)

        handler = IngestHandler(ignore_errors=True)
        handler.set_dataset(dataset)
        jobs = handler.get_jobs(windows, 0)
        assert len(jobs) > 0

        def bad_ingest(self: Any, *a: Any, **kw: Any) -> Any:
            raise RuntimeError("ingest boom")

        monkeypatch.setattr(LocalFiles, "ingest", bad_ingest)
        summary = handler(jobs)

        ls = summary.layer_summaries["local_file"]
        assert isinstance(ls.ingest_counts, IngestCounts)
        assert any("ingest boom" in msg for msg in ls.error_messages)


class TestMaterializeDatasetWindows:
    """Tests for materialize_dataset_windows."""

    def test_materialize_error_captured(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that materialize errors are captured in the summary."""
        dataset = _make_local_files_dataset(tmp_path)
        windows = dataset.load_windows()
        prepare_dataset_windows(dataset, windows)

        def bad_materialize(self: Any, *a: Any, **kw: Any) -> Any:
            raise RuntimeError("mat boom")

        monkeypatch.setattr(VectorMaterializer, "materialize", bad_materialize)
        summary = materialize_dataset_windows(dataset, windows, ignore_errors=True)

        ls = summary.layer_summaries["local_file"]
        assert ls.windows_failed == 1
        assert ls.num_windows_materialized == 0
        assert any("mat boom" in msg for msg in ls.error_messages)
