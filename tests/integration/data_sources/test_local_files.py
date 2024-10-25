from rslearn.dataset import Dataset
from rslearn.dataset.manage import (
    ingest_dataset_windows,
    materialize_dataset_windows,
    prepare_dataset_windows,
)
from rslearn.utils.vector_format import load_vector_format


class TestLocalFiles:
    """Tests the LocalFiles data source.

    1. Create GeoJSON as a local file to extract data from.
    2. Create a corresponding dataset config file.
    3. Create a window intersecting the features.
    3. Run prepare, ingest, materialize, and make sure it gets the features.
    """

    def test_sample_dataset(self, local_files_dataset: Dataset) -> None:
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
