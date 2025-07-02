"""Data source for ESA WorldCover 2021."""

import os
import shutil
import tempfile
import zipfile

import requests
from fsspec.implementations.local import LocalFileSystem
from upath import UPath
import hashlib

from rslearn.config import LayerConfig
from rslearn.data_sources.local_files import LocalFiles
from rslearn.log_utils import get_logger
from rslearn.utils.fsspec import get_upath_local, join_upath, open_atomic

logger = get_logger(__name__)


class WorldCerealConfidences(LocalFiles):
    """A data source for the ESA WorldCereal 2021 agricultural land cover map.

    For details about the land cover map, see https://esa-worldcereal.org/en.
    """
    ZENODO_RECORD_ID = 7875105
    ZENODO_URL = f"https://zenodo.org/api/deposit/depositions/{ZENODO_RECORD_ID}/files"

    # these are the subset of filenames we want to download, which contain the
    # model confidence values. This defines the order of the bands in the
    # final output tif files
    ZIP_FILENAMES = [
        "WorldCereal_2021_tc-annual_temporarycrops_confidence.zip",
        "WorldCereal_2021_tc-maize-main_irrigation_confidence.zip",
        "WorldCereal_2021_tc-maize-main_maize_confidence.zip",
        "WorldCereal_2021_tc-maize-second_irrigation_confidence.zip",
        "WorldCereal_2021_tc-maize-second_maize_confidence.zip",
        "WorldCereal_2021_tc-springcereals_springcereals_confidence.zip",
        "WorldCereal_2021_tc-wintercereals_irrigation_confidence.zip",
    ]
    TIMEOUT_SECONDS = 10

    def __init__(
        self,
        config: LayerConfig,
        worldcereal_dir: UPath,
    ) -> None:
        """Create a new WorldCereal.

        Args:
            config: configuration for this layer. It should specify a single band
                called B1 which will contain the land cover class.
            worldcereal_dir: the directory to extract the WorldCereal GeoTIFF files. For
                high performance, this should be a local directory; if the dataset is
                remote, prefix with a protocol ("file://") to use a local directory
                instead of a path relative to the dataset path.
        """
        tif_dir = self.download_and_process_worldcereal_data(worldcereal_dir)
        super().__init__(config, tif_dir)

    @staticmethod
    def from_config(config: LayerConfig, ds_path: UPath) -> "LocalFiles":
        """Creates a new LocalFiles instance from a configuration dictionary."""
        if config.data_source is None:
            raise ValueError("LocalFiles data source requires a data source config")
        d = config.data_source.config_dict
        return WorldCerealConfidences(
            config=config, worldcereal_dir=join_upath(ds_path, d["worldcereal_dir"])
        )

    @staticmethod
    def zip_filepath_from_filename(filename: str) -> str:
        """Given a filename, return the filepath of the extracted tifs.
        """
        prefix = "data/worldcereal_data/MAP-v3/2021"
        aez_name = "aez_downsampled"
        # [:-4] to remove ".zip"
        _, _, season, product, confidence = filename[:-4].split("_")
        return f"{prefix}/{season}/{product}/{aez_name}/{confidence}"

    @staticmethod
    def all_aezs_from_tifs(filepath: UPath) -> set[int]:
        """Given a filepath containing many tif files, extract all the AEZs."""
        all_tifs = filepath.glob("*.tif")
        aezs: set = set()
        for tif_file in all_tifs:
            aezs.add(int(tif_file.name.split("_")[0]))
        return aezs

    @classmethod
    def download_and_process_worldcereal_data(cls, worldcereal_dir: UPath) -> UPath:
        """Download and extract the WorldCereal data.

        If the data was previously downloaded, this function returns quickly.

        Args:
            worldcereal_dir: the directory to download to.

        Returns:
            the sub-directory containing GeoTIFFs
        """
        # Download the zip files (if they don't already exist).
        zip_dir = worldcereal_dir / "zips"
        zip_dir.mkdir(parents=True, exist_ok=True)

        # Fetch list of files from Zenodo's Deposition Files API
        response = requests.get(cls.ZENODO_URL)
        response.raise_for_status()
        files_data = response.json()

        files_as_dict = {f["filename"]: f for f in files_data}
        # now its also in the right order for when we generate the files
        ordered_files = [files_as_dict[z_f] for z_f in cls.ZIP_FILENAMES]
        for file_info in ordered_files:
            filename: str = file_info['filename']
            if filename not in cls.ZIP_FILENAMES:
                logger.info(f"Skipping {filename}, which is not a confidence layer")
                continue
            if not filename.startswith("WorldCereal") and filename.endswith(".zip"):
                logger.info(f"Skipping download for {filename}")
                continue
            file_url = file_info['links']['download']
            # Determine full filepath and create necessary folders for nested structure
            filepath = zip_dir / filename
            if filepath.exists():
                continue
            # Download the file with resume support
            with requests.get(file_url, stream=True, timeout=cls.TIMEOUT_SECONDS) as r:
                r.raise_for_status()
                with open_atomic(filepath, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

        # Extract the zip files.
        # We use a .extraction_complete file to indicate that the extraction is done.
        tif_dir = worldcereal_dir / "tifs_intermediate"
        tif_dir.mkdir(parents=True, exist_ok=True)
        for file_info in ordered_files:
            filename: str = file_info['filename']
            zip_fname = zip_dir / filename

            # THIS IS JUST FOR TESTING. REMOVE BEFORE MERGING
            if not zip_fname.exists():
                print(f"Skipping {zip_fname}")
                continue

            completed_fname = zip_dir / (filename + ".extraction_complete")
            if completed_fname.exists():
                logger.debug("%s has already been extracted", filename)
                continue
            logger.info("extracting %s to %s", filename, tif_dir)

            # If the tif_dir is remote, we need to extract to a temporary local
            # directory first and then copy it over.
            if isinstance(tif_dir.fs, LocalFileSystem):
                local_dir = tif_dir.path
            else:
                tmp_dir = tempfile.TemporaryDirectory()
                local_dir = tmp_dir.name

            with get_upath_local(zip_fname) as local_fname:
                with zipfile.ZipFile(local_fname) as zip_f:
                    zip_f.extractall(local_dir)

            # Copy it over if the tif_dir was remote.
            if not isinstance(tif_dir.fs, LocalFileSystem):
                for fname in os.listdir(local_dir):
                    with open(os.path.join(local_dir, fname), "rb") as src:
                        with (tif_dir / fname).open("wb") as dst:
                            shutil.copyfileobj(src, dst)

            # Mark the extraction complete.
            completed_fname.touch()

        # finally, for each AEZ we will combine the bands so that they
        # are stored in a single tif file
        for file_info in ordered_files:
            filename = file_info["filename"]
            path_to_tifs = tif_dir / cls.zip_filepath_from_filename(filename)

        return tif_dir
