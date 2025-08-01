"""Data source for ESA WorldCover 2021."""

import os
import shutil
import tempfile
import zipfile
from typing import Any

import requests
from fsspec.implementations.local import LocalFileSystem
from upath import UPath

from rslearn.config import DataSourceConfig, LayerConfig, QueryConfig
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
        "WorldCereal_2021_tc-wintercereals_wintercereals_confidence.zip",
    ]
    TIMEOUT_SECONDS = 10

    # this can be obtained using the following code:
    # ```
    # response = requests.get(cls.ZENODO_URL)
    # response.raise_for_status()
    # ZENODO_FILES_DATA = response.json()
    # ```
    # we hardcode it here because othewerwise we get complaints from
    # zenodo about repeatedly asking for it.
    ZENODO_FILES_DATA: list[dict] = [
        {
            "id": "2fed6859-5729-4ab1-9d33-e15464c99a5b",
            "filename": "WorldCereal_2021_tc-annual_temporarycrops_confidence.zip",
            "filesize": 24969180828.0,
            "checksum": "84a953be71292d02cceb6c64b2008ad7",
            "links": {
                "self": "https://zenodo.org/api/deposit/depositions/7875105/files/2fed6859-5729-4ab1-9d33-e15464c99a5b",
                "download": "https://zenodo.org/api/records/7875105/files/WorldCereal_2021_tc-annual_temporarycrops_confidence.zip/content",
            },
        },
        {
            "id": "54d63601-cda8-4f10-8710-a2068e697418",
            "filename": "WorldCereal_2021_tc-maize-main_irrigation_confidence.zip",
            "filesize": 11327157543.0,
            "checksum": "c509ee2cb8b6fc44383788ffaa248950",
            "links": {
                "self": "https://zenodo.org/api/deposit/depositions/7875105/files/54d63601-cda8-4f10-8710-a2068e697418",
                "download": "https://zenodo.org/api/records/7875105/files/WorldCereal_2021_tc-maize-main_irrigation_confidence.zip/content",
            },
        },
        {
            "id": "277c0d06-b5ae-4748-bad1-c135084276ef",
            "filename": "WorldCereal_2021_tc-maize-main_maize_confidence.zip",
            "filesize": 10442831518.0,
            "checksum": "0e6bb70209a83b526ec146e5e4ed3451",
            "links": {
                "self": "https://zenodo.org/api/deposit/depositions/7875105/files/277c0d06-b5ae-4748-bad1-c135084276ef",
                "download": "https://zenodo.org/api/records/7875105/files/WorldCereal_2021_tc-maize-main_maize_confidence.zip/content",
            },
        },
        {
            "id": "f47baf24-27d9-4913-a483-ec86ae87e60a",
            "filename": "WorldCereal_2021_tc-maize-second_irrigation_confidence.zip",
            "filesize": 3813149175.0,
            "checksum": "cb8b91155c8fcf38f869875f2cb35200",
            "links": {
                "self": "https://zenodo.org/api/deposit/depositions/7875105/files/f47baf24-27d9-4913-a483-ec86ae87e60a",
                "download": "https://zenodo.org/api/records/7875105/files/WorldCereal_2021_tc-maize-second_irrigation_confidence.zip/content",
            },
        },
        {
            "id": "d3a0df02-8034-463f-a923-2bfe0c2719ac",
            "filename": "WorldCereal_2021_tc-maize-second_maize_confidence.zip",
            "filesize": 3752378387.0,
            "checksum": "8a819762b7f3950839b0e832cb346e30",
            "links": {
                "self": "https://zenodo.org/api/deposit/depositions/7875105/files/d3a0df02-8034-463f-a923-2bfe0c2719ac",
                "download": "https://zenodo.org/api/records/7875105/files/WorldCereal_2021_tc-maize-second_maize_confidence.zip/content",
            },
        },
        {
            "id": "a0b91677-f110-4df5-a5fd-7b1849895a02",
            "filename": "WorldCereal_2021_tc-springcereals_springcereals_confidence.zip",
            "filesize": 4708773375.0,
            "checksum": "fd8dec8de691738df520c1ab451c7870",
            "links": {
                "self": "https://zenodo.org/api/deposit/depositions/7875105/files/a0b91677-f110-4df5-a5fd-7b1849895a02",
                "download": "https://zenodo.org/api/records/7875105/files/WorldCereal_2021_tc-springcereals_springcereals_confidence.zip/content",
            },
        },
        {
            "id": "23301576-64d2-48a1-9b19-0c126158c24d",
            "filename": "WorldCereal_2021_tc-wintercereals_irrigation_confidence.zip",
            "filesize": 11447731232.0,
            "checksum": "f84c4088ac42bb67f308be50159ca778",
            "links": {
                "self": "https://zenodo.org/api/deposit/depositions/7875105/files/23301576-64d2-48a1-9b19-0c126158c24d",
                "download": "https://zenodo.org/api/records/7875105/files/WorldCereal_2021_tc-wintercereals_irrigation_confidence.zip/content",
            },
        },
        {
            "id": "b4ce9cc1-a745-450a-b2e9-c4fb08059a93",
            "filename": "WorldCereal_2021_tc-wintercereals_wintercereals_confidence.zip",
            "filesize": 10174751452.0,
            "checksum": "5870da83aaa4b3761cad3750feb73e43",
            "links": {
                "self": "https://zenodo.org/api/deposit/depositions/7875105/files/b4ce9cc1-a745-450a-b2e9-c4fb08059a93",
                "download": "https://zenodo.org/api/records/7875105/files/WorldCereal_2021_tc-wintercereals_wintercereals_confidence.zip/content",
            },
        },
    ]

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
        tif_dir, tif_filepaths = self.download_worldcereal_data(worldcereal_dir)
        all_aezs: set[int] = set()
        for _, tif_path in tif_filepaths.items():
            # firstly, go through all the files to collect the AEZs
            # then, go through the AEZs and get the files
            # then, fill the missing ones with 0s
            all_aezs.update(self.all_aezs_from_tifs(tif_path))
        # now that we have all our aezs, lets match them to the bands
        spec_dicts: list[dict] = []
        for aez in all_aezs:
            spec_dict: dict[str, Any] = {
                # must be a str since we / with a posix path later
                "name": str(aez),
                "fnames": [],
                "bands": [],
            }
            for band, tif_path in tif_filepaths.items():
                aez_band_filepath = self.filepath_for_product_aez(tif_path, aez)
                if aez_band_filepath is not None:
                    spec_dict["fnames"].append(aez_band_filepath.absolute().as_uri())
                    spec_dict["bands"].append([band])
            spec_dicts.append(spec_dict)
        # add this to the config
        if config.data_source is not None:
            if "item_specs" in config.data_source.config_dict:
                logger.warning(
                    "Overwriting item_specs in WorldCereal config.data_source"
                )
            config.data_source.config_dict["item_specs"] = spec_dicts
        else:
            config.data_source = DataSourceConfig(
                name="rslearn.data_sources.WorldCerealConfidence",
                query_config=QueryConfig.from_config({}),
                config_dict={"item_specs": spec_dicts},
            )

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
    def band_from_zipfilename(filename: str) -> str:
        """Return the band name given the zipfilename."""
        # [:-4] to remove ".zip"
        _, _, season, product, confidence = filename[:-4].split("_")
        # band names must not contain '_'
        return "-".join([season, product, confidence])

    @staticmethod
    def zip_filepath_from_filename(filename: str) -> str:
        """Given a filename, return the filepath of the extracted tifs."""
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

    @staticmethod
    def filepath_for_product_aez(path_to_tifs: UPath, aez: int) -> UPath | None:
        """Given a path for the tifs for a band and an aez, return the tif file if it exists."""
        aez_file = list(path_to_tifs.glob(f"{aez}_*.tif"))
        if len(aez_file) == 0:
            return None
        elif len(aez_file) == 1:
            return aez_file[0]
        raise ValueError(f"Got more than one tif for {aez} in {path_to_tifs}")

    @classmethod
    def download_worldcereal_data(
        cls, worldcereal_dir: UPath
    ) -> tuple[UPath, dict[str, UPath]]:
        """Download and extract the WorldCereal data.

        If the data was previously downloaded, this function returns quickly.

        Args:
            worldcereal_dir: the directory to download to.

        Returns:
            tif_dir: the sub-directory containing GeoTIFFs
            tif_filepaths: tif dir is nested (i.e. tif_dir points to "data" while the tifs
                are actually in "data/worldcereal/MAP-v3/2021..."). This points to the
                specific directories containing the tifs for each band.
        """
        # Download the zip files (if they don't already exist).
        zip_dir = worldcereal_dir / "zips"
        zip_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Worldcereal zipfile: {zip_dir}")

        # Fetch list of files from Zenodo's Deposition Files API
        # f["filename"] maps to the ZIP_FILENAMES
        files_as_dict = {f["filename"]: f for f in cls.ZENODO_FILES_DATA}
        # now its also in the right order for when we generate the files
        ordered_files = [files_as_dict[z_f] for z_f in cls.ZIP_FILENAMES]
        for file_info in ordered_files:
            filename: str = file_info["filename"]
            if filename not in cls.ZIP_FILENAMES:
                logger.info(f"Skipping {filename}, which is not a confidence layer")
                continue
            file_url = file_info["links"]["download"]
            # Determine full filepath and create necessary folders for nested structure
            filepath = zip_dir / filename
            if filepath.exists():
                continue
            # Download the file with resume support
            logger.info(f"Downloading {file_url} to {filepath}")
            with requests.get(file_url, stream=True, timeout=cls.TIMEOUT_SECONDS) as r:
                r.raise_for_status()
                with open_atomic(filepath, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

        # Extract the zip files.
        # We use a .extraction_complete file to indicate that the extraction is done.
        tif_dir = worldcereal_dir / "tifs"
        tif_dir.mkdir(parents=True, exist_ok=True)
        for file_info in ordered_files:
            filename = file_info["filename"]
            zip_fname = zip_dir / filename

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
        tif_filepaths = {
            cls.band_from_zipfilename(file_info["filename"]): tif_dir
            / cls.zip_filepath_from_filename(file_info["filename"])
            for file_info in ordered_files
        }

        return tif_dir, tif_filepaths
