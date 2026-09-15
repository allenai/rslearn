"""Data source for ESA WorldCover 2021."""

import functools
import hashlib
import json
import os
import shutil
import tempfile
import time
import zipfile

import requests
from fsspec.implementations.local import LocalFileSystem
from upath import UPath

from rslearn.config import LayerType
from rslearn.data_sources.local_files import LocalFiles, RasterItemSpec
from rslearn.log_utils import get_logger
from rslearn.utils.fsspec import get_upath_local, join_upath

from .data_source import DataSourceContext, Item

logger = get_logger(__name__)


class WorldCereal(LocalFiles):
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
        "WorldCereal_2021_tc-annual_temporarycrops_classification.zip",
        "WorldCereal_2021_tc-maize-main_irrigation_confidence.zip",
        "WorldCereal_2021_tc-maize-main_irrigation_classification.zip",
        "WorldCereal_2021_tc-maize-main_maize_confidence.zip",
        "WorldCereal_2021_tc-maize-main_maize_classification.zip",
        "WorldCereal_2021_tc-maize-second_irrigation_confidence.zip",
        "WorldCereal_2021_tc-maize-second_irrigation_classification.zip",
        "WorldCereal_2021_tc-maize-second_maize_confidence.zip",
        "WorldCereal_2021_tc-maize-second_maize_classification.zip",
        "WorldCereal_2021_tc-springcereals_springcereals_confidence.zip",
        "WorldCereal_2021_tc-springcereals_springcereals_classification.zip",
        "WorldCereal_2021_tc-wintercereals_irrigation_confidence.zip",
        "WorldCereal_2021_tc-wintercereals_irrigation_classification.zip",
        "WorldCereal_2021_tc-wintercereals_wintercereals_confidence.zip",
        "WorldCereal_2021_tc-wintercereals_wintercereals_classification.zip",
    ]
    # Timeouts for downloading files from Zenodo. CONNECT_TIMEOUT_SECONDS bounds how
    # long we wait to establish the connection; READ_TIMEOUT_SECONDS bounds how long
    # we wait between chunks once the download is underway. These files are
    # multi-gigabyte, and Zenodo can stall well past 10 seconds between chunks, so
    # the read timeout needs to be generous.
    CONNECT_TIMEOUT_SECONDS = 10
    READ_TIMEOUT_SECONDS = 60

    # Number of bytes to read from the response stream at a time.
    DOWNLOAD_CHUNK_SIZE = 1024 * 1024

    # Maximum size of a single ranged HTTP request. Zenodo has been observed to
    # reject/reset multi-gigabyte single-range requests, and to return 504 for an
    # unranged full-file GET on the largest archives, even though small ranges
    # succeed reliably. So we always fetch in bounded chunks -- even for a brand
    # new download -- rather than only using Range requests once resuming.
    RANGE_SIZE_BYTES = 128 * 1024 * 1024

    # Number of consecutive chunk failures to tolerate (with backoff) before
    # giving up entirely. Since each chunk is capped at RANGE_SIZE_BYTES, a
    # failure only costs progress on the current chunk, not the whole file. This
    # is the *only* retry budget for a chunk request: requests.Session does not
    # retry failed requests by default, so a persistent failure can't multiply
    # into (adapter retries) x (this loop's retries) requests.
    MAX_DOWNLOAD_ATTEMPTS = 5
    DOWNLOAD_RETRY_BACKOFF_SECONDS = 5.0

    # Exceptions that indicate a transient network problem worth retrying (as
    # opposed to e.g. a 404 from a bad URL, which is not retried).
    RETRIABLE_DOWNLOAD_EXCEPTIONS = (
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.ChunkedEncodingError,
    )

    # HTTP statuses worth retrying (mirrors the status_forcelist that
    # create_retry_session would otherwise use, since we disable its automatic
    # retries here in favor of a single retry budget).
    RETRIABLE_STATUS_CODES = frozenset(
        {
            requests.codes.too_many_requests,
            requests.codes.internal_server_error,
            requests.codes.bad_gateway,
            requests.codes.service_unavailable,
            requests.codes.gateway_timeout,
        }
    )

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
            "id": "21551c80-0df9-4add-abaa-b66fff68179c",
            "filename": "WorldCereal_2021_tc-annual_temporarycrops_classification.zip",
            "filesize": 15500797967.0,
            "checksum": "c006c34fca0253251a8d1ea73cf837a8",
            "links": {
                "self": "https://zenodo.org/api/deposit/depositions/7875105/files/21551c80-0df9-4add-abaa-b66fff68179c",
                "download": "https://zenodo.org/api/records/7875105/files/WorldCereal_2021_tc-annual_temporarycrops_classification.zip/content",
            },
        },
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
            "id": "2cab95a8-24d9-45cf-ac70-67fa4b6bda64",
            "filename": "WorldCereal_2021_tc-maize-main_irrigation_classification.zip",
            "filesize": 17247922829.0,
            "checksum": "ceaf240dc4bba5e19491dd3c9893ae34",
            "links": {
                "self": "https://zenodo.org/api/deposit/depositions/7875105/files/2cab95a8-24d9-45cf-ac70-67fa4b6bda64",
                "download": "https://zenodo.org/api/records/7875105/files/WorldCereal_2021_tc-maize-main_irrigation_classification.zip/content",
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
            "id": "b2278b6c-c2f5-49c1-8ebc-e828dbf8c27d",
            "filename": "WorldCereal_2021_tc-maize-main_maize_classification.zip",
            "filesize": 18210475632.0,
            "checksum": "ff298db1b654b91fcfa27495d878932d",
            "links": {
                "self": "https://zenodo.org/api/deposit/depositions/7875105/files/b2278b6c-c2f5-49c1-8ebc-e828dbf8c27d",
                "download": "https://zenodo.org/api/records/7875105/files/WorldCereal_2021_tc-maize-main_maize_classification.zip/content",
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
            "id": "d9c5dbe4-d027-47aa-bb6e-806c9964f73e",
            "filename": "WorldCereal_2021_tc-maize-second_irrigation_classification.zip",
            "filesize": 6703649764.0,
            "checksum": "7221b40181835c5226d357ae3fec434f",
            "links": {
                "self": "https://zenodo.org/api/deposit/depositions/7875105/files/d9c5dbe4-d027-47aa-bb6e-806c9964f73e",
                "download": "https://zenodo.org/api/records/7875105/files/WorldCereal_2021_tc-maize-second_irrigation_classification.zip/content",
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
            "id": "93ae9f7f-f989-4fc5-837a-d27652b761f7",
            "filename": "WorldCereal_2021_tc-maize-second_maize_classification.zip",
            "filesize": 6917008439.0,
            "checksum": "aa883b52451f878e6b4462d27410707e",
            "links": {
                "self": "https://zenodo.org/api/deposit/depositions/7875105/files/93ae9f7f-f989-4fc5-837a-d27652b761f7",
                "download": "https://zenodo.org/api/records/7875105/files/WorldCereal_2021_tc-maize-second_maize_classification.zip/content",
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
            "id": "7a257437-89fe-4278-94fe-90a66e81e1bd",
            "filename": "WorldCereal_2021_tc-springcereals_springcereals_classification.zip",
            "filesize": 7008931281.0,
            "checksum": "bb6e1124938e3a68b6e47d156f17bf86",
            "links": {
                "self": "https://zenodo.org/api/deposit/depositions/7875105/files/7a257437-89fe-4278-94fe-90a66e81e1bd",
                "download": "https://zenodo.org/api/records/7875105/files/WorldCereal_2021_tc-springcereals_springcereals_classification.zip/content",
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
            "id": "a5774a05-ee8e-42df-bf06-68ebc6c14426",
            "filename": "WorldCereal_2021_tc-wintercereals_activecropland_classification.zip",
            "filesize": 20001277863.0,
            "checksum": "3933653452a2e0b821c35091b6f4a035",
            "links": {
                "self": "https://zenodo.org/api/deposit/depositions/7875105/files/a5774a05-ee8e-42df-bf06-68ebc6c14426",
                "download": "https://zenodo.org/api/records/7875105/files/WorldCereal_2021_tc-wintercereals_activecropland_classification.zip/content",
            },
        },
        {
            "id": "5a4adaa6-e50a-469a-b401-6ccca02de443",
            "filename": "WorldCereal_2021_tc-wintercereals_irrigation_classification.zip",
            "filesize": 18019534510.0,
            "checksum": "5032b11cf380d8cef07767e86ef4ee54",
            "links": {
                "self": "https://zenodo.org/api/deposit/depositions/7875105/files/5a4adaa6-e50a-469a-b401-6ccca02de443",
                "download": "https://zenodo.org/api/records/7875105/files/WorldCereal_2021_tc-wintercereals_irrigation_classification.zip/content",
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
            "id": "9ab67c40-9072-44dc-8f6b-892fcaa3c079",
            "filename": "WorldCereal_2021_tc-wintercereals_wintercereals_classification.zip",
            "filesize": 18523882137.0,
            "checksum": "386ce3fca8ba5577e2b62d6f3ea45b27",
            "links": {
                "self": "https://zenodo.org/api/deposit/depositions/7875105/files/9ab67c40-9072-44dc-8f6b-892fcaa3c079",
                "download": "https://zenodo.org/api/records/7875105/files/WorldCereal_2021_tc-wintercereals_wintercereals_classification.zip/content",
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
        worldcereal_dir: str,
        band: str | None = None,
        context: DataSourceContext = DataSourceContext(),
    ) -> None:
        """Create a new WorldCereal.

        Args:
            worldcereal_dir: the directory to extract the WorldCereal GeoTIFF files. For
                high performance, this should be a local directory; if the dataset is
                remote, prefix with a protocol ("file://") to use a local directory
                instead of a path relative to the dataset path.
            band: the worldcereal band to process. This will only be used if the layer
                config is missing from the context.
            context: the data source context.
        """
        if context.ds_path is not None:
            worldcereal_upath = join_upath(context.ds_path, worldcereal_dir)
        else:
            worldcereal_upath = UPath(worldcereal_dir)

        if context.layer_config is not None:
            if len(context.layer_config.band_sets) != 1:
                raise ValueError("expected a single band set")
            if len(context.layer_config.band_sets[0].bands) != 1:
                raise ValueError("expected band set to have a single band")
            self.band = context.layer_config.band_sets[0].bands[0]
        elif band is not None:
            self.band = band
        else:
            raise ValueError("band must be set if layer config is not in the context")

        tif_dir, tif_filepath = self.download_worldcereal_data(
            self.band, worldcereal_upath
        )
        all_aezs: set[int] = self.all_aezs_from_tifs(tif_filepath)

        # now that we have all our aezs, lets match them to the bands
        item_specs: list[RasterItemSpec] = []
        for aez in all_aezs:
            item_spec = RasterItemSpec(
                fnames=[],
                bands=[],
                # must be a str since we / with a posix path later
                name=str(aez),
            )
            aez_band_filepath = self.filepath_for_product_aez(tif_filepath, aez)
            if aez_band_filepath is not None:
                item_spec.fnames.append(aez_band_filepath.absolute().as_uri())
                assert item_spec.bands is not None
                item_spec.bands.append([self.band])
            item_specs.append(item_spec)
        if len(item_specs) == 0:
            raise ValueError(f"No AEZ files found for {self.band}")

        super().__init__(
            src_dir=tif_dir.absolute().as_uri(),
            raster_item_specs=item_specs,
            layer_type=LayerType.RASTER,
            context=context,
        )

    @staticmethod
    def band_from_zipfilename(filename: str) -> str:
        """Return the band name given the zipfilename."""
        # [:-4] to remove ".zip"
        _, _, season, product, confidence_or_classification = filename[:-4].split("_")
        # band names must not contain '_'
        return "-".join([season, product, confidence_or_classification])

    @staticmethod
    def zip_filepath_from_filename(filename: str) -> str:
        """Given a filename, return the filepath of the extracted tifs."""
        _, _, season, product, confidence_or_classification = filename[:-4].split("_")
        prefix = "data/worldcereal_data/MAP-v3/2021"
        if confidence_or_classification == "confidence":
            aez_name = "aez_downsampled"
        else:
            aez_name = "aez"
        # [:-4] to remove ".zip"

        return f"{prefix}/{season}/{product}/{aez_name}/{confidence_or_classification}"

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

    @staticmethod
    def _compute_md5(local_path: str) -> str:
        """Compute the MD5 checksum of a local file, reading it in chunks."""
        hasher = hashlib.md5()
        with open(local_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def _download_with_resume(
        cls,
        file_url: str,
        dest_path: UPath,
        expected_size: int,
        expected_checksum: str,
    ) -> None:
        """Download a file in bounded chunks, resuming and verifying checksums.

        Args:
            file_url: the URL to download from.
            dest_path: the destination path for the completed download.
            expected_size: the expected size of the file in bytes.
            expected_checksum: the expected MD5 checksum of the file.
        """
        if isinstance(dest_path.fs, LocalFileSystem):
            cls._download_to_local_path_with_resume(
                file_url, dest_path.path, expected_size, expected_checksum
            )
            return

        # Most object stores don't support appending to an object, and a write
        # that's interrupted partway can leave a partial/corrupt object sitting
        # at the final key with no atomicity -- which download_worldcereal_data's
        # `if not zip_filepath.exists()` check would then treat as a complete
        # download forever. So for remote destinations, stage the download (with
        # the same bounded-chunk retrying and checksum verification used for
        # local paths) to a local temp file first, and only copy it to the
        # remote destination once it's verified complete and correct.
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = os.path.join(tmp_dir, "download")
            cls._download_to_local_path_with_resume(
                file_url, local_path, expected_size, expected_checksum
            )
            with open(local_path, "rb") as src, dest_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)

    @classmethod
    def _download_to_local_path_with_resume(
        cls,
        file_url: str,
        local_path: str,
        expected_size: int,
        expected_checksum: str,
    ) -> None:
        """Download a file to a local path in bounded, retried, verified chunks.

        The file is fetched as a sequence of Range requests of at most
        RANGE_SIZE_BYTES each -- even on the very first chunk of a brand new
        download -- since Zenodo has been observed to reject or reset
        multi-gigabyte single-range transfers, and to return 504 for an unranged
        full-file GET on the largest archives. Downloaded bytes are appended to a
        stably-named temporary file (rather than a per-process name) so that if
        the download is interrupted -- by a timeout, a dropped connection, or the
        process being killed -- a later attempt resumes from the exact byte
        offset already on disk instead of starting over. Once the full file has
        been fetched, its size and MD5 checksum are verified against the values
        Zenodo reports before it is atomically renamed into place; a mismatch is
        treated as corruption, and the corrupt data is deleted.

        Retries (for both transient network exceptions and retriable HTTP status
        codes) are all funneled through a single MAX_DOWNLOAD_ATTEMPTS budget:
        the underlying session has its own automatic retries disabled, so a
        persistent failure can't multiply into two nested retry loops.

        Args:
            file_url: the URL to download from.
            local_path: the local destination path for the completed download.
            expected_size: the expected size of the file in bytes.
            expected_checksum: the expected MD5 checksum of the file.
        """
        session = requests.Session()
        tmp_path = local_path + ".partial"
        consecutive_failures = 0

        while True:
            current_size = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
            if current_size >= expected_size:
                break

            range_end = min(current_size + cls.RANGE_SIZE_BYTES, expected_size) - 1
            headers = {"Range": f"bytes={current_size}-{range_end}"}
            failure_reason: Exception | str | None = None

            try:
                with session.get(
                    file_url,
                    stream=True,
                    timeout=(cls.CONNECT_TIMEOUT_SECONDS, cls.READ_TIMEOUT_SECONDS),
                    headers=headers,
                ) as r:
                    if r.status_code == requests.codes.partial_content:
                        with open(tmp_path, "ab") as f:
                            f.writelines(
                                r.iter_content(chunk_size=cls.DOWNLOAD_CHUNK_SIZE)
                            )
                    elif r.status_code in cls.RETRIABLE_STATUS_CODES:
                        failure_reason = f"status {r.status_code}"
                    else:
                        # The server returned something other than a partial
                        # response or a known-transient status. Appending a
                        # non-partial body (which would start from byte 0) to
                        # our partially-downloaded file would corrupt it, so
                        # bail out entirely rather than retrying the same
                        # request.
                        raise ValueError(
                            f"server did not return a partial response for a "
                            f"range request to {file_url} (status "
                            f"{r.status_code}); cannot safely download in "
                            "bounded chunks"
                        )
            except cls.RETRIABLE_DOWNLOAD_EXCEPTIONS as e:
                failure_reason = e

            if failure_reason is not None:
                consecutive_failures += 1
                if consecutive_failures >= cls.MAX_DOWNLOAD_ATTEMPTS:
                    raise ValueError(
                        f"download of {file_url} failed after "
                        f"{cls.MAX_DOWNLOAD_ATTEMPTS} consecutive attempts "
                        f"(last failure: {failure_reason})"
                    )
                logger.warning(
                    "chunk download of %s failed (%s), %d/%d consecutive "
                    "failures, retrying from byte %d",
                    file_url,
                    failure_reason,
                    consecutive_failures,
                    cls.MAX_DOWNLOAD_ATTEMPTS,
                    os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0,
                )
                time.sleep(cls.DOWNLOAD_RETRY_BACKOFF_SECONDS * consecutive_failures)
                continue

            consecutive_failures = 0

        actual_size = os.path.getsize(tmp_path)
        if actual_size != expected_size:
            raise ValueError(
                f"downloaded size {actual_size} for {file_url} does not match "
                f"expected size {expected_size}"
            )

        actual_checksum = cls._compute_md5(tmp_path)
        if actual_checksum != expected_checksum:
            os.remove(tmp_path)
            raise ValueError(
                f"checksum mismatch for {file_url} (got {actual_checksum}, "
                f"expected {expected_checksum}); deleted corrupt download"
            )

        os.rename(tmp_path, local_path)

    @classmethod
    def download_worldcereal_data(
        cls, band: str, worldcereal_dir: UPath
    ) -> tuple[UPath, dict[str, UPath]]:
        """Download and extract the WorldCereal data.

        If the data was previously downloaded, this function returns quickly.

        Args:
            band: the worldcereal band to download.
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
        logger.debug(f"Worldcereal zipfile: {zip_dir}")

        # Fetch list of files from Zenodo's Deposition Files API
        # f["filename"] maps to the ZIP_FILENAMES
        files_to_download = [
            f
            for f in cls.ZENODO_FILES_DATA
            if cls.band_from_zipfilename(f["filename"]) == band
        ]
        if len(files_to_download) != 1:
            raise ValueError(
                f"Got != 1 suitable filenames for {band}: {[f['filename'] for f in files_to_download]}"
            )
        file_to_download = files_to_download[0]
        # now its also in the right order for when we generate the files
        filename: str = file_to_download["filename"]
        if filename not in cls.ZIP_FILENAMES:
            raise ValueError(f"Unsupported filename {filename} for band {band}")
        file_url = file_to_download["links"]["download"]
        # Determine full filepath and create necessary folders for nested structure
        zip_filepath = zip_dir / filename
        if not zip_filepath.exists():
            logger.debug(f"Downloading {file_url} to {zip_filepath}")
            cls._download_with_resume(
                file_url=file_url,
                dest_path=zip_filepath,
                expected_size=int(file_to_download["filesize"]),
                expected_checksum=file_to_download["checksum"],
            )

        # Extract the zip files.
        # We use a .extraction_complete file to indicate that the extraction is done.
        tif_dir = worldcereal_dir / "tifs"
        tif_dir.mkdir(parents=True, exist_ok=True)

        completed_fname = zip_dir / (filename + ".extraction_complete")
        if completed_fname.exists():
            logger.debug("%s has already been extracted", filename)
        else:
            logger.debug("extracting %s to %s", filename, tif_dir)

            # If the tif_dir is remote, we need to extract to a temporary local
            # directory first and then copy it over.
            if isinstance(tif_dir.fs, LocalFileSystem):
                local_dir = tif_dir.path
            else:
                tmp_dir = tempfile.TemporaryDirectory()
                local_dir = tmp_dir.name

            with get_upath_local(zip_filepath) as local_fname:
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
        tif_filepath = tif_dir / cls.zip_filepath_from_filename(filename)

        return tif_dir, tif_filepath

    @functools.cache
    def list_items(self) -> list[Item]:
        """Lists items from the source directory while maintaining a cache file.

        This is identical to LocalFiles.list_items except that a unique summary
        is made per band (since we treat each band separately now.)
        """
        cache_fname = self.src_dir / f"{self.band}_summary.json"
        if not cache_fname.exists():
            logger.debug("cache at %s does not exist, listing items", cache_fname)
            items = self.importer.list_items(self.src_dir)
            serialized_items = [item.serialize() for item in items]
            with cache_fname.open("w") as f:
                json.dump(serialized_items, f)
            return items

        logger.debug("loading item list from cache at %s", cache_fname)
        with cache_fname.open() as f:
            serialized_items = json.load(f)
        return [
            self.deserialize_item(serialized_item)
            for serialized_item in serialized_items
        ]
