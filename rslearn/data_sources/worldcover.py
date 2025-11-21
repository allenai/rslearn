"""Data source for ESA WorldCover 2021."""

import os
import shutil
import tempfile
import zipfile

import requests
from fsspec.implementations.local import LocalFileSystem
from upath import UPath

from rslearn.config import LayerType
from rslearn.data_sources import DataSourceContext
from rslearn.data_sources.local_files import LocalFiles
from rslearn.log_utils import get_logger
from rslearn.utils.fsspec import get_upath_local, join_upath, open_atomic

logger = get_logger(__name__)


class WorldCover(LocalFiles):
    """A data source for the ESA WorldCover 2021 land cover map.

    For details about the land cover map, see https://worldcover2021.esa.int/.

    This data source downloads the 18 zip files that contain the map. They are then
    extracted, yielding 2,651 GeoTIFF files. These are then used with
    rslearn.data_sources.local_files.LocalFiles to implement the data source.
    """

    BASE_URL = "https://worldcover2021.esa.int/data/archive/"
    ZIP_FILENAMES = [
        "ESA_WorldCover_10m_2021_v200_60deg_macrotile_N30E000.zip",
        "ESA_WorldCover_10m_2021_v200_60deg_macrotile_N30E060.zip",
        "ESA_WorldCover_10m_2021_v200_60deg_macrotile_N30E120.zip",
        "ESA_WorldCover_10m_2021_v200_60deg_macrotile_N30W060.zip",
        "ESA_WorldCover_10m_2021_v200_60deg_macrotile_N30W120.zip",
        "ESA_WorldCover_10m_2021_v200_60deg_macrotile_N30W180.zip",
        "ESA_WorldCover_10m_2021_v200_60deg_macrotile_S30E000.zip",
        "ESA_WorldCover_10m_2021_v200_60deg_macrotile_S30E060.zip",
        "ESA_WorldCover_10m_2021_v200_60deg_macrotile_S30E120.zip",
        "ESA_WorldCover_10m_2021_v200_60deg_macrotile_S30W060.zip",
        "ESA_WorldCover_10m_2021_v200_60deg_macrotile_S30W120.zip",
        "ESA_WorldCover_10m_2021_v200_60deg_macrotile_S30W180.zip",
        "ESA_WorldCover_10m_2021_v200_60deg_macrotile_S90E000.zip",
        "ESA_WorldCover_10m_2021_v200_60deg_macrotile_S90E060.zip",
        "ESA_WorldCover_10m_2021_v200_60deg_macrotile_S90E120.zip",
        "ESA_WorldCover_10m_2021_v200_60deg_macrotile_S90W060.zip",
        "ESA_WorldCover_10m_2021_v200_60deg_macrotile_S90W120.zip",
        "ESA_WorldCover_10m_2021_v200_60deg_macrotile_S90W180.zip",
    ]
    TIMEOUT_SECONDS = 10

    def __init__(
        self,
        worldcover_dir: str,
        context: DataSourceContext = DataSourceContext(),
    ) -> None:
        """Create a new WorldCover.

        Args:
            config: configuration for this layer. It should specify a single band
                called B1 which will contain the land cover class.
            worldcover_dir: the directory to extract the WorldCover GeoTIFF files. For
                high performance, this should be a local directory; if the dataset is
                remote, prefix with a protocol ("file://") to use a local directory
                instead of a path relative to the dataset path.
            context: the data source context.
        """
        if context.ds_path is not None:
            worldcover_upath = join_upath(context.ds_path, worldcover_dir)
        else:
            worldcover_upath = UPath(worldcover_dir)

        tif_dir = self.download_worldcover_data(worldcover_upath)

        super().__init__(
            src_dir=tif_dir,
            layer_type=LayerType.RASTER,
            context=context,
        )

    def download_worldcover_data(self, worldcover_dir: UPath) -> UPath:
        """Download and extract the WorldCover data.

        If the data was previously downloaded, this function returns quickly.

        Args:
            worldcover_dir: the directory to download to.

        Returns:
            the sub-directory containing GeoTIFFs
        """
        # Download the zip files (if they don't already exist).
        zip_dir = worldcover_dir / "zips"
        zip_dir.mkdir(parents=True, exist_ok=True)
        for fname in self.ZIP_FILENAMES:
            src_url = self.BASE_URL + fname
            dst_fname = zip_dir / fname
            if dst_fname.exists():
                logger.debug("%s has already been downloaded at %s", fname, dst_fname)
                continue
            logger.info("downloading %s to %s", src_url, dst_fname)
            with requests.get(src_url, stream=True, timeout=self.TIMEOUT_SECONDS) as r:
                r.raise_for_status()
                with open_atomic(dst_fname, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

        # Extract the zip files.
        # We use a .extraction_complete file to indicate that the extraction is done.
        tif_dir = worldcover_dir / "tifs"
        tif_dir.mkdir(parents=True, exist_ok=True)
        for fname in self.ZIP_FILENAMES:
            zip_fname = zip_dir / fname
            completed_fname = zip_dir / (fname + ".extraction_complete")
            if completed_fname.exists():
                logger.debug("%s has already been extracted", fname)
                continue
            logger.info("extracting %s to %s", fname, tif_dir)

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

        return tif_dir
