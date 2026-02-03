"""Data from worldpop.org."""

import random
from datetime import timedelta
from html.parser import HTMLParser
from urllib.parse import urljoin

import requests
from upath import UPath

from rslearn.config import LayerType
from rslearn.data_sources import DataSourceContext
from rslearn.data_sources.local_files import LocalFiles
from rslearn.log_utils import get_logger
from rslearn.utils.fsspec import join_upath, open_atomic

logger = get_logger(__name__)


class LinkExtractor(HTMLParser):
    """Extract links from HTML.

    The links attribute will be filled with the href attribute of all links that appear
    on the HTML page.
    """

    def __init__(self) -> None:
        """Create a new LinkExtractor."""
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Handle start of tag from the HTML parsing."""
        if tag.lower() != "a":
            return
        for name, value in attrs:
            if name.lower() != "href":
                continue
            if value is None:
                continue
            self.links.append(value)


class WorldPop(LocalFiles):
    """World population data from worldpop.org.

    Currently, this only supports the WorldPop Constrained 2020 100 m Resolution
    dataset. See https://hub.worldpop.org/project/categories?id=3 for details.

    The data is split by country. We implement with LocalFiles data source for
    simplicity, but it means that all of the data must be downloaded first.
    """

    INDEX_URLS = [
        "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/",
        "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/maxar_v1/",
    ]
    FILENAME_SUFFIX = "_ppp_2020_constrained.tif"

    def __init__(
        self,
        worldpop_dir: str,
        timeout: timedelta = timedelta(seconds=30),
        context: DataSourceContext = DataSourceContext(),
    ):
        """Create a new WorldPop.

        Args:
            worldpop_dir: the directory to extract the WorldPop GeoTIFF files. For
                high performance, this should be a local directory; if the dataset is
                remote, prefix with a protocol ("file://") to use a local directory
                instead of a path relative to the dataset path.
            timeout: timeout for HTTP requests.
            context: the data source context.
        """
        if context.ds_path is not None:
            worldpop_upath = join_upath(context.ds_path, worldpop_dir)
        else:
            worldpop_upath = UPath(worldpop_dir)
        worldpop_upath.mkdir(parents=True, exist_ok=True)
        self.download_worldpop_data(worldpop_upath, timeout)
        super().__init__(
            src_dir=worldpop_upath.absolute().as_uri(),
            layer_type=LayerType.RASTER,
            context=context,
        )

    def download_worldpop_data(self, worldpop_dir: UPath, timeout: timedelta) -> None:
        """Download and extract the WorldPop data.

        If the data was previously downloaded, this function returns quickly.

        Args:
            worldpop_dir: the directory to download to.
            timeout: timeout for HTTP requests.
        """
        completed_fname = worldpop_dir / "completed"
        if completed_fname.exists():
            return

        # Scan the index URLs to get all the per-country subfolders.
        # These should be four characters with slash at the end, like "USA/".
        country_urls = []
        for index_url in self.INDEX_URLS:
            logger.info(f"Getting per-country subfolders from {index_url}")
            response = requests.get(index_url, timeout=timeout.total_seconds())
            response.raise_for_status()
            parser = LinkExtractor()
            parser.feed(response.text)
            country_urls.extend(
                [
                    urljoin(index_url, href)
                    for href in parser.links
                    if len(href) == 4 and href[3] == "/"
                ]
            )

        logger.info(f"Got {len(country_urls)} country subfolders to download")
        # Shuffling here enables the user to run multiple processes to speed up the
        # download.
        random.shuffle(country_urls)

        # Now iterate over the country-level URLs and download the GeoTIFF.
        for country_url in country_urls:
            response = requests.get(country_url, timeout=timeout.total_seconds())
            response.raise_for_status()
            parser = LinkExtractor()
            parser.feed(response.text)
            tif_links = [
                urljoin(country_url, href)
                for href in parser.links
                if href.endswith(self.FILENAME_SUFFIX)
            ]
            if len(tif_links) != 1:
                raise ValueError(
                    f"expected {country_url} to contain one GeoTIFF ending in {self.FILENAME_SUFFIX} but got {parser.links}"
                )

            country_fname = tif_links[0].split("/")[-1]
            dst_fname = worldpop_dir / country_fname
            if dst_fname.exists():
                continue

            logger.info(f"Downloading from {tif_links[0]} to {dst_fname}")
            with requests.get(
                tif_links[0], stream=True, timeout=timeout.total_seconds()
            ) as r:
                r.raise_for_status()
                with open_atomic(dst_fname, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

        completed_fname.touch()
