"""Helper functions for raster data sources."""

from rslearn.config import BandSetConfig
from rslearn.log_utils import get_logger

logger = get_logger(__name__)


def is_raster_needed(raster_bands: list[str], band_sets: list[BandSetConfig]) -> bool:
    """Check if the raster by comparing its bands to the configured bands.

    Args:
        raster_bands: the list of bands in the raster in question.
        band_sets: the band sets configured in the dataset.

    Returns:
        whether the raster is needed for the dataset.
    """
    for band_set in band_sets:
        for band in band_set.bands:
            if band in raster_bands:
                return True
    return False
