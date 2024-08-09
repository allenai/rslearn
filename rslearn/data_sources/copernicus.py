"""Data source for raster data in ESA Copernicus API."""

import xml.etree.ElementTree as ET
from collections.abc import Callable

import numpy as np
import numpy.typing as npt


def get_harmonize_callback(
    tree: ET.ElementTree,
) -> Callable[[npt.NDArray], npt.NDArray] | None:
    """Gets the harmonization callback based on the metadata XML.

    Harmonization ensures that scenes before and after processing baseline 04.00
    are comparable. 04.00 introduces +1000 offset to the pixel values to include
    more information about dark areas.

    Args:
        tree: the parsed XML tree

    Returns:
        None if no callback is needed, or the callback to subtract the new offset
    """
    offset = None
    for el in tree.iter("RADIO_ADD_OFFSET"):
        value = int(el.text)
        if offset is None:
            offset = value
            assert offset <= 0
            # For now assert the offset is always -1000.
            assert offset == -1000
        else:
            assert offset == value

    if offset is None or offset == 0:
        return None

    def callback(array):
        return np.clip(array, -offset, None) + offset

    return callback
