"""Classes to remap raster values."""

from typing import Any

import numpy as np
import numpy.typing as npt
from class_registry import ClassRegistry

Remappers = ClassRegistry()


class Remapper:
    def __call__(
        self, array: npt.NDArray[Any], dtype: npt.DTypeLike
    ) -> npt.NDArray[Any]:
        """Compute remapped array.

        Args:
            array: the array to remap
            dtype: dtype of output array

        Returns:
            remapped array
        """
        raise NotImplementedError


@Remappers.register("linear")
class LinearRemapper(Remapper):
    def __init__(self, config: dict[str, Any]):
        self.src_min = config["src"][0]
        self.src_max = config["src"][1]
        self.dst_min = config["dst"][0]
        self.dst_max = config["dst"][1]

    def __call__(
        self, array: npt.NDArray[Any], dtype: npt.DTypeLike
    ) -> npt.NDArray[Any]:
        """Compute remapped array.

        Args:
            array: the array to remap
            dtype: dtype of output array

        Returns:
            remapped array
        """
        array = (array.astype(np.float64) - self.src_min) * (
            self.dst_max - self.dst_min
        ) / (self.src_max - self.src_min) + self.dst_min
        array = np.clip(array, self.dst_min, self.dst_max).astype(dtype)
        return array


def load_remapper(config: dict[str, Any]) -> Remapper:
    """Load a remapper from a configuration dictionary."""
    return Remappers.get(config["name"], config=config)
