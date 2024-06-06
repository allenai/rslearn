"""Training tasks."""

from typing import Any, Union

import numpy.typing as npt
from class_registry import ClassRegistry

from rslearn.utils import Feature

Tasks = ClassRegistry()


class Task:
    """Represents an ML task like object detection or segmentation.

    A task specifies how raster or vector data should be processed into targets that
    can be passed to models. It also specifies evaluation functions for computing
    metrics comparing targets/outputs.
    """

    def get_target(self, data: Union[npt.NDArray[Any], list[Feature]]) -> Any:
        """Processes the data into targets.

        Args:
            data: raster or vector data to process

        Returns:
            the processed targets that are compatible with both metrics and loss
                functions
        """
        raise NotImplementedError
