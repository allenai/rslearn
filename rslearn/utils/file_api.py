import os
from typing import BinaryIO


class FileAPI:
    """A generic API for reading and writing binary data associated with filenames."""

    def open(self, fname: str, mode: str) -> BinaryIO:
        """Open a file for reading or writing.

        Args:
            fname: filename to open
            mode: must be rb or wb

        Returns:
            file-like object
        """
        raise NotImplementedError

    def exists(self, fname: str) -> bool:
        """Returns whether the filename exists or not."""
        raise NotImplementedError


class LocalFileAPI:
    """A FileAPI implementation that uses a local directory."""

    def __init__(self, root_dir):
        self.root_dir = root_dir

    def open(self, fname: str, mode: str) -> BinaryIO:
        """Open a file for reading or writing.

        Args:
            fname: filename to open
            mode: must be rb or wb

        Returns:
            file-like object
        """
        return open(os.path.join(self.root_dir, fname), mode)

    def exists(self, fname: str) -> bool:
        """Returns whether the filename exists or not."""
        return os.path.exists(os.path.join(self.root_dir, fname))
