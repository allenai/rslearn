"""Implementations of a simple file access interface."""

import os
from collections.abc import Generator
from contextlib import contextmanager
from io import BytesIO, StringIO
from typing import Any, BinaryIO, Callable, Optional, TextIO, Union

import boto3
import botocore


class FileAPI:
    """A generic API for reading and writing binary data associated with filenames."""

    def open(
        self, fname: str, mode: str
    ) -> Generator[Union[BinaryIO, TextIO], None, None]:
        """Open a file for reading or writing.

        Args:
            fname: filename to open
            mode: must be r, w, rb, or wb

        Returns:
            file-like object
        """
        raise NotImplementedError

    def open_atomic(
        self, fname: str, mode: str
    ) -> Generator[Union[BinaryIO, TextIO], None, None]:
        """Open a file for atomic writing.

        Guarantees that overlapping calls to open_atomic will lead to one or the other
        being written.

        Args:
            fname: filename to open
            mode: either w or wb

        Returns:
            file-like object
        """
        raise NotImplementedError

    def exists(self, fname: str) -> bool:
        """Returns whether the filename exists or not."""
        raise NotImplementedError

    def get_folder(self, *args) -> "FileAPI":
        """Get FileAPI for a sub-path of this FileAPI.

        Args:
            args: the path elements

        Returns:
            Another FileAPI rooted at the path.
        """
        raise NotImplementedError

    def join(self, *args) -> str:
        """Join the specified path elements."""
        raise NotImplementedError

    def listdir(self, *args) -> list[str]:
        """List the path elements sharing the specified prefix.

        Args:
            args: the path elements

        Returns:
            next path elements
        """
        raise NotImplementedError


class LocalFileAPI:
    """A FileAPI implementation that uses a local directory."""

    def __init__(self, root_dir: str):
        """Initialize a new LocalFileAPI.

        Args:
            root_dir: the directory to store the files
        """
        self.root_dir = root_dir

    def open(
        self, fname: str, mode: str
    ) -> Generator[Union[BinaryIO, TextIO], None, None]:
        """Open a file for reading or writing.

        Args:
            fname: filename to open
            mode: must be r, w, rb, or wb

        Returns:
            file-like object
        """
        return open(os.path.join(self.root_dir, fname), mode)

    @contextmanager
    def open_atomic(
        self, fname: str, mode: str
    ) -> Generator[Union[BinaryIO, TextIO], None, None]:
        """Open a file for atomic writing.

        Will write to a temporary file, and rename it to the destination upon success.

        Args:
            fname: the file path to be opened
            mode: either w or wb

        Returns:
            file-like object
        """
        tmpname = fname + ".tmp." + str(os.getpid())
        with open(os.path.join(self.root_dir, tmpname), mode) as file:
            yield file
        os.rename(
            os.path.join(self.root_dir, tmpname), os.path.join(self.root_dir, fname)
        )

    def exists(self, *args) -> bool:
        """Returns whether the filename exists or not.

        Args:
            args: the path elements

        Returns:
            whether the file exists
        """
        return os.path.exists(os.path.join(self.root_dir, *args))

    def get_folder(self, *args) -> "FileAPI":
        """Get FileAPI for a sub-path of this FileAPI.

        Args:
            args: the path elements

        Returns:
            Another FileAPI rooted at the path.
        """
        root_dir = os.path.join(self.root_dir, self.join(*args))
        os.makedirs(root_dir, exist_ok=True)
        return LocalFileAPI(root_dir)

    def join(self, *args) -> str:
        """Join the specified path elements."""
        return os.path.join(*args)

    def listdir(self, *args) -> list[str]:
        """List the path elements sharing the specified prefix.

        Args:
            args: the path elements

        Returns:
            next path elements
        """
        return os.listdir(os.path.join(self.root_dir, *args))


class CallbackIO:
    """Provides enter/exit for accessing an IO with a callback upon exit.

    This enables providing file-like semantics when the underlying system only supports
    reading/writing the entire file.
    """

    def __init__(
        self,
        callback: Optional[Callable[[bytes], None]] = None,
        buf: Union[BinaryIO, TextIO] = BytesIO(),
    ):
        """Create a new CallbackIO.

        Args:
            callback: the callback to pass the buffer value to after the IO is closed.
            buf: the buffer, can by BytesIO or TextIO, defaults to an empty BytesIO().
        """
        self.callback = callback
        self.buf = buf

    def __enter__(self) -> Union[BinaryIO, TextIO]:
        """Enter the CallbackIO.

        Returns:
            the buffer.
        """
        return self.buf

    def __exit__(self, type, value, traceback):
        """Exit the CallbackIO.

        Runs the callback function with the buffer state.

        Args:
            type: ignored
            value: ignored
            traceback: ignored
        """
        if self.callback:
            self.callback(self.buf.getvalue())


class S3FileAPI:
    """A FileAPI for S3-compatible object storage."""

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        bucket_name: Optional[str] = None,
        prefix: str = "",
        bucket: Optional[Any] = None,
    ):
        """Initialize a new S3FileAPI.

        Args:
            endpoint_url: the endpoint URL of the S3-compatible bucket.
            access_key_id: access key ID.
            secret_access_key: secret access key.
            bucket_name: the bucket name.
            prefix: optional prefix within the bucket to root this S3FileAPI.
            bucket: the boto3 Bucket object.
        """
        self.prefix = prefix
        self.bucket_name = bucket_name

        if bucket:
            self.bucket = bucket
        else:
            s3 = boto3.resource(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
            )
            self.bucket = s3.Bucket(bucket_name)

    def open(self, fname: str, mode: str) -> Generator[BinaryIO, None, None]:
        """Open a file for reading or writing.

        Args:
            fname: filename to open
            mode: must be rb or wb

        Returns:
            file-like object
        """
        if mode in ["rb", "r"]:
            obj = self.bucket.Object(self.prefix + fname)
            buf = BytesIO()
            obj.download_fileobj(buf)
            if mode == "rb":
                return CallbackIO(buf=buf)
            else:
                return CallbackIO(buf=StringIO(buf.getvalue().decode()))

        elif mode in ["wb", "w"]:

            def callback(value):
                if mode == "w":
                    value = value.encode()
                self.bucket.put_object(
                    Key=self.prefix + fname,
                    Body=value,
                )

            if mode == "wb":
                buf = BytesIO()
            else:
                buf = StringIO()
            return CallbackIO(callback=callback, buf=buf)

        raise Exception(f"bad mode {mode}")

    def open_atomic(self, fname: str, mode: str) -> Generator[BinaryIO, None, None]:
        """Open a file for atomic writing.

        Will write to a temporary file, and rename it to the destination upon success.

        Args:
            fname: the file path to be opened
            mode: either w or wb

        Returns:
            file-like object
        """
        return self.open(fname, mode)

    def exists(self, *args) -> bool:
        """Returns whether the filename exists or not.

        Args:
            args: the path elements

        Returns:
            whether the file exists
        """
        try:
            self.bucket.Object(self.prefix + self.join(*args)).load()
            return True
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def get_folder(self, *args) -> "FileAPI":
        """Get FileAPI for a sub-path of this FileAPI.

        Args:
            args: the path elements

        Returns:
            Another FileAPI rooted at the path.
        """
        return S3FileAPI(prefix=self._add_trailing_slash(self.prefix + self.join(*args)), bucket=self.bucket)

    def join(self, *args) -> str:
        """Join the specified path elements."""
        if args:
            return "/".join(args)
        else:
            return ""

    def _add_trailing_slash(self, path: str) -> str:
        if path and path[-1] != "/":
            return path + "/"
        else:
            return path

    def listdir(self, *args) -> list[str]:
        """List the path elements sharing the specified prefix.

        Args:
            args: the path elements

        Returns:
            next path elements
        """
        paginator = self.bucket.meta.client.get_paginator("list_objects")
        response = paginator.paginate(Bucket=self.bucket_name, Prefix=self._add_trailing_slash(self.prefix + self.join(*args)), Delimiter="/")
        prefixes = []
        for result in response:
            for el in result.get("CommonPrefixes", []):
                prefixes.append(el["Prefix"].split("/")[-2])
        return prefixes
