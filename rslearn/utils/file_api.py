"""Implementations of a simple file access interface."""

import functools
import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from io import BytesIO, StringIO
from typing import Any, BinaryIO, TextIO

import boto3
import botocore
from class_registry import ClassRegistry

FileAPIs = ClassRegistry()


class FileAPI:
    """A generic API for reading and writing binary data associated with filenames."""

    def open(self, fname: str, mode: str) -> Generator[BinaryIO | TextIO, None, None]:
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
    ) -> Generator[BinaryIO | TextIO, None, None]:
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

    def to_str(self) -> str:
        """Encode this FileAPI to a string."""
        raise NotImplementedError


@FileAPIs.register("file")
class LocalFileAPI(FileAPI):
    """A FileAPI implementation that uses a local directory."""

    def __init__(self, root_dir: str):
        """Initialize a new LocalFileAPI.

        Args:
            root_dir: the directory to store the files
        """
        self.root_dir = root_dir

    def open(self, fname: str, mode: str) -> Generator[BinaryIO | TextIO, None, None]:
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
    ) -> Generator[BinaryIO | TextIO, None, None]:
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

    @staticmethod
    def from_str(s: str) -> FileAPI:
        """Parse the FileAPI string.

        Args:
            s: the string specifying this FileAPI.

        Returns:
            a FileAPI instance.
        """
        protocol, remainder = s.split("://", 1)
        assert protocol == "file"
        return LocalFileAPI(remainder)

    def to_str(self) -> str:
        """Encode this FileAPI to a string."""
        return f"file://{self.root_dir}"


class CallbackIO:
    """Provides enter/exit for accessing an IO with a callback upon exit.

    This enables providing file-like semantics when the underlying system only supports
    reading/writing the entire file.
    """

    def __init__(
        self,
        callback: Callable[[bytes], None] | None = None,
        buf: BinaryIO | TextIO = BytesIO(),
    ):
        """Create a new CallbackIO.

        Args:
            callback: the callback to pass the buffer value to after the IO is closed.
            buf: the buffer, can by BytesIO or TextIO, defaults to an empty BytesIO().
        """
        self.callback = callback
        self.buf = buf

    def __enter__(self) -> BinaryIO | TextIO:
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


BUCKET_CACHE = {}
"""Cache for boto3.Bucket objects."""


@FileAPIs.register("s3")
class S3FileAPI(FileAPI):
    """A FileAPI for S3-compatible object storage."""

    def __init__(
        self,
        endpoint_url: str,
        bucket_name: str,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        prefix: str = "",
    ):
        """Initialize a new S3FileAPI.

        Args:
            endpoint_url: the endpoint URL of the S3-compatible bucket.
            access_key_id: access key ID.
            secret_access_key: secret access key.
            bucket_name: the bucket name.
            prefix: optional prefix within the bucket to root this S3FileAPI.
        """
        self.endpoint_url = endpoint_url
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.bucket_name = bucket_name
        self.prefix = prefix

        if self.access_key_id is None:
            self.access_key_id = os.environ["S3_ACCESS_KEY_ID"]
        if self.secret_access_key is None:
            self.secret_access_key = os.environ["S3_SECRET_ACCESS_KEY"]

        self.bucket = S3FileAPI.get_bucket(
            self.endpoint_url,
            self.bucket_name,
            self.access_key_id,
            self.secret_access_key,
        )

    @staticmethod
    @functools.cache
    def get_bucket(
        endpoint_url: str, bucket_name: str, access_key_id: str, secret_access_key: str
    ) -> Any:
        """Get bucket with the specified parameters.

        The bucket is cached so we will reuse it for other prefixes in the same bucket.

        Args:
            endpoint_url: the endpoint URL of the S3-compatible bucket.
            access_key_id: access key ID.
            secret_access_key: secret access key.
            bucket_name: the bucket name.
        """
        s3 = boto3.resource(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )
        return s3.Bucket(bucket_name)

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
        return S3FileAPI(
            prefix=self._add_trailing_slash(self.prefix + self.join(*args)),
            endpoint_url=self.endpoint_url,
            access_key_id=self.access_key_id,
            secret_access_key=self.secret_access_key,
            bucket_name=self.bucket_name,
        )

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
        response = paginator.paginate(
            Bucket=self.bucket_name,
            Prefix=self._add_trailing_slash(self.prefix + self.join(*args)),
            Delimiter="/",
        )
        prefixes = []
        for result in response:
            for el in result.get("CommonPrefixes", []):
                prefixes.append(el["Prefix"].split("/")[-2])
        return prefixes

    @staticmethod
    def from_str(s: str) -> FileAPI:
        """Parse the FileAPI string.

        Args:
            s: the string specifying this FileAPI.

        Returns:
            a FileAPI instance.
        """
        protocol, remainder = s.split("://", 1)
        assert protocol == "s3"
        bucket_name, remainder = remainder.split("/", 1)
        parts = remainder.split("?", 1)
        if len(parts) == 1:
            prefix = parts[0]
            query_args = ""
        else:
            prefix = parts[0]
            query_args = parts[1]
        kwargs = {}
        for part in query_args.split("&"):
            if not part:
                continue
            k, v = part.split("=", 1)
            kwargs[k] = v
        return S3FileAPI(
            bucket_name=bucket_name,
            prefix=prefix,
            **kwargs,
        )

    def to_str(self) -> str:
        """Encode this FileAPI to a string."""
        return (
            f"s3://{self.bucket_name}/{self.prefix}"
            + f"?endpoint_url={self.endpoint_url}"
            + f"&access_key_id={self.access_key_id}"
            + f"&secret_access_key={self.secret_access_key}"
        )

    def __reduce__(self) -> tuple[Callable[[], "S3FileAPI"], tuple[str]]:
        """Serialization function."""
        return (S3FileAPI.from_str, (self.to_str(),))


def parse_file_api_string(s: str) -> FileAPI:
    """Parse a FileAPI string.

    Specify a local filesystem:
    file:///path/to/file/
    (It can also just be "/path/to/file/" and default assumes LocalFileAPI.)

    Specify an S3 bucket:
    s3://bucket-name/path/prefix/?endpoint_url=https://...&...

    Args:
        s: string specifying the FileAPI.

    Returns:
        a FileAPI instance.
    """
    if "://" in s:
        protocol, _ = s.split("://", 1)
        cls = FileAPIs.get_class(protocol)
        return cls.from_str(s)
    else:
        return LocalFileAPI(s)
