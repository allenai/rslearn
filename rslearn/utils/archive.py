"""Archive extraction helpers."""

import os
import pathlib
import shutil
import stat
import zipfile
from collections.abc import Iterable


def _validate_zip_member(
    zip_info: zipfile.ZipInfo, destination: pathlib.Path
) -> pathlib.Path:
    """Return the extraction path for a zip member, rejecting unsafe names."""
    member_name = zip_info.filename
    windows_path = pathlib.PureWindowsPath(member_name)
    posix_path = pathlib.PurePosixPath(member_name)

    if (
        windows_path.is_absolute()
        or windows_path.drive
        or posix_path.is_absolute()
        or not posix_path.parts
        or "\\" in member_name
        or ".." in posix_path.parts
    ):
        raise ValueError(f"unsafe zip member path: {member_name}")

    file_type = stat.S_IFMT(zip_info.external_attr >> 16)
    if file_type == stat.S_IFLNK:
        raise ValueError(f"refusing to extract symlink zip member: {member_name}")

    target_path = (destination / pathlib.Path(*posix_path.parts)).resolve()
    if not target_path.is_relative_to(destination.resolve()):
        raise ValueError(f"unsafe zip member path: {member_name}")
    return target_path


def safe_extract_zip(
    zip_file: zipfile.ZipFile,
    destination: str | os.PathLike,
    members: Iterable[str | zipfile.ZipInfo] | None = None,
) -> list[pathlib.Path]:
    """Safely extract zip members into a destination directory.

    This rejects absolute paths, parent-directory traversal, Windows drive paths,
    backslash-separated paths, and symlinks before writing any member.
    """
    destination_path = pathlib.Path(destination).resolve()
    destination_path.mkdir(parents=True, exist_ok=True)

    zip_members = members
    if zip_members is None:
        zip_members = zip_file.infolist()

    extracted_paths = []
    for member in zip_members:
        zip_info = (
            member if isinstance(member, zipfile.ZipInfo) else zip_file.getinfo(member)
        )
        target_path = _validate_zip_member(zip_info, destination_path)

        if zip_info.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zip_file.open(zip_info) as src, target_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        extracted_paths.append(target_path)

    return extracted_paths


def safe_extract_zip_member(
    zip_file: zipfile.ZipFile,
    member: str | zipfile.ZipInfo,
    destination: str | os.PathLike,
) -> pathlib.Path:
    """Safely extract one zip member and return its local path."""
    return safe_extract_zip(zip_file, destination, [member])[0]
