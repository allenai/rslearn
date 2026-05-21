"""Tests for archive helpers."""

import pathlib
import stat
import zipfile

import pytest

from rslearn.utils.archive import safe_extract_zip, safe_extract_zip_member


def test_safe_extract_zip_extracts_nested_file(tmp_path: pathlib.Path) -> None:
    """Verify safe extraction preserves normal nested paths."""
    archive_path = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive_path, "w") as zip_file:
        zip_file.writestr("nested/file.txt", "ok")

    destination = tmp_path / "out"
    with zipfile.ZipFile(archive_path) as zip_file:
        extracted_paths = safe_extract_zip(zip_file, destination)

    assert extracted_paths == [destination / "nested" / "file.txt"]
    assert (destination / "nested" / "file.txt").read_text() == "ok"


@pytest.mark.parametrize(
    "member_name",
    [
        "../escape.txt",
        "/tmp/escape.txt",
        "C:/tmp/escape.txt",
        "",
        r"..\escape.txt",
        r"nested\escape.txt",
    ],
)
def test_safe_extract_zip_rejects_unsafe_paths(
    tmp_path: pathlib.Path, member_name: str
) -> None:
    """Verify unsafe zip member paths are rejected before extraction."""
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as zip_file:
        zip_file.writestr(member_name, "bad")

    destination = tmp_path / "out"
    with zipfile.ZipFile(archive_path) as zip_file:
        with pytest.raises(ValueError, match="unsafe zip member path"):
            safe_extract_zip(zip_file, destination)

    assert not (tmp_path / "escape.txt").exists()


def test_safe_extract_zip_rejects_symlink_members(tmp_path: pathlib.Path) -> None:
    """Verify symlink entries are rejected."""
    archive_path = tmp_path / "symlink.zip"
    zip_info = zipfile.ZipInfo("link")
    zip_info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive_path, "w") as zip_file:
        zip_file.writestr(zip_info, "target")

    with zipfile.ZipFile(archive_path) as zip_file:
        with pytest.raises(ValueError, match="refusing to extract symlink"):
            safe_extract_zip(zip_file, tmp_path / "out")


def test_safe_extract_zip_member_returns_extracted_path(
    tmp_path: pathlib.Path,
) -> None:
    """Verify single-member extraction returns the extracted path."""
    archive_path = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive_path, "w") as zip_file:
        zip_file.writestr("file.txt", "ok")

    destination = tmp_path / "out"
    with zipfile.ZipFile(archive_path) as zip_file:
        extracted_path = safe_extract_zip_member(zip_file, "file.txt", destination)

    assert extracted_path == destination / "file.txt"
    assert extracted_path.read_text() == "ok"
