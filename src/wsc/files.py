"""
How the pipeline reads and writes the files it keeps.
"""

import bz2
import gzip
from collections.abc import Generator
from contextlib import contextmanager
from enum import StrEnum
from io import BufferedIOBase
from pathlib import Path
from typing import IO, Literal, overload

COMPRESSED_SUFFIXES = (".gz", ".bz2", ".zst")


class Compression(StrEnum):
    """
    Supported compression formats for generated files.
    """

    GZIP = "gz"
    BZIP2 = "bz2"
    ZSTANDARD = "zst"


def compressed_name(
    name: str,
    compression: Compression | None,
) -> str:
    """
    Append the selected compression suffix to an output filename.

    Args:
        name: Uncompressed filename.
        compression: Requested compression, if any.

    Returns:
        The filename to write.
    """
    return f"{name}.{compression}" if compression else name


def existing_file(
    path: Path,
) -> Path:
    """
    Find a cached file under its plain or compressed name.

    Args:
        path: Uncompressed cache path.

    Returns:
        The first existing path, or the original path when absent.
    """
    if path.exists():
        return path

    base = path.with_suffix("") if path.suffix in COMPRESSED_SUFFIXES else path

    for suffix in ("", *COMPRESSED_SUFFIXES):
        candidate = base.with_name(f"{base.name}{suffix}")

        if candidate.exists():
            return candidate

    return path


def format_suffix(
    path: Path,
) -> str:
    """
    Find the format suffix beneath compression and partial-file suffixes.

    Args:
        path: File whose logical format is needed.

    Returns:
        The lowercase format suffix, or an empty string.
    """
    suffixes = [suffix.lower() for suffix in path.suffixes]

    if suffixes and suffixes[-1] == ".part":
        _ = suffixes.pop()

    if suffixes and suffixes[-1] in COMPRESSED_SUFFIXES:
        _ = suffixes.pop()

    return suffixes[-1] if suffixes else ""


@overload
def open_compressed(
    path: Path,
    mode: Literal["rb", "wb"],
) -> BufferedIOBase | IO[bytes]: ...


@overload
def open_compressed(
    path: Path,
    mode: Literal["rt", "wt"],
    *,
    newline: str | None = None,
) -> IO[str]: ...


def open_compressed(
    path: Path,
    mode: Literal["rb", "rt", "wb", "wt"],
    *,
    newline: str | None = None,
) -> BufferedIOBase | IO[bytes] | IO[str]:
    """
    Open a file using the compression named by its suffix.

    Args:
        path: File to read or write.
        mode: Binary or text reading or writing mode.
        newline: Text newline policy, when needed by CSV files.

    Returns:
        The open file.
    """
    suffix = path.with_suffix("").suffix if path.suffix == ".part" else path.suffix
    suffix = suffix.lower()

    if mode in ("rb", "wb"):
        match suffix:
            case ".gz":
                return gzip.open(path, mode)

            case ".bz2":
                return bz2.open(path, mode)

            case ".zst":
                from compression import zstd

                return zstd.open(path, mode)

            case _:
                return path.open(mode)

    match suffix:
        case ".gz":
            return gzip.open(path, mode, encoding="utf-8", newline=newline)

        case ".bz2":
            return bz2.open(path, mode, encoding="utf-8", newline=newline)

        case ".zst":
            from compression import zstd

            return zstd.open(path, mode, encoding="utf-8", newline=newline)

        case _:
            return path.open(mode, encoding="utf-8", newline=newline)


@contextmanager
def partial_file(
    output_path: Path,
    *,
    resumable: bool = False,
) -> Generator[Path]:
    """
    Hand over a sibling .part file, moved into place once the block ends well.

    Resumable transfers retain partial output after failure.

    Args:
        output_path: Where the finished file is placed.
        resumable: Whether a failure leaves what was written behind.

    Yields:
        The file to write to, never the final one.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(f"{output_path.name}.part")

    try:
        yield partial_path
    except BaseException:
        if not resumable:
            partial_path.unlink(missing_ok=True)

        raise

    _ = partial_path.replace(output_path)
