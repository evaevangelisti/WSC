"""How the pipeline reads and writes the files it keeps."""

import bz2
import gzip
from collections.abc import Generator
from contextlib import contextmanager
from io import BufferedIOBase
from pathlib import Path
from typing import IO, Literal, overload


@overload
def open_compressed(
    input_path: Path,
    mode: Literal["rb"],
) -> BufferedIOBase: ...


@overload
def open_compressed(
    input_path: Path,
    mode: Literal["rt"],
) -> IO[str]: ...


def open_compressed(
    input_path: Path,
    mode: Literal["rb", "rt"],
) -> BufferedIOBase | IO[str]:
    """
    Open a file for reading, decompressing it by the suffix it carries.

    Args:
        input_path: The file to read.
        mode: Binary or text reading mode.

    Returns:
        The open file.
    """
    encoding = None if mode == "rb" else "utf-8"

    match input_path.suffix:
        case ".gz":
            return gzip.open(input_path, mode, encoding=encoding)

        case ".bz2":
            return bz2.open(input_path, mode, encoding=encoding)

        case ".zst":
            from compression import zstd

            return zstd.open(input_path, mode, encoding=encoding)

        case _:
            return input_path.open(mode, encoding=encoding)


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
