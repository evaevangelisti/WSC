"""
Read and publish compressed directory archives.
"""

import tarfile
from collections.abc import Iterator
from pathlib import Path

from .files import Compression, open_compressed, partial_file


def archive_path(
    directory: Path,
    compression: Compression,
) -> Path:
    """
    Name the compressed archive beside its source directory.

    Args:
        directory: Directory represented by the archive.
        compression: Compression format for the tar archive.

    Returns:
        Path to the compressed tar archive.
    """
    return directory.with_name(f"{directory.name}.tar.{compression}")


def publish_archive(
    directory: Path,
    output_dir: Path,
    compression: Compression,
) -> None:
    """
    Publish a complete compressed archive of staged output files.

    Args:
        directory: Staged directory containing finished output files.
        output_dir: Logical output directory and archive name.
        compression: Compression format for the tar archive.
    """
    output_path = archive_path(output_dir, compression)

    with (
        partial_file(output_path) as partial_path,
        open_compressed(partial_path, "wb") as stream,
        tarfile.open(fileobj=stream, mode="w|") as archive,
    ):
        archive.add(directory, arcname=output_dir.name)


def archive_lines(
    path: Path,
    name: str,
) -> Iterator[str]:
    """
    Stream a named file from the root directory of a tar archive.

    Args:
        path: Compressed collection archive.
        name: Filename to read from the archive root directory.

    Yields:
        Lines from the selected file.

    Raises:
        FileNotFoundError: If the archive lacks the named file.
    """
    with (
        open_compressed(path, "rb") as source,
        tarfile.open(fileobj=source, mode="r|") as archive,
    ):
        for member in archive:
            if member.name.rsplit("/", maxsplit=1)[-1] != name or not member.isfile():
                continue

            member_file = archive.extractfile(member)

            if member_file is None:
                raise FileNotFoundError(f"Missing {name} in {path}")

            with member_file:
                for line in member_file:
                    yield line.decode("utf-8")

            return

    raise FileNotFoundError(f"Missing {name} in {path}")
