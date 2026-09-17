"""Build common manifest fields."""

from datetime import UTC, datetime
from pathlib import Path


def describe_source(
    path: Path,
) -> dict[str, object]:
    """
    Describe a source file for a manifest.

    The description includes its resolved path, size, and modification time.

    Args:
        path: Source file to describe.

    Returns:
        Manifest-compatible source metadata.

    Raises:
        OSError: If the source metadata cannot be read.
    """
    status = path.stat()

    return {
        "path": str(path.resolve()),
        "bytes": status.st_size,
        "modified_at": datetime.fromtimestamp(status.st_mtime, UTC).isoformat(),
    }
