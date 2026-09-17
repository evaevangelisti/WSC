"""Publish staged report artifacts atomically."""

import json
from collections.abc import Iterable, Mapping
from pathlib import Path


def stage_json(
    path: Path,
    document: Mapping[str, object],
) -> None:
    """
    Write one indented JSON document to a staging path.

    The document is encoded as UTF-8 and terminated with a newline.

    Args:
        path: Staging path for the JSON document.
        document: JSON-compatible document to serialize.
    """
    _ = path.write_text(
        json.dumps(document, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )


def publish_files(
    staging_dir: Path,
    output_dir: Path,
    files: Iterable[str],
) -> None:
    """
    Replace published files with their staged counterparts.

    Parent directories are created before each replacement.

    Args:
        staging_dir: Directory containing the staged files.
        output_dir: Directory receiving the published files.
        files: Relative file paths to publish in order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in files:
        staged = staging_dir / name

        destination = output_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)

        _ = staged.replace(destination)
