"""
Writing of extracted data to disk.
"""

from collections.abc import Callable
from pathlib import Path

from ..models import Lemma
from .base import Writer


def _jsonl(
    output_path: Path,
) -> Writer[Lemma]:
    """
    Build the JSONL writer.

    Args:
        output_path: Where the finished file is placed.

    Returns:
        A writer for that path, not yet open.
    """
    from .formats.jsonl import JsonlWriter

    return JsonlWriter(output_path)


_WRITERS: dict[str, Callable[[Path], Writer[Lemma]]] = {
    ".jsonl": _jsonl,
}


def open_writer(
    output_path: Path,
) -> Writer[Lemma]:
    """
    Open a writer for the format the file extension names.

    The file itself is opened when the writer is entered, not here.

    Args:
        output_path: Where to write; its suffix selects the format.

    Returns:
        A writer for that format, not yet open.

    Raises:
        ValueError: If the suffix names no known format.
    """
    factory = _WRITERS.get(output_path.suffix.lower())
    if factory is None:
        known = ", ".join(sorted(_WRITERS))
        raise ValueError(f"Unknown format {output_path.suffix!r}; try one of {known}")

    return factory(output_path)


__all__ = [
    "Writer",
    "open_writer",
]
