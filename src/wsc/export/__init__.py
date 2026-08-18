"""
Writing of extracted data to disk.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .base import Writer

if TYPE_CHECKING:
    # Typeshed alone declares what a dataclass is known by.
    from _typeshed import DataclassInstance


class _Factory(Protocol):
    """
    Builds the writer of one format, generic where a plain callable is not.
    """

    def __call__[T: "DataclassInstance"](
        self,
        output_path: Path,
    ) -> Writer[T]: ...


def _jsonl[T: "DataclassInstance"](
    output_path: Path,
) -> Writer[T]:
    """
    Build the JSONL writer.

    Args:
        output_path: Where the finished file is placed.

    Returns:
        A writer for that path, not yet open.
    """
    from .formats.jsonl import JsonlWriter

    return JsonlWriter[T](output_path)


_WRITERS: dict[str, _Factory] = {
    ".jsonl": _jsonl,
}


def open_writer[T: "DataclassInstance"](
    output_path: Path,
) -> Writer[T]:
    """
    Open a writer for the format the file extension names.

    The file itself is opened when the writer is entered, not here. What is
    written is the caller's to say, no format depending on which dataclass
    reaches it.

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
