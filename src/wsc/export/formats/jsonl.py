"""
JSONL output, one JSON object per line.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import IO, override

from ...models import Lemma
from ..base import Writer

type Json = (
    str | int | float | bool | list[Json] | tuple[Json, ...] | dict[str, Json] | None
)


class JsonlWriter(Writer[Lemma]):
    """
    Write lemmas as JSON objects, one per line.
    """

    def __init__(
        self,
        output_path: Path,
    ) -> None:
        """
        Prepare a writer, without touching the filesystem yet.

        Args:
            output_path: Where the finished file is placed.
        """
        super().__init__(output_path)
        self._file: IO[str] | None = None

    @override
    def _open(
        self,
        path: Path,
    ) -> None:
        """
        Open the file for writing.

        Args:
            path: Where to write; always the .part file, never the final one.
        """
        self._file = path.open("w", encoding="utf-8")

    @override
    def _close(
        self,
    ) -> None:
        """Close the file, flushing whatever is still buffered."""
        if self._file is not None:  # pragma: no branch
            self._file.close()
            self._file = None

    @classmethod
    def _prune(
        cls,
        value: Json,
    ) -> Json:
        """
        Drop keys holding nothing, recursively.

        Absence says as much as emptiness, and in fewer bytes: a sentence
        with no reference reads back as an Example.

        Args:
            value: A value taken from a dataclass dump.

        Returns:
            The same value, with every empty key gone.
        """
        match value:
            case dict():
                return {
                    k: cls._prune(v)
                    for k, v in value.items()
                    if v not in (None, (), [], {})
                }

            case list() | tuple():
                return [cls._prune(item) for item in value]

            case _:
                return value

    @override
    def write(
        self,
        item: Lemma,
    ) -> None:
        """
        Append one lemma as a line of JSON.

        Args:
            item: The lemma to write.

        Raises:
            RuntimeError: If the writer has not been entered.
        """
        if self._file is None:
            raise RuntimeError("Writer is not open; use it as a context manager")

        record = self._prune(asdict(item))
        _ = self._file.write(f"{json.dumps(record, ensure_ascii=False)}\n")
