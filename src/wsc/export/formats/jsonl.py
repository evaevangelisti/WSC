"""
JSONL output, one JSON object per line.
"""

import json
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, cast, override

from ..base import Writer

if TYPE_CHECKING:
    # Typeshed alone declares what a dataclass is known by.
    from _typeshed import DataclassInstance

type Json = (
    str | int | float | bool | list[Json] | tuple[Json, ...] | dict[str, Json] | None
)


class JsonlWriter[T: "DataclassInstance"](Writer[T]):
    """
    Write dataclasses as JSON objects, one per line.
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
    def _record(
        cls,
        value: object,
    ) -> Json:
        """
        Read one value into what is written, dropping what holds nothing.

        Args:
            value: An item to write, or a part of one.

        Returns:
            The same value, as JSON holds it.
        """
        match value:
            case list() | tuple():
                return [cls._record(item) for item in cast(Sequence[object], value)]

            case Mapping():
                return {
                    key: cls._record(carried)
                    for key, carried in cast(Mapping[str, object], value).items()
                }

            # Sorted, so that one collection reads back the same as the next.
            case AbstractSet():
                return [
                    cls._record(item) for item in sorted(cast(AbstractSet[str], value))
                ]

            case _ if is_dataclass(value) and not isinstance(value, type):
                record: dict[str, Json] = {}

                for field in fields(value):
                    carried = cast(object, getattr(value, field.name))

                    if carried not in (None, "", (), [], {}, frozenset()):
                        record[field.name] = cls._record(carried)

                return record

            case _:
                return cast(Json, value)

    @override
    def write(
        self,
        item: T,
    ) -> None:
        """
        Append one item as a line of JSON.

        Args:
            item: The dataclass to write.

        Raises:
            RuntimeError: If the writer has not been entered.
        """
        if self._file is None:
            raise RuntimeError("Writer is not open; use it as a context manager")

        record = self._record(item)
        _ = self._file.write(f"{json.dumps(record, ensure_ascii=False)}\n")
