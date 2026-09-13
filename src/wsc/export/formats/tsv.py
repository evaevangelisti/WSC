"""TSV output with caller-defined columns and rows."""

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, override

from ..base import Writer


class TSVWriter(Writer[Mapping[str, object]]):
    """Write mappings as tab-separated rows under caller-defined columns."""

    def __init__(
        self,
        path: Path,
        fields: Sequence[str],
    ) -> None:
        """
        Prepare an atomic tabular writer.

        Args:
            path: TSV destination.
            fields: Column names in output order.
        """
        super().__init__(path)

        self._fields: tuple[str, ...] = tuple(fields)

        self._file: IO[str] | None = None

        self._writer: csv.DictWriter[str] | None = None

    @override
    def _open(
        self,
        path: Path,
    ) -> None:
        """
        Open the temporary output and write its column names.

        Args:
            path: Temporary output path.
        """
        self._file = path.open("w", encoding="utf-8", newline="")

        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self._fields,
            delimiter="\t",
        )

        self._writer.writeheader()

    @override
    def _close(
        self,
    ) -> None:
        """Flush and close the tabular stream."""
        if self._file is not None:
            self._file.close()
            self._file = None

        self._writer = None

    @override
    def write(
        self,
        item: Mapping[str, object],
    ) -> None:
        """
        Write one mapping using standard CSV quoting and value conversion.

        Args:
            item: Column values; omitted columns produce empty cells.

        Raises:
            RuntimeError: If the writer has not been entered.
            ValueError: If the mapping contains undeclared columns.
        """
        if self._writer is None:
            raise RuntimeError("TSVWriter must be used as a context manager")

        self._writer.writerow(item)

    def flush(
        self,
    ) -> None:
        """
        Make buffered rows visible in the partial file.

        Raises:
            RuntimeError: If the writer has not been entered.
        """
        if self._file is None:
            raise RuntimeError("TSVWriter must be used as a context manager")

        self._file.flush()
