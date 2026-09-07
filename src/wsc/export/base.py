"""Atomic writing, shared by every output format."""

from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType
from typing import Self


class Writer[T](ABC):
    """
    A sink for items of one kind, writing atomically.

    Output goes to a sibling .part file, moved into place only once writing finishes. A
    run that fails leaves neither behind.
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
        self._output_path: Path = output_path
        self._partial_path: Path = output_path.with_name(f"{output_path.name}.part")

    def __enter__(
        self,
    ) -> Self:
        """
        Create the parent directory and start writing to the .part file.

        Returns:
            The writer, ready to accept items.
        """
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._open(self._partial_path)

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Move the .part file into place, or remove it if anything went wrong."""
        succeeded = exc_type is None

        try:
            self._close()
        except Exception:
            # Closing flushes buffered output before the atomic rename.
            succeeded = False
            raise
        finally:
            if succeeded:
                _ = self._partial_path.replace(self._output_path)
            else:
                self._partial_path.unlink(missing_ok=True)

    @abstractmethod
    def _open(
        self,
        path: Path,
    ) -> None:
        """
        Acquire whatever the format needs in order to write.

        Args:
            path: Where to write; always the .part file, never the final one.
        """

    @abstractmethod
    def _close(
        self,
    ) -> None:
        """Flush any buffered output and release what _open acquired."""

    @abstractmethod
    def write(
        self,
        item: T,
    ) -> None:
        """
        Append one item.

        Args:
            item: The item to write.

        Raises:
            RuntimeError: If the writer has not been entered.
        """
