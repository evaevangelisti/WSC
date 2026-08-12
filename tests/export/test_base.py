"""
Tests for src/wsc/export/base.py.

The atomicity every format inherits is tested here, on a writer of its own,
so that a new format need only be tested on what it writes.
"""

from pathlib import Path
from typing import IO, override

import pytest

from wsc.export import Writer


class _TextWriter(Writer[str]):
    """
    The plainest writer there could be: one line per item.

    Attributes:
        closed: Whether closing was reached, however the block ended.
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
        self.closed: bool = False

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
        """Close the file, and record that it happened."""
        self.closed = True

        if self._file is not None:
            self._file.close()
            self._file = None

    @override
    def write(
        self,
        item: str,
    ) -> None:
        """
        Append one line.

        Args:
            item: The line to write.

        Raises:
            RuntimeError: If the writer has not been entered.
        """
        if self._file is None:
            raise RuntimeError("Writer is not open; use it as a context manager")

        _ = self._file.write(f"{item}\n")


class _FailingWriter(_TextWriter):
    """
    A writer that cannot be closed, however well the writing itself went.
    """

    @override
    def _close(
        self,
    ) -> None:
        """
        Fail where a format would fail: flushing what is still buffered.

        Raises:
            OSError: Always.
        """
        super()._close()

        raise OSError("the disk filled up")


class TestWriter:
    """
    Writing through a .part file, so a run cut short leaves nothing.
    """

    def test_touches_nothing_before_it_is_entered(
        self,
        tmp_path: Path,
    ) -> None:
        """Building a writer settles where output goes, and nothing more."""
        _ = _TextWriter(tmp_path / "out" / "senses.txt")

        assert not (tmp_path / "out").exists()

    def test_creates_the_parent_directory(
        self,
        tmp_path: Path,
    ) -> None:
        """An export may be the first thing written where it is going."""
        output_path = tmp_path / "deep" / "deeper" / "senses.txt"

        with _TextWriter(output_path) as writer:
            writer.write("a line")

        assert output_path.exists()

    def test_holds_output_back_until_the_block_is_left(
        self,
        tmp_path: Path,
    ) -> None:
        """Output appears whole, or not at all: a file that is there is finished."""
        output_path = tmp_path / "senses.txt"

        with _TextWriter(output_path) as writer:
            writer.write("a line")

            assert not output_path.exists()

        assert output_path.read_text(encoding="utf-8") == "a line\n"

    def test_leaves_no_partial_file_behind(
        self,
        tmp_path: Path,
    ) -> None:
        """The .part file is removed once its contents are in place."""
        output_path = tmp_path / "senses.txt"

        with _TextWriter(output_path) as writer:
            writer.write("a line")

        assert list(tmp_path.iterdir()) == [output_path]

    def test_writes_nothing_when_the_block_raises(
        self,
        tmp_path: Path,
    ) -> None:
        """A half-written export cannot be resumed, so none is left behind."""
        output_path = tmp_path / "senses.txt"

        def write_then_fail() -> None:
            with _TextWriter(output_path) as writer:
                writer.write("a line")

                raise RuntimeError("something went wrong")

        with pytest.raises(RuntimeError, match="something went wrong"):
            write_then_fail()

        assert list(tmp_path.iterdir()) == []

    def test_closes_the_file_even_when_the_block_raises(
        self,
        tmp_path: Path,
    ) -> None:
        """Whatever the format acquired is released, however the block ended."""
        writer = _TextWriter(tmp_path / "senses.txt")

        with pytest.raises(RuntimeError), writer:
            raise RuntimeError("something went wrong")

        assert writer.closed

    def test_writes_nothing_when_closing_fails(
        self,
        tmp_path: Path,
    ) -> None:
        """
        Closing is where buffered output reaches the disk.

        A failure there leaves the file short of what was written to it, so
        it counts as a run that failed.
        """
        output_path = tmp_path / "senses.txt"

        with (
            pytest.raises(OSError, match="the disk filled up"),
            _FailingWriter(output_path) as writer,
        ):
            writer.write("a line")

        assert list(tmp_path.iterdir()) == []

    def test_refuses_to_write_before_it_is_entered(
        self,
        tmp_path: Path,
    ) -> None:
        """There is nothing to write to until the block opens it."""
        writer = _TextWriter(tmp_path / "senses.txt")

        with pytest.raises(RuntimeError, match="context manager"):
            writer.write("a line")
