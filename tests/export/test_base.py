"""
Tests for src/wsc/export/base.py.

The atomicity every format inherits is tested through the writer open_writer
hands back, so that no format written only for a test has to exist. The one
exception is below: nothing a test can do to a file makes closing it fail, so
a writer that cannot be closed is written here rather than arranged for.
"""

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import override

import pytest
from hypothesis import given
from hypothesis import strategies as st
from strategies import lemmas

from wsc.export import Writer, open_writer
from wsc.export.formats.jsonl import JsonlWriter
from wsc.models import POS, Lemma

_WRITTEN = st.lists(lemmas, max_size=3)

_DIRECTORIES = st.lists(
    st.sampled_from(["out", "deeper", "deepest"]),
    min_size=1,
    max_size=3,
)


class _UnclosableWriter(JsonlWriter):
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


def _write(
    writer: Writer[Lemma],
    written: Iterable[Lemma],
) -> None:
    """
    Write lemmas through a writer, entering it and leaving it.

    Args:
        writer: Where the lemmas go.
        written: The lemmas to write.
    """
    with writer:
        for lemma in written:
            writer.write(lemma)


def _write_then_fail(
    writer: Writer[Lemma],
    written: Iterable[Lemma],
) -> None:
    """
    Write lemmas and fail before the block is left, as a run cut short does.

    Args:
        writer: Where the lemmas go.
        written: The lemmas to write.

    Raises:
        RuntimeError: Always, once they have been written.
    """
    with writer:
        for lemma in written:
            writer.write(lemma)

        raise RuntimeError("something went wrong")


class TestWriter:
    """
    Writing through a .part file, so a run cut short leaves nothing.
    """

    @given(_DIRECTORIES, _WRITTEN)
    def test_creates_the_parent_directory(
        self,
        workspace: Callable[[], Path],
        directories: list[str],
        written: list[Lemma],
    ) -> None:
        """An export may be the first thing written where it is going."""
        output_path = workspace().joinpath(*directories) / "senses.jsonl"

        with open_writer(output_path) as writer:
            for lemma in written:
                writer.write(lemma)

        assert output_path.exists()

    @given(_WRITTEN)
    def test_holds_output_back_until_the_block_is_left(
        self,
        workspace: Callable[[], Path],
        written: list[Lemma],
    ) -> None:
        """Output appears whole, or not at all: a file that is there is finished."""
        output_path = workspace() / "senses.jsonl"

        with open_writer(output_path) as writer:
            for lemma in written:
                writer.write(lemma)

            assert not output_path.exists()

        assert output_path.exists()

    @given(_WRITTEN)
    def test_leaves_no_partial_file_behind(
        self,
        workspace: Callable[[], Path],
        written: list[Lemma],
    ) -> None:
        """The .part file is removed once its contents are in place."""
        directory = workspace()
        output_path = directory / "senses.jsonl"

        with open_writer(output_path) as writer:
            for lemma in written:
                writer.write(lemma)

        assert list(directory.iterdir()) == [output_path]

    @given(_WRITTEN)
    def test_writes_nothing_when_the_block_raises(
        self,
        workspace: Callable[[], Path],
        written: list[Lemma],
    ) -> None:
        """A half-written export cannot be resumed, so none is left behind."""
        directory = workspace()

        with pytest.raises(RuntimeError, match="something went wrong"):
            _write_then_fail(open_writer(directory / "senses.jsonl"), written)

        assert list(directory.iterdir()) == []

    @given(_WRITTEN)
    def test_lets_go_of_the_file_however_the_block_ended(
        self,
        workspace: Callable[[], Path],
        written: list[Lemma],
    ) -> None:
        """Whatever the format acquired is released, a block that failed included."""
        writer = open_writer(workspace() / "senses.jsonl")

        with pytest.raises(RuntimeError, match="something went wrong"):
            _write_then_fail(writer, written)

        with pytest.raises(RuntimeError, match="context manager"):
            writer.write(Lemma("bank.noun.1", "bank", POS.NOUN))

    @given(_WRITTEN)
    def test_writes_nothing_when_closing_fails(
        self,
        workspace: Callable[[], Path],
        written: list[Lemma],
    ) -> None:
        """
        Closing is where buffered output reaches the disk.

        A failure there leaves the file short of what was written to it, so
        it counts as a run that failed.
        """
        directory = workspace()

        with pytest.raises(OSError, match="the disk filled up"):
            _write(_UnclosableWriter(directory / "senses.jsonl"), written)

        assert list(directory.iterdir()) == []
