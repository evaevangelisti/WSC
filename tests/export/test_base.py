"""
Tests for src/wsc/export/base.py.

The atomicity every format inherits is tested through the writer open_writer hands back,
so no format written only for a test has to exist.
"""

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import override

import pytest
from hypothesis import given
from hypothesis import strategies as st
from strategies import lemmas

from wsc.export import Writer, open_writer
from wsc.export.formats.jsonl import JSONLWriter
from wsc.models import POS, Lemma

_WRITTEN = st.lists(lemmas, max_size=3)

_DIRECTORIES = st.lists(
    st.sampled_from(["out", "deeper", "deepest"]),
    min_size=1,
    max_size=3,
)


def _open(
    output_path: Path,
) -> Writer[Lemma]:
    """
    Open the writer these properties are stated through.

    Lemma fixtures exercise atomic writer behavior.

    Args:
        output_path: Where the finished file is placed.

    Returns:
        A writer for that path, not yet open.
    """
    return open_writer(output_path)


class _UnclosableWriter(JSONLWriter[Lemma]):
    """A writer that cannot be closed, however well the writing itself went."""

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


def _interrupt_writing(
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
    """Writing through a .part file, so a run cut short leaves nothing."""

    @given(_DIRECTORIES, _WRITTEN)
    def test_publishes_complete_output(
        self,
        workspace: Callable[[], Path],
        directories: list[str],
        written: list[Lemma],
    ) -> None:
        """Writing creates parent directories and publishes output only on closure."""
        output_path = workspace().joinpath(*directories) / "senses.jsonl"

        with _open(output_path) as writer:
            for lemma in written:
                writer.write(lemma)

            assert not output_path.exists()

        assert list(output_path.parent.iterdir()) == [output_path]

    @given(_WRITTEN, st.binary(max_size=100))
    def test_discards_failed_output(
        self,
        workspace: Callable[[], Path],
        written: list[Lemma],
        original: bytes,
    ) -> None:
        """A failed replacement preserves completed output and removes partial data."""
        directory = workspace()
        output_path = directory / "senses.jsonl"
        _ = output_path.write_bytes(original)

        with pytest.raises(RuntimeError, match="something went wrong"):
            _interrupt_writing(_open(output_path), written)

        assert output_path.read_bytes() == original
        assert list(directory.iterdir()) == [output_path]

    @given(_WRITTEN)
    def test_closes_output_stream(
        self,
        workspace: Callable[[], Path],
        written: list[Lemma],
    ) -> None:
        """Whatever the format acquired is released, a block that failed included."""
        writer = _open(workspace() / "senses.jsonl")

        with pytest.raises(RuntimeError, match="something went wrong"):
            _interrupt_writing(writer, written)

        with pytest.raises(RuntimeError, match="context manager"):
            writer.write(Lemma("bank.noun.1", "bank", POS.NOUN))

    @given(_WRITTEN)
    def test_discards_unflushed_output(
        self,
        workspace: Callable[[], Path],
        written: list[Lemma],
    ) -> None:
        """
        Closing is where buffered output reaches the disk.

        A failure there leaves the file short of what was written to it, so it counts as
        a run that failed.
        """
        directory = workspace()

        with pytest.raises(OSError, match="the disk filled up"):
            _write(_UnclosableWriter(directory / "senses.jsonl"), written)

        assert list(directory.iterdir()) == []
