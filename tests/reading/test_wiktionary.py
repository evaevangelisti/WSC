"""Collected entries retain their evidence through JSONL export and reading."""

from collections.abc import Callable
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st
from strategies import lemmas

from wsc.export import Writer, open_writer
from wsc.models import Lemma
from wsc.reading import read_lemmas


@given(st.lists(lemmas, max_size=4))
def test_restores_collected_evidence(
    workspace: Callable[[], Path],
    entries: list[Lemma],
) -> None:
    """Reading restores variants, translation tables, quotations, and offset sources."""
    path = workspace() / "senses.jsonl"
    writer: Writer[Lemma] = open_writer(path)

    with writer:
        for entry in entries:
            writer.write(entry)

    assert list(read_lemmas(path)) == entries
