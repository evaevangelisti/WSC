"""
Collected entries retain their evidence through JSONL export and reading.
"""

from collections.abc import Callable
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st
from strategies import lemmas

from wsc.export import Writer, open_writer
from wsc.models import POS, Language, Lemma, TranslationTable
from wsc.reading import read_lemmas


@given(st.lists(lemmas, max_size=4))
def test_restores_collected_evidence(
    workspace: Callable[[], Path],
    entries: list[Lemma],
) -> None:
    """
    Reading restores variants, translation tables, quotations, and offset sources.
    """
    path = workspace() / "senses.jsonl"
    writer: Writer[Lemma] = open_writer(path)

    with writer:
        for entry in entries:
            writer.write(entry)

    assert list(read_lemmas(path)) == entries


def test_restores_translation_language_labels(
    workspace: Callable[[], Path],
) -> None:
    """
    Export and reading preserve names without changing grouped translations.

    Args:
        workspace: Sets aside a directory for the file being written.
    """
    entry = Lemma(
        "bank.noun",
        "bank",
        POS.NOUN,
        translation_tables=(
            TranslationTable(
                "bank.noun.tr.1",
                "Financial institution.",
                {Language("it", "Italian"): frozenset({"banca"})},
            ),
        ),
    )
    path = workspace() / "senses.jsonl"

    writer: Writer[Lemma] = open_writer(path)

    with writer:
        writer.write(entry)

    assert list(read_lemmas(path)) == [entry]
