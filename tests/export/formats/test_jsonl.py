"""
Tests for src/wsc/export/formats/jsonl.py.

Complete document comparisons detect missing and unexpected fields.
"""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from strategies import RawJson, lemmas, words

from wsc.export import Writer, open_writer
from wsc.models import (
    POS,
    Lemma,
    Quotation,
    Sense,
    Sentence,
    WordNetAlignment,
    WordNetRelation,
)

_WRITTEN = st.lists(lemmas, max_size=3)


def _serialize_sentence(
    sentence: Sentence,
) -> RawJson:
    """
    Spell out what one sentence is expected to read as.

    Args:
        sentence: The example or quotation written.

    Returns:
        The JSON object it becomes, without the keys holding nothing.
    """
    record: RawJson = {"text": sentence.text}

    if sentence.word_offsets:
        record["word_offsets"] = [
            {
                "offset": list(word_offset.offset),
                "sources": list(word_offset.sources),
            }
            for word_offset in sentence.word_offsets
        ]

    if isinstance(sentence, Quotation):
        record["reference"] = sentence.reference

        if sentence.year is not None:
            record["year"] = sentence.year

    return record


def _serialize_sense(
    sense: Sense,
) -> RawJson:
    """
    Spell out what one sense is expected to read as.

    Args:
        sense: The meaning written.

    Returns:
        The JSON object it becomes, without the keys holding nothing.
    """
    record: RawJson = {"id": sense.id, "glosses": list(sense.glosses)}

    if sense.translations:
        record["translations"] = {
            language: sorted(words) for language, words in sense.translations.items()
        }

    if sense.etymology:
        record["etymology"] = sense.etymology

    for key, values in (
        ("synonyms", sense.synonyms),
        ("topics", sense.topics),
        ("tags", sense.tags),
        ("sentences", tuple(map(_serialize_sentence, sense.sentences))),
        ("wikidata_ids", sense.wikidata_ids),
    ):
        if values:
            record[key] = list(values)

    if sense.wordnet:
        record["wordnet"] = [
            {"synset_id": alignment.synset_id, "relation": alignment.relation}
            for alignment in sense.wordnet
        ]

    return record


def _serialize_lemma(
    lemma: Lemma,
) -> RawJson:
    """
    Spell out what one lemma is expected to read as.

    Args:
        lemma: The lemma written.

    Returns:
        The JSON object it becomes, without the keys holding nothing.
    """
    record: RawJson = {"id": lemma.id, "lemma": lemma.lemma, "pos": lemma.pos.value}

    if lemma.senses:
        record["senses"] = [_serialize_sense(sense) for sense in lemma.senses]

    if lemma.variants:
        record["variants"] = sorted(lemma.variants)

    if lemma.translation_tables:
        record["translation_tables"] = [
            {
                key: value
                for key, value in {
                    "id": table.id,
                    "gloss": table.gloss,
                    "translations": {
                        language: sorted(words)
                        for language, words in table.translations.items()
                    },
                }.items()
                if value not in ("", {})
            }
            for table in lemma.translation_tables
        ]

    return record


@pytest.fixture
def write(
    workspace: Callable[[], Path],
) -> Callable[..., str]:
    """
    Write lemmas and hand back the file as it stands on disk.

    Args:
        workspace: Sets aside a directory for the file being written.

    Returns:
        A writer returning the complete serialized text.
    """

    def run(
        *written: Lemma,
    ) -> str:
        output_path = workspace() / "senses.jsonl"

        writer: Writer[Lemma] = open_writer(output_path)

        with writer:
            for lemma in written:
                writer.write(lemma)

        return output_path.read_text(encoding="utf-8")

    return run


class TestJSONLWriter:
    """One JSON object per line."""

    @given(_WRITTEN)
    def test_writes_json_lines(
        self,
        write: Callable[..., str],
        written: list[Lemma],
    ) -> None:
        """A line at a time is what lets a reader stream the file back."""
        text = write(*written)

        assert text.count("\n") == len(written)
        assert not text or text.endswith("\n")

    @given(_WRITTEN)
    def test_serializes_lemma_fields(
        self,
        write: Callable[..., str],
        written: list[Lemma],
    ) -> None:
        """Nothing the extractor gathered is dropped, and nothing empty is kept."""
        lines = [line for line in write(*written).split("\n") if line]

        assert [json.loads(line) for line in lines] == [
            _serialize_lemma(lemma) for lemma in written
        ]

    @given(words)
    def test_preserves_unicode_text(
        self,
        write: Callable[..., str],
        headword: str,
    ) -> None:
        """Serialized text preserves Unicode characters."""
        assert headword in write(Lemma(f"{headword}.noun.1", headword, POS.NOUN))

    def test_writes_alignment_fields(
        self,
        write: Callable[..., str],
    ) -> None:
        """An alignment becomes part of its sense's record."""
        sense = Sense(
            "bank.noun.1",
            ("A financial institution.",),
            translations={"it": frozenset({"banca"})},
            wordnet=(WordNetAlignment("i54321", WordNetRelation.EQUIVALENT),),
        )

        text = write(Lemma("bank.noun", "bank", POS.NOUN, senses=[sense]))

        assert json.loads(text)["senses"] == [
            {
                "id": "bank.noun.1",
                "glosses": ["A financial institution."],
                "translations": {"it": ["banca"]},
                "wordnet": [{"synset_id": "i54321", "relation": "equivalent"}],
            },
        ]

    def test_rejects_unopened_writer(
        self,
        workspace: Callable[[], Path],
    ) -> None:
        """The file is opened on entry, so there is nowhere to write before it."""
        writer: Writer[Lemma] = open_writer(workspace() / "senses.jsonl")

        with pytest.raises(RuntimeError, match="context manager"):
            writer.write(Lemma("bank.noun.1", "bank", POS.NOUN))

    def test_rejects_closed_writer(
        self,
        workspace: Callable[[], Path],
    ) -> None:
        """Closing releases the file handle."""
        writer: Writer[Lemma] = open_writer(workspace() / "senses.jsonl")
        lemma = Lemma("bank.noun.1", "bank", POS.NOUN)

        with writer:
            writer.write(lemma)

        with pytest.raises(RuntimeError, match="context manager"):
            writer.write(lemma)
