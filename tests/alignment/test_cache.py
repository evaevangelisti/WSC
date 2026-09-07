"""
Public TSV persistence and collection-reading contracts.
"""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from wsc.alignment import serialize_alignment
from wsc.constants import ALIGNMENT_FIELDS
from wsc.export import TSVWriter, Writer, open_writer
from wsc.models import (
    POS,
    Lemma,
    Quotation,
    Sense,
    WordNetAlignment,
    WordNetRelation,
    WordOffset,
    WordOffsetSource,
)
from wsc.models.alignment import (
    AlignmentQuery,
    AlignmentResult,
    AlignmentScore,
    AlignmentTask,
    Definition,
)
from wsc.reading import read_alignments, read_lemmas, read_metadata


@given(
    st.text(min_size=1, max_size=60),
    st.floats(min_value=-100, max_value=100, allow_nan=False),
)
def test_tsv_round_trip_preserves_arbitrary_text_and_empty_candidates(
    workspace: Callable[[], Path],
    text: str,
    score: float,
) -> None:
    """Tabs, quotes, newlines, Unicode, and empty candidate sets survive TSV replay."""
    path = workspace() / "wordnet.tsv"
    query = AlignmentQuery(
        AlignmentTask.WORDNET,
        "id",
        "lemma",
        text,
        POS.NOUN,
        (Definition("s", (text,)),),
        (Definition("i1", (text,)),),
    )
    empty = AlignmentQuery(
        AlignmentTask.WORDNET,
        "empty",
        "lemma",
        text,
        POS.NOUN,
        (Definition("s2", (text,)),),
        (),
    )
    results = [
        AlignmentResult(query, (AlignmentScore("s", "i1", "equivalent", score),)),
        AlignmentResult(empty, ()),
    ]
    with TSVWriter(path, ALIGNMENT_FIELDS) as writer:
        writer.write({"context": json.dumps({"model": text}, ensure_ascii=False)})
        for result in results:
            for row in serialize_alignment(result):
                writer.write(row)

    assert read_metadata(path) == {"model": text}
    assert list(read_alignments(path)) == results


def test_failed_cache_write_preserves_previous_completed_file(tmp_path: Path) -> None:
    """Interrupted inference never replaces completed evidence."""
    path = tmp_path / "translations.tsv"
    _ = path.write_text("previous", encoding="utf-8")

    with (
        pytest.raises(RuntimeError, match="inference failed"),
        TSVWriter(path, ALIGNMENT_FIELDS) as _writer,
    ):
        raise RuntimeError("inference failed")

    assert path.read_text(encoding="utf-8") == "previous"
    assert not path.with_name("translations.tsv.part").exists()


def test_collection_round_trip_preserves_source_fields_and_relations(
    tmp_path: Path,
) -> None:
    """Alignment reading retains source fields and existing associations."""
    lemma = Lemma(
        "word.noun",
        "word",
        POS.NOUN,
        variants=frozenset({"words.noun"}),
        senses=[
            Sense(
                "s",
                ("parent", "leaf"),
                etymology="origin",
                topics=("topic",),
                tags=("tag",),
                synonyms=("term",),
                sentences=[
                    Quotation(
                        "a word",
                        "reference",
                        1900,
                        word_offsets=(
                            WordOffset(
                                (2, 6),
                                (WordOffsetSource.BOLD, WordOffsetSource.LEMMATIZER),
                            ),
                        ),
                    )
                ],
                translations={"it": frozenset({"parola"})},
                wikidata_ids=("Q1",),
                wordnet=(WordNetAlignment("i1", WordNetRelation.WIKTIONARY_NARROWER),),
            )
        ],
    )
    path = tmp_path / "collection.jsonl"
    writer: Writer[Lemma] = open_writer(path)
    with writer:
        writer.write(lemma)

    assert list(read_lemmas(path)) == [lemma]
    assert "translations" not in json.loads(path.read_text(encoding="utf-8"))
