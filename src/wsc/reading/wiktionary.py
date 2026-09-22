"""Read collected JSONL records without repeating extraction."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from ..files import open_compressed
from ..models import (
    POS,
    Example,
    Lemma,
    Quotation,
    Sense,
    Sentence,
    TranslationTable,
    WordNetAlignment,
    WordNetRelation,
    WordOffset,
    WordOffsetSource,
)


class OffsetRecord(TypedDict):
    """Serialized candidate range and its proposing methods."""

    offset: tuple[int, int]
    sources: list[str]


class SentenceRecord(TypedDict):
    """Serialized example or quotation."""

    text: str
    word_offsets: NotRequired[list[OffsetRecord]]
    reference: NotRequired[str]
    year: NotRequired[int]


class AlignmentRecord(TypedDict):
    """Serialized WordNet identifier and directed relation."""

    synset_id: str
    relation: str


class TranslationTableRecord(TypedDict):
    """Serialized collected translation table."""

    id: str
    gloss: str
    translations: dict[str, list[str]]


class SenseRecord(TypedDict):
    """Serialized collected sense with optional aligned resources."""

    id: str
    glosses: list[str]
    etymology: NotRequired[str]
    synonyms: NotRequired[list[str]]
    topics: NotRequired[list[str]]
    tags: NotRequired[list[str]]
    sentences: NotRequired[list[SentenceRecord]]
    translation_table: NotRequired[TranslationTableRecord]
    wikidata_ids: NotRequired[list[str]]
    wordnet: NotRequired[list[AlignmentRecord]]


class LemmaRecord(TypedDict):
    """Serialized collected lemma."""

    id: str
    lemma: str
    pos: str
    variants: NotRequired[list[str]]
    senses: NotRequired[list[SenseRecord]]
    translation_tables: NotRequired[list[TranslationTableRecord]]


def _parse_sentence(
    record: SentenceRecord,
) -> Sentence:
    """
    Reconstruct an attestation with its original offsets and provenance.

    Args:
        record: Serialized attestation.

    Returns:
        The corresponding example or quotation.
    """
    offsets = tuple(
        WordOffset(
            (offset["offset"][0], offset["offset"][1]),
            tuple(WordOffsetSource(source) for source in offset["sources"]),
        )
        for offset in record.get("word_offsets", [])
    )

    if "reference" in record:
        return Quotation(
            record["text"],
            record["reference"],
            record.get("year"),
            word_offsets=offsets,
        )

    return Example(
        record["text"],
        word_offsets=offsets,
    )


def parse_lemma(
    record: LemmaRecord,
) -> Lemma:
    """
    Restore the domain model from a collected JSON object.

    Args:
        record: Serialized collection entry.

    Returns:
        Entry preserving collected and previously aligned fields.
    """
    return Lemma(
        record["id"],
        record["lemma"],
        POS(record["pos"]),
        variants=frozenset(record.get("variants", [])),
        senses=[
            Sense(
                sense["id"],
                tuple(sense["glosses"]),
                etymology=sense.get("etymology", ""),
                synonyms=tuple(sense.get("synonyms", [])),
                topics=tuple(sense.get("topics", [])),
                tags=tuple(sense.get("tags", [])),
                sentences=[
                    _parse_sentence(item) for item in sense.get("sentences", [])
                ],
                translation_table=(
                    TranslationTable(
                        table["id"],
                        table["gloss"],
                        {
                            language: frozenset(words)
                            for language, words in table["translations"].items()
                        },
                    )
                    if (table := sense.get("translation_table")) is not None
                    else None
                ),
                wikidata_ids=tuple(sense.get("wikidata_ids", [])),
                wordnet=tuple(
                    WordNetAlignment(
                        item["synset_id"], WordNetRelation(item["relation"])
                    )
                    for item in sense.get("wordnet", [])
                ),
            )
            for sense in record.get("senses", [])
        ],
        translation_tables=tuple(
            TranslationTable(
                table["id"],
                table["gloss"],
                {
                    language: frozenset(words)
                    for language, words in table["translations"].items()
                },
            )
            for table in record.get("translation_tables", [])
        ),
    )


def read_lemmas(
    path: Path,
) -> Iterator[Lemma]:
    """
    Stream collected entries without loading the corpus into memory.

    Args:
        path: Collected JSONL file, optionally compressed.

    Yields:
        Reconstructed collection entries.
    """
    with open_compressed(path, "rt") as stream:
        for line in stream:
            yield parse_lemma(cast(LemmaRecord, json.loads(line)))
