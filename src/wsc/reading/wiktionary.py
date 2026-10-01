"""
Read collected JSONL records without repeating extraction.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from ..archives import archive_lines
from ..files import open_compressed
from ..models import (
    POS,
    Example,
    Language,
    Lemma,
    Quotation,
    Sense,
    Sentence,
    SynsetAlignment,
    SynsetRelation,
    TranslationTable,
    WordOffset,
    WordOffsetSource,
)


class OffsetRecord(TypedDict):
    """
    Serialized candidate range and its proposing methods.
    """

    offset: tuple[int, int]
    sources: list[str]


class SentenceRecord(TypedDict):
    """
    Serialized example or quotation.
    """

    text: str
    word_offsets: NotRequired[list[OffsetRecord]]
    reference: NotRequired[str]
    year: NotRequired[int]


class SynsetAlignmentRecord(TypedDict):
    """
    Serialized synset identifier and directed relation.
    """

    synset_id: str
    relation: str
    sources: NotRequired[list[str]]


class LanguageRecord(TypedDict):
    """
    Serialized translation language.
    """

    code: str
    label: NotRequired[str]


class TranslationRecord(TypedDict):
    """
    Serialized language and the words translated into it.
    """

    language: LanguageRecord
    words: list[str]


class TranslationTableRecord(TypedDict):
    """
    Serialized collected translation table.
    """

    id: str
    gloss: str
    translations: list[TranslationRecord] | dict[str, list[str]]
    languages: NotRequired[list[LanguageRecord]]


class SenseRecord(TypedDict):
    """
    Serialized collected sense with optional aligned resources.
    """

    id: str
    glosses: list[str]
    etymology: NotRequired[str]
    synonyms: NotRequired[list[str]]
    tags: NotRequired[list[str]]
    topics: NotRequired[list[str]]
    sentences: NotRequired[list[SentenceRecord]]
    translation_table: NotRequired[TranslationTableRecord]
    wikidata_ids: NotRequired[list[str]]
    synsets: NotRequired[list[SynsetAlignmentRecord]]


class LemmaRecord(TypedDict):
    """
    Serialized collected lemma.
    """

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


def _parse_translation_table(
    record: TranslationTableRecord,
) -> TranslationTable:
    """
    Restore translations from labeled or earlier code-indexed records.

    Args:
        record: Serialized translation table.

    Returns:
        Table indexed internally by language objects.
    """
    serialized = record["translations"]

    if isinstance(serialized, dict):
        labels = {
            language["code"]: language.get("label", "")
            for language in record.get("languages", [])
        }

        translations = {
            Language(code, labels.get(code, "")): frozenset(words)
            for code, words in serialized.items()
        }
    else:
        translations = {
            Language(
                item["language"]["code"],
                item["language"].get("label", ""),
            ): frozenset(item["words"])
            for item in serialized
        }

    return TranslationTable(record["id"], record["gloss"], translations)


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
                tags=tuple(sense.get("tags", [])),
                topics=tuple(sense.get("topics", [])),
                sentences=[
                    _parse_sentence(item) for item in sense.get("sentences", [])
                ],
                translation_table=(
                    _parse_translation_table(table)
                    if (table := sense.get("translation_table")) is not None
                    else None
                ),
                wikidata_ids=tuple(sense.get("wikidata_ids", [])),
                synsets=tuple(
                    SynsetAlignment(
                        item["synset_id"],
                        SynsetRelation(item["relation"]),
                        tuple(item.get("sources", [])),
                    )
                    for item in sense.get("synsets", [])
                ),
            )
            for sense in record.get("senses", [])
        ],
        translation_tables=tuple(
            _parse_translation_table(table)
            for table in record.get("translation_tables", [])
        ),
    )


def _jsonl_lines(
    path: Path,
) -> Iterator[str]:
    """
    Read lines from a plain or individually compressed JSONL file.

    Args:
        path: Source JSONL file.

    Yields:
        Serialized JSONL lines.
    """
    with open_compressed(path, "rt") as stream:
        yield from stream


def read_lemmas(
    path: Path,
) -> Iterator[Lemma]:
    """
    Stream collected entries without loading the corpus into memory.

    Args:
        path: Collected JSONL file or compressed collection archive.

    Yields:
        Reconstructed collection entries.
    """
    if path.name.endswith((".tar.gz", ".tar.bz2", ".tar.zst")):
        from ..constants import COLLECTION_FILES

        lines = archive_lines(path, COLLECTION_FILES["senses"])
    else:
        lines = _jsonl_lines(path)

    for line in lines:
        yield parse_lemma(cast(LemmaRecord, json.loads(line)))
