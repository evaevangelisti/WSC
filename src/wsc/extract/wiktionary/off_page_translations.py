"""
The translations Wiktionary keeps away from the entry.

Tables move to a subpage; a shared meaning is pointed at.
"""

import json
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import TypedDict, cast

from ...constants import LANGUAGE
from ...files import open_compressed
from ...identifiers import lemma_id
from ...models import Language, TranslationTable
from ..dump.translations import PageTranslations
from ..translations import clean_translations, translation_gloss_key
from .merge import add_translations
from .parts import parse_translations
from .schema import RawEntry, parse_pos

type OffPageTranslations = dict[str, tuple[TranslationTable, ...]]
"""
What each entry is translated by elsewhere, by the name of the entry.
"""

type TranslationGlosses = dict[str, frozenset[str]]
"""
Normalized translation glosses already supplied for each entry.
"""


class _TranslationRecord(TypedDict):
    """
    Serialized language and its translated words.
    """

    language: dict[str, str]
    words: list[str]


def index_translation_glosses(
    entries: Iterable[RawEntry],
) -> TranslationGlosses:
    """
    Index the translation tables already supplied by Wiktextract.

    Args:
        entries: Parsed entries to inspect.

    Returns:
        Normalized translation glosses grouped by entry identifier.
    """
    indexed: dict[str, set[str]] = {}
    incomplete: dict[str, set[str]] = {}

    for entry in entries:
        if entry.get("lang_code") != LANGUAGE:
            continue

        word = entry.get("word", "").strip()

        if not word:
            continue

        try:
            pos = parse_pos(entry.get("pos", ""))
        except ValueError:
            continue

        entry_id = lemma_id(word, pos)

        for translation in entry.get("translations", []):
            written = translation.get("word", "")

            if (
                any(marker in written for marker in ("[[", "]]", "{{", "}}"))
                and clean_translations(translation.get("lang_code", ""), written)
                is None
            ):
                incomplete.setdefault(entry_id, set()).add(
                    translation_gloss_key(translation.get("sense", "")),
                )

        for table in parse_translations(entry.get("translations", [])):
            indexed.setdefault(entry_id, set()).add(
                translation_gloss_key(table.gloss),
            )

    return {
        entry_id: frozenset(glosses - incomplete.get(entry_id, set()))
        for entry_id, glosses in indexed.items()
    }


def _label_tables(
    tables: tuple[TranslationTable, ...],
    language_labels: dict[str, str],
) -> tuple[TranslationTable, ...]:
    """
    Add Wiktextract language labels to dump-only translation tables.

    Args:
        tables: Tables whose codes may lack a language name.
        language_labels: Names observed in Wiktextract translations.

    Returns:
        Tables with available language labels attached.
    """
    return tuple(
        replace(
            table,
            translations={
                Language(
                    language.code,
                    language.label or language_labels.get(language.code, ""),
                ): words
                for language, words in table.translations.items()
            },
        )
        for table in tables
    )


def _read_pointed_translations(
    entries: Iterable[RawEntry],
    pointed_ids: set[str],
) -> tuple[dict[str, tuple[TranslationTable, ...]], dict[str, str]]:
    """
    Read the translations of the entries some pointer names.

    Tables retain their gloss while entries from separate etymologies are merged.

    Args:
        entries: The parsed extraction, walked once.
        pointed_ids: What names the entries worth reading.

    Returns:
        Translation tables by entry and language labels seen in Wiktextract.
    """
    pointed_translations: dict[str, tuple[TranslationTable, ...]] = {}
    language_labels: dict[str, str] = {}

    for entry in entries:
        if entry.get("lang_code") != LANGUAGE:
            continue

        for translation in entry.get("translations", []):
            if (code := translation.get("lang_code")) and (
                label := translation.get("lang")
            ):
                _ = language_labels.setdefault(code, label)

        try:
            pos = parse_pos(entry.get("pos", ""))
        except ValueError:
            continue

        pointed_id = lemma_id(entry.get("word", "").strip(), pos)

        if pointed_id not in pointed_ids:
            continue

        pointed_tables = parse_translations(
            entry.get("translations", []),
            lemma_id=pointed_id,
        )

        pointed_translations[pointed_id] = add_translations(
            pointed_translations.get(pointed_id, ()),
            pointed_tables,
            pointed_id,
        )

    return (
        {
            entry_id: _label_tables(tables, language_labels)
            for entry_id, tables in pointed_translations.items()
        },
        language_labels,
    )


def build_off_page_translations(
    page_translations: Iterable[PageTranslations],
    entries: Iterable[RawEntry],
) -> OffPageTranslations:
    """
    Settle what each entry is translated by elsewhere.

    A pointer is answered here, leaving collecting a table to look up.

    Args:
        page_translations: What a walk of the dump found.
        entries: The parsed extraction, walked once to answer the pointers.

    Returns:
        The tables to add to each entry, by the name of the entry.
    """
    translated_pages = list(page_translations)

    pointed_ids = {
        lemma_id(pointed_lemma, page.pos)
        for page in translated_pages
        for pointed_lemmas in page.pointers.values()
        for pointed_lemma in pointed_lemmas
    }

    pointed_translations, language_labels = _read_pointed_translations(
        entries,
        pointed_ids,
    )

    translated_pages = [
        replace(
            page,
            translations=_label_tables(page.translations, language_labels),
        )
        for page in translated_pages
    ]

    for page in translated_pages:
        pointed_id = lemma_id(page.lemma, page.pos)

        if pointed_id not in pointed_ids:
            continue

        pointed_translations[pointed_id] = add_translations(
            pointed_translations.get(pointed_id, ()),
            page.translations,
            pointed_id,
        )

    off_page_translations: OffPageTranslations = {}

    for page in translated_pages:
        page_id = lemma_id(page.lemma, page.pos)
        translation_tables = off_page_translations.setdefault(page_id, ())

        translation_tables = add_translations(
            translation_tables,
            page.translations,
            page_id,
        )

        for gloss, pointed_lemmas in page.pointers.items():
            matching_translation_tables = tuple(
                TranslationTable(
                    table.id,
                    gloss,
                    table.translations,
                )
                for pointed_lemma in pointed_lemmas
                for table in pointed_translations.get(
                    lemma_id(pointed_lemma, page.pos),
                    (),
                )
                if translation_gloss_key(table.gloss) == translation_gloss_key(gloss)
            )

            if matching_translation_tables:
                translation_tables = add_translations(
                    translation_tables,
                    matching_translation_tables,
                    page_id,
                )

        off_page_translations[page_id] = translation_tables

    return {
        entry_id: translation_tables
        for entry_id, translation_tables in off_page_translations.items()
        if translation_tables
    }


def write_off_page_translations(
    output_path: Path,
    off_page: OffPageTranslations,
) -> None:
    """
    Write what was settled, as one object naming every entry.

    Args:
        output_path: Where the file is placed.
        off_page: The tables to add to each entry.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = {
        entry_id: [
            {
                "id": table.id,
                "gloss": table.gloss,
                "translations": [
                    {
                        "language": {
                            "code": language.code,
                            "label": language.label,
                        },
                        "words": sorted(words),
                    }
                    for language, words in table.translations.items()
                ],
            }
            for table in tables
        ]
        for entry_id, tables in sorted(off_page.items())
    }

    with open_compressed(output_path, "wt") as stream:
        _ = stream.write(json.dumps(written, ensure_ascii=False))


def _read_table_translations(
    table: dict[str, object],
) -> dict[Language, frozenset[str]]:
    """
    Read labeled translations or the preceding code-indexed cache format.

    Args:
        table: Serialized supplemental translation table.

    Returns:
        Words grouped by their language objects.
    """
    serialized = table["translations"]

    if isinstance(serialized, dict):
        labels = {
            language["code"]: language["label"]
            for language in cast(list[dict[str, str]], table.get("languages", []))
        }

        return {
            Language(code, labels.get(code, "")): frozenset(words)
            for code, words in cast(dict[str, list[str]], serialized).items()
        }

    return {
        Language(
            record["language"]["code"],
            record["language"].get("label", ""),
        ): frozenset(record["words"])
        for record in cast(list[_TranslationRecord], serialized)
    }


def read_off_page_translations(
    input_path: Path,
) -> OffPageTranslations:
    """
    Read back what a walk of the dump settled.

    Args:
        input_path: The file the walk was written to.

    Returns:
        The tables to add to each entry, by the name of the entry.
    """
    with open_compressed(input_path, "rt") as stream:
        read = cast(dict[str, list[dict[str, object]]], json.load(stream))

    return {
        entry_id: tuple(
            TranslationTable(
                str(table["id"]),
                str(table["gloss"]),
                _read_table_translations(table),
            )
            for table in tables
        )
        for entry_id, tables in read.items()
    }
