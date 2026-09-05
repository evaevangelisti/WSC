"""
The translations Wiktionary keeps away from the entry.

Tables move to a subpage; a shared meaning is pointed at.
"""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import cast

from ...models import POS, Translations
from ..dump import PageTranslations
from .identifiers import lemma_id
from .merge import add_translations
from .parts import parse_translations
from .schema import RawEntry

type OffPageTranslations = dict[str, Translations]
"""What each entry is translated by elsewhere, by the name of the entry."""


def _flatten_translations(
    translations: Translations,
) -> dict[str, frozenset[str]]:
    """
    Gather every table of an entry into one, whatever gloss headed it.

    A pointer names a headword, not a table.

    Args:
        translations: What the entry pointed at carries.

    Returns:
        The words each language offers for it.
    """
    flattened_words: dict[str, frozenset[str]] = {}

    for translated_words in translations.values():
        for language, words in translated_words.items():
            flattened_words[language] = (
                flattened_words.get(language, frozenset()) | words
            )

    return flattened_words


def _read_pointed_translations(
    entries: Iterable[RawEntry],
    pointed_ids: set[str],
) -> dict[str, dict[str, frozenset[str]]]:
    """
    Read the translations of the entries some pointer names.

    One name gathers every etymology, which wiktextract writes as an entry
    apiece.

    Args:
        entries: The parsed extraction, walked once.
        pointed_ids: What names the entries worth reading.

    Returns:
        The words each language offers for an entry, by the name of the entry.
    """
    pointed_translations: dict[str, dict[str, frozenset[str]]] = {}

    for entry in entries:
        try:
            pos = POS(entry.get("pos", ""))
        except ValueError:
            continue

        pointed_id = lemma_id(entry.get("word", "").strip(), pos)
        if pointed_id not in pointed_ids:
            continue

        kept_words = pointed_translations.setdefault(pointed_id, {})
        pointed_tables = parse_translations(entry.get("translations", []))

        for language, words in _flatten_translations(pointed_tables).items():
            kept_words[language] = kept_words.get(language, frozenset()) | words

    return pointed_translations


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

    pointed_translations = _read_pointed_translations(entries, pointed_ids)

    off_page_translations: OffPageTranslations = {}

    for page in translated_pages:
        page_id = lemma_id(page.lemma, page.pos)
        entry_translations = off_page_translations.setdefault(page_id, {})

        add_translations(entry_translations, page.translations)

        for gloss, pointed_lemmas in page.pointers.items():
            for pointed_lemma in pointed_lemmas:
                pointed_words = pointed_translations.get(
                    lemma_id(pointed_lemma, page.pos)
                )

                if pointed_words:
                    add_translations(entry_translations, {gloss: pointed_words})

    return {
        entry_id: tables for entry_id, tables in off_page_translations.items() if tables
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
        entry_id: {
            gloss: {
                language: sorted(words) for language, words in translated_words.items()
            }
            for gloss, translated_words in tables.items()
        }
        for entry_id, tables in sorted(off_page.items())
    }

    _ = output_path.write_text(
        json.dumps(written, ensure_ascii=False), encoding="utf-8"
    )


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
    read = cast(
        dict[str, dict[str, dict[str, list[str]]]],
        json.loads(input_path.read_text(encoding="utf-8")),
    )

    return {
        entry_id: {
            gloss: {
                language: frozenset(words)
                for language, words in translated_words.items()
            }
            for gloss, translated_words in tables.items()
        }
        for entry_id, tables in read.items()
    }
