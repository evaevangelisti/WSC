"""
Prepare supplemental translations and explicit Wikidata sense identifiers.
"""

from collections.abc import Iterator
from logging import getLogger
from pathlib import Path

from ...constants import LANGUAGE_SECTION
from ..wiktionary import (
    build_off_page_translations,
    index_translation_glosses,
    read_entries,
    write_off_page_translations,
)
from .extractor import DumpExtractor
from .pages import read_pages
from .wikidata import (
    DumpDefinition,
    index_wikidata_ids,
    read_page_wikidata,
    write_wikidata_ids,
)

_LOGGER = getLogger(__name__)


def extract_dump_resources(
    dump_path: Path,
    output_path: Path,
    off_page_translations_path: Path,
    wikidata_ids_path: Path,
    *,
    refresh: bool,
) -> None:
    """
    Read supplemental translations and explicit IDs in one dump pass.

    Args:
        dump_path: Source Wikitext dump.
        output_path: Parsed Wiktextract entries.
        off_page_translations_path: Cache for supplemental translations.
        wikidata_ids_path: Cache for explicit Wikidata sense identifiers.
        refresh: Whether Wiktextract entries were regenerated.
    """
    definitions: list[DumpDefinition] = []
    extract_identifiers = refresh or not wikidata_ids_path.exists()

    def scanned_pages() -> Iterator[tuple[str, str]]:
        """
        Read dump pages while retaining explicit Wikidata definitions.

        Yields:
            Each page title and its markup.
        """
        for title, markup in read_pages(dump_path):
            if extract_identifiers and "{{senseid" in markup.casefold():
                definitions.extend(read_page_wikidata(title, markup, LANGUAGE_SECTION))

            yield title, markup

    if refresh or not off_page_translations_path.exists():
        parsed_glosses = index_translation_glosses(
            read_entries(output_path, "Indexing translation tables"),
        )

        off_page_translations = build_off_page_translations(
            DumpExtractor(LANGUAGE_SECTION).extract_pages(
                scanned_pages(),
                parsed_glosses,
            ),
            read_entries(output_path, "Answering the pointers"),
        )

        write_off_page_translations(off_page_translations_path, off_page_translations)

        _LOGGER.info(
            "Parsed %s: %s off-page translation entries",
            output_path,
            len(off_page_translations),
        )
    else:
        for _ in scanned_pages():
            pass

    if extract_identifiers:
        identifiers, unmatched = index_wikidata_ids(
            definitions,
            read_entries(output_path, "Indexing Wikidata IDs"),
        )

        write_wikidata_ids(wikidata_ids_path, identifiers)

        _LOGGER.info(
            "Parsed %s: %s Wikidata-linked senses",
            output_path,
            len(identifiers),
        )

        if unmatched:
            _LOGGER.warning("Could not match %s dump sense IDs", unmatched)
