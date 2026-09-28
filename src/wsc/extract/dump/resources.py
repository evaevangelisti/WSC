"""
Prepare supplemental translations, Wikidata identifiers, and source markup.
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
from .source_markup import (
    MarkupIndex,
    read_markup_index,
    read_page_markup,
    write_markup_index,
)
from .wikidata import (
    DumpDefinition,
    index_wikidata_ids,
    read_page_wikidata,
    write_wikidata_ids,
)

_LOGGER = getLogger(__name__)


def _scanned_pages(
    dump_path: Path,
    definitions: list[DumpDefinition],
    markup_index: MarkupIndex,
    *,
    extract_identifiers: bool,
    extract_markup: bool,
) -> Iterator[tuple[str, str]]:
    """
    Read dump pages while retaining requested source fragments.

    Args:
        dump_path: Source Wikitext dump.
        definitions: Destination for explicit Wikidata definitions.
        markup_index: Destination for mathematical and score source fragments.
        extract_identifiers: Whether to index Wikidata definitions.
        extract_markup: Whether to index source tags.

    Yields:
        Each page title and its markup.
    """
    for title, markup in read_pages(dump_path):
        folded = markup.casefold() if extract_identifiers or extract_markup else ""

        if extract_identifiers and "{{senseid" in folded:
            definitions.extend(read_page_wikidata(title, markup, LANGUAGE_SECTION))

        if extract_markup and ("<math" in folded or "<score" in folded):
            page = read_page_markup(markup, LANGUAGE_SECTION)

            if page is not None:
                markup_index[title] = page

        yield title, markup


def extract_dump_resources(
    dump_path: Path,
    output_path: Path,
    off_page_translations_path: Path,
    wikidata_ids_path: Path,
    markup_index_path: Path,
    *,
    refresh: bool,
) -> None:
    """
    Read supplemental translations, explicit IDs, and markup in one dump pass.

    Args:
        dump_path: Source Wikitext dump.
        output_path: Parsed Wiktextract entries.
        off_page_translations_path: Cache for supplemental translations.
        wikidata_ids_path: Cache for explicit Wikidata sense identifiers.
        markup_index_path: Cache for mathematical source and score contexts.
        refresh: Whether Wiktextract entries were regenerated.
    """
    definitions: list[DumpDefinition] = []

    markup_index: MarkupIndex = {}

    extract_identifiers = refresh or not wikidata_ids_path.exists()
    extract_markup = refresh or not markup_index_path.exists()

    pages = _scanned_pages(
        dump_path,
        definitions,
        markup_index,
        extract_identifiers=extract_identifiers,
        extract_markup=extract_markup,
    )

    if refresh or not off_page_translations_path.exists():
        parsed_glosses = index_translation_glosses(
            read_entries(output_path, "Indexing translation tables"),
        )

        off_page_translations = build_off_page_translations(
            DumpExtractor(LANGUAGE_SECTION).extract_pages(
                pages,
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
        for _ in pages:
            pass

    if extract_identifiers:
        if not extract_markup:
            markup_index = read_markup_index(markup_index_path)

        identifiers, unmatched = index_wikidata_ids(
            definitions,
            read_entries(output_path, "Indexing Wikidata IDs"),
            markup_index,
        )

        write_wikidata_ids(wikidata_ids_path, identifiers)

        _LOGGER.info(
            "Parsed %s: %s Wikidata-linked senses",
            output_path,
            len(identifiers),
        )

        if unmatched:
            _LOGGER.warning(
                "Could not match %s of %s dump definitions with Wikidata IDs",
                unmatched,
                len(definitions),
            )

    if extract_markup:
        write_markup_index(markup_index_path, markup_index)

        _LOGGER.info("Indexed source markup on %s pages", len(markup_index))
