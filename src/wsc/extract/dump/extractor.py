"""Extraction of the translations wiktextract leaves behind."""

from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path

from ...identifiers import lemma_id
from ..translations import translation_gloss_key
from .pages import read_pages
from .translations import SUBPAGE_SUFFIX, PageTranslations, read_page

_POINTER = "{{trans-see"
_POINTER_IN_TABLE = "{{trans-top-see"
_TABLE = "{{trans-top"


class DumpExtractor:
    """
    Reads translation tables Wiktextract leaves behind.

    Cross-page tables resolve pointers; direct tables fill gaps in the parsed source.
    """

    def __init__(
        self,
        language_section: str,
    ) -> None:
        """
        Set which language's sections are read.

        Args:
            language_section: Language section heading, such as English.
        """
        self._language_section: str = language_section

    def extract(
        self,
        input_path: Path,
        parsed_glosses: Mapping[str, frozenset[str]] | None = None,
    ) -> Iterator[PageTranslations]:
        """
        Read dump translations missing from the parsed entries.

        Args:
            input_path: The dump to read, compressed or not.
            parsed_glosses: Table glosses already supplied by Wiktextract.

        Yields:
            One record per part of speech a page translates elsewhere.
        """
        known_glosses = parsed_glosses or {}

        for title, markup in read_pages(input_path):
            if not (
                title.endswith(SUBPAGE_SUFFIX)
                or _POINTER in markup
                or _POINTER_IN_TABLE in markup
                or (parsed_glosses is not None and _TABLE in markup)
            ):
                continue

            for page in read_page(title, markup, self._language_section):
                extracted_page = page

                if not title.endswith(SUBPAGE_SUFFIX):
                    parsed_table_glosses = known_glosses.get(
                        lemma_id(page.lemma, page.pos),
                        frozenset(),
                    )

                    extracted_page = replace(
                        page,
                        translations=(
                            tuple(
                                table
                                for table in page.translations
                                if translation_gloss_key(table.gloss)
                                not in parsed_table_glosses
                            )
                            if parsed_glosses is not None
                            else ()
                        ),
                    )

                if extracted_page.translations or extracted_page.pointers:
                    yield extracted_page
