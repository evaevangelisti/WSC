"""Extraction of the translations wiktextract leaves behind."""

from collections.abc import Iterator
from pathlib import Path

from .pages import read_pages
from .translations import SUBPAGE_SUFFIX, PageTranslations, read_page

_POINTER = "{{trans-see"
_POINTER_IN_TABLE = "{{trans-top-see"


class DumpExtractor:
    """
    Reads the translations Wiktionary keeps away from the entry.

    wiktextract reads a page at a time and misses both.
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
    ) -> Iterator[PageTranslations]:
        """
        Read the translations a dump keeps away from the entries.

        Args:
            input_path: The dump to read, compressed or not.

        Yields:
            One record per part of speech a page translates elsewhere.
        """
        for title, markup in read_pages(input_path):
            if not (
                title.endswith(SUBPAGE_SUFFIX)
                or _POINTER in markup
                or _POINTER_IN_TABLE in markup
            ):
                continue

            for page in read_page(title, markup, self._language_section):
                if page.translations or page.pointers:
                    yield page
