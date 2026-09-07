"""Reading of what wiktextract made of a Wiktionary dump."""

from .entries import read_entries
from .extractor import WiktionaryExtractor
from .off_page_translations import (
    OffPageTranslations,
    build_off_page_translations,
    read_off_page_translations,
    write_off_page_translations,
)
from .schema import narrow

__all__ = [
    "OffPageTranslations",
    "WiktionaryExtractor",
    "build_off_page_translations",
    "narrow",
    "read_entries",
    "read_off_page_translations",
    "write_off_page_translations",
]
