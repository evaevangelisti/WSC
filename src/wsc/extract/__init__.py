"""Reading of source dictionaries into models."""

from .dump import DumpExtractor
from .offsets import build_query, find_word_offsets, match_forms, open_locator
from .wiktionary import (
    OffPageTranslations,
    WiktionaryExtractor,
    build_off_page_translations,
    narrow,
    read_entries,
    read_off_page_translations,
    write_off_page_translations,
)
from .wordnet import WordNetExtractor

__all__ = [
    "DumpExtractor",
    "OffPageTranslations",
    "WiktionaryExtractor",
    "WordNetExtractor",
    "build_off_page_translations",
    "build_query",
    "find_word_offsets",
    "match_forms",
    "narrow",
    "open_locator",
    "read_entries",
    "read_off_page_translations",
    "write_off_page_translations",
]
