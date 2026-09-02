"""
Reading of a parsed Wiktionary dump into models.
"""

from .extractor import WiktionaryExtractor
from .offsets import build_query, find_word_offsets, match_forms, open_locator

__all__ = [
    "WiktionaryExtractor",
    "build_query",
    "find_word_offsets",
    "match_forms",
    "open_locator",
]
