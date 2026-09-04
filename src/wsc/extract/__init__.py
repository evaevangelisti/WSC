"""
Reading of source dictionaries into models.
"""

from .offsets import build_query, find_word_offsets, match_forms, open_locator
from .wiktionary import WiktionaryExtractor
from .wordnet import WordNetExtractor

__all__ = [
    "WiktionaryExtractor",
    "WordNetExtractor",
    "build_query",
    "find_word_offsets",
    "match_forms",
    "open_locator",
]
