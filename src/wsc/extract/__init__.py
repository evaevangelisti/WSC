"""
Reading of source dictionaries into models.
"""

from .offsets import find_word_offsets
from .wiktionary import WiktionaryExtractor

__all__ = [
    "WiktionaryExtractor",
    "find_word_offsets",
]
