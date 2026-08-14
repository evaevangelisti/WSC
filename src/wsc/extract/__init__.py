"""
Reading of source dictionaries into models.
"""

from .offsets import find_word_offsets
from .resources import WiktionaryExtractor, WordNetExtractor

__all__ = [
    "WiktionaryExtractor",
    "WordNetExtractor",
    "find_word_offsets",
]
