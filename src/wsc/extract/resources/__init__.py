"""
The reader for each source, one module apiece.
"""

from .wiktionary import WiktionaryExtractor
from .wordnet import WordNetExtractor

__all__ = [
    "WiktionaryExtractor",
    "WordNetExtractor",
]
